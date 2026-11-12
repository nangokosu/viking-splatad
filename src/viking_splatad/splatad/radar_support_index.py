"""Exact radar ellipsoid support using immutable radius-bucketed CPU spatial indexes.

The tree only eliminates impossible pairs. The final inclusive Mahalanobis test
still runs on the model's device/dtype, and Gaussians are queried at every step.
"""

import math

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import chi2


class RadarSupportIndex:
    """Reusable anchor index with bounded candidate batches and no learned state."""

    def __init__(self, means, covariances, gate_quantile: float, chunk_size: int, workers: int = 1):
        """Cache immutable targets and conservative per-radius trees.

        Args:
            means, covariances: Validated torch anchors [S,3], [S,3,3].
            gate_quantile: Inclusive chi-square probability; chunk_size: pair-batch side budget.
            workers: CPU threads from the actual Slurm allocation; one outside Slurm.
        """
        import torch

        if not 0 < gate_quantile < 1 or chunk_size <= 0 or workers <= 0:
            raise ValueError("valid gate, chunk budget and CPU allocation required")
        self.means, self.covariances = means, covariances
        self.gate_quantile, self.chunk_size, self.workers = gate_quantile, chunk_size, workers
        self.threshold = float(chi2.ppf(gate_quantile, df=3))
        self.pair_budget = chunk_size * chunk_size
        with torch.no_grad():
            self.inverses = torch.linalg.inv(covariances)
        means_cpu = means.detach().cpu().numpy().astype(np.float64)
        covariance_cpu = covariances.detach().cpu().numpy().astype(np.float64)
        # For SPD Sigma, lambda_max <= trace(Sigma). This sphere contains the
        # whole ellipsoid; it cannot drop a valid pair. Powers-of-two buckets keep
        # one unusually broad target from expanding every other target's search.
        radii = np.sqrt(self.threshold * np.trace(covariance_cpu, axis1=1, axis2=2))
        if not np.isfinite(radii).all() or np.any(radii <= 0):
            raise ValueError("finite positive anchor search radii required")
        bucket_ids = np.floor(np.log2(radii)).astype(np.int64)
        self.buckets = []
        for bucket_id in np.unique(bucket_ids):
            indices = np.flatnonzero(bucket_ids == bucket_id)
            radius = np.nextafter(radii[indices].max(), math.inf)
            self.buckets.append((cKDTree(means_cpu[indices]), indices, radius))
        self.last_candidate_pairs = 0

    def check_targets(self, means, covariances, gate_quantile: float, chunk_size: int):
        """Reject reuse with different tensors or gate/budget settings; return None."""
        if (means is not self.means or covariances is not self.covariances
                or gate_quantile != self.gate_quantile or chunk_size != self.chunk_size):
            raise ValueError("radar support index must match its immutable target tensors and settings")

    def support_masks(self, gaussian_means):
        """Return exact Gaussian/anchor masks at the current means with bounded temporary memory.

        Args:
            gaussian_means: Validated static Gaussian means [G,3] on the target device.

        Returns:
            Boolean [G], [S]. Spatial queries are exact (eps=0), then the same
            ellipsoid predicate is evaluated for all candidate pairs. Degenerate
            dense scenes can still require O(G*S) work; no k-neighbor cap truncates support.
        """
        import torch

        supported = torch.zeros(len(gaussian_means), device=gaussian_means.device, dtype=torch.bool)
        seed_supported = torch.zeros(len(self.means), device=gaussian_means.device, dtype=torch.bool)
        points_cpu = gaussian_means.detach().cpu().numpy()
        self.last_candidate_pairs = 0
        # A count pass establishes the exact maximum neighbor count before a
        # rectangular query. No Python list is allocated per Gaussian and no
        # fixed k cap can drop a valid neighbor.
        query_budget = max(self.chunk_size, self.pair_budget)
        with torch.no_grad():
            for tree, seed_indices, radius in self.buckets:
                for query_start in range(0, len(points_cpu), query_budget):
                    query = points_cpu[query_start:query_start + query_budget]
                    counts = tree.query_ball_point(query, radius, eps=0., workers=self.workers, return_length=True)
                    self.last_candidate_pairs += int(np.sum(counts))
                    maximum_count = int(np.max(counts, initial=0))
                    if maximum_count == 0:
                        continue
                    rows_per_batch = max(1, self.pair_budget // maximum_count)
                    for row_start in range(0, len(query), rows_per_batch):
                        row_stop = min(len(query), row_start + rows_per_batch)
                        neighbor_count = int(np.max(counts[row_start:row_stop], initial=0))
                        if neighbor_count == 0:
                            continue
                        _, neighbors = tree.query(query[row_start:row_stop], k=neighbor_count,
                                                  distance_upper_bound=radius, eps=0., workers=self.workers)
                        neighbors = np.asarray(neighbors).reshape(row_stop - row_start, neighbor_count)
                        valid = neighbors < tree.n
                        gaussian_indices = np.broadcast_to(np.arange(query_start + row_start,
                            query_start + row_stop)[:, None], neighbors.shape)[valid]
                        local_seed_indices = neighbors[valid]
                        candidate_seeds = seed_indices[local_seed_indices]
                        for pair_start in range(0, len(candidate_seeds), self.pair_budget):
                            pair_slice = slice(pair_start, pair_start + self.pair_budget)
                            gaussian_ids = torch.as_tensor(gaussian_indices[pair_slice], device=gaussian_means.device)
                            seed_ids = torch.as_tensor(candidate_seeds[pair_slice], device=gaussian_means.device)
                            deltas = gaussian_means[gaussian_ids] - self.means[seed_ids]
                            distance_squared = torch.einsum("ni,nij,nj->n", deltas, self.inverses[seed_ids], deltas)
                            inside = distance_squared <= self.threshold
                            # Duplicate writes all write True; overlapping anchors do
                            # not multiply the loss or overwrite a true mask with false.
                            supported[gaussian_ids[inside]] = True
                            seed_supported[seed_ids[inside]] = True
        return supported, seed_supported
