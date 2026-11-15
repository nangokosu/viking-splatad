"""Exact radar ellipsoid support using an immutable sparse cell index on the model device.

Cells eliminate impossible pairs; every surviving pair still uses the inclusive
Mahalanobis predicate. There is no neighbor cap, support subsampling or CPU query
in the training-step path. No custom CUDA kernel or extra dependency is required.
"""

import numpy as np
from scipy.stats import chi2


class RadarSupportIndex:
    """Reusable anchor-cell CSR index, queried afresh at current Gaussian positions."""

    def __init__(self, means, covariances, gate_quantile: float, chunk_size: int, workers: int = 1):
        """Build conservative anchor AABB cell lists once on CPU and transfer their CSR arrays.

        Args:
            means, covariances: Validated torch anchors [S,3], [S,3,3].
            gate_quantile: Chi-square probability; chunk_size: pair-batch side budget.
            workers: Reserved for API compatibility; construction never uses hidden threads.
        """
        import torch

        if (not 0 < gate_quantile < 1 or isinstance(chunk_size, bool)
                or not isinstance(chunk_size, int) or chunk_size <= 0 or workers <= 0):
            raise ValueError("valid gate, integer chunk budget and CPU allocation required")
        self.means, self.covariances = means, covariances
        self.gate_quantile, self.chunk_size = gate_quantile, chunk_size
        self.threshold = float(chi2.ppf(gate_quantile, df=3))
        self.pair_budget = chunk_size * chunk_size
        with torch.no_grad():
            self.inverses = torch.linalg.inv(covariances)
        means_cpu = means.detach().cpu().numpy().astype(np.float64)
        covariance_cpu = covariances.detach().cpu().numpy().astype(np.float64)
        radii = np.sqrt(self.threshold * np.diagonal(covariance_cpu, axis1=1, axis2=2))
        if not len(means_cpu) or not np.isfinite(radii).all() or np.any(radii <= 0):
            raise ValueError("nonempty anchors with finite positive coordinate radii required")
        # The ellipsoid AABB has half-width sqrt(chi2 * Sigma_aa) on axis a.
        # A conservative machine-precision guard affects only candidate search,
        # never the final gate; it covers rounded float32 coordinate/inverse math.
        precision = np.finfo(means.detach().cpu().numpy().dtype).eps
        guard = np.sqrt(precision) * np.maximum(1., np.abs(means_cpu) + radii)
        lower, upper = means_cpu - radii - guard, means_cpu + radii + guard
        origin = lower.min(axis=0)
        cell_width = np.median(2 * radii, axis=0)
        # An exact index may be coarsened to bound its storage; this changes work,
        # not membership. With sufficiently wide cells every anchor has one cell.
        entry_budget = max(self.pair_budget, len(means_cpu))
        while True:
            first = np.floor((lower - origin) / cell_width).astype(np.int64)
            last = np.floor((upper - origin) / cell_width).astype(np.int64)
            per_axis_counts = last - first + 1
            counts = np.prod(per_axis_counts, axis=1, dtype=np.int64)
            if np.any(counts <= 0):
                raise ValueError("anchor cell index extent overflow")
            if int(counts.sum()) <= entry_budget:
                break
            # Dyadic coarsening is an indexing policy, not a radar uncertainty change.
            cell_width *= 2
        shape = last.max(axis=0) + 1
        if int(shape[0]) * int(shape[1]) * int(shape[2]) > np.iinfo(np.int64).max:
            raise ValueError("anchor cell keys exceed signed 64-bit index capacity")
        keys, seed_ids = [], []
        for seed_index in range(len(means_cpu)):
            cells = np.indices(tuple(per_axis_counts[seed_index]), dtype=np.int64).reshape(3, -1).T
            cells += first[seed_index]
            keys.append((cells[:, 0] * shape[1] + cells[:, 1]) * shape[2] + cells[:, 2])
            seed_ids.append(np.full(len(cells), seed_index, dtype=np.int64))
        keys, seed_ids = np.concatenate(keys), np.concatenate(seed_ids)
        order = np.argsort(keys, kind="stable")
        keys, seed_ids = keys[order], seed_ids[order]
        unique_keys, starts, cell_counts = np.unique(keys, return_index=True, return_counts=True)
        offsets = np.concatenate([starts, [len(keys)]])
        device = means.device
        self.cell_keys = torch.as_tensor(unique_keys, device=device)
        self.cell_offsets = torch.as_tensor(offsets, device=device)
        self.cell_seeds = torch.as_tensor(seed_ids, device=device)
        # Float64 cell arithmetic keeps lookup roundoff separate from the model's
        # float32 ellipsoid arithmetic. The guard above makes the lookup conservative.
        self.origin = torch.as_tensor(origin, device=device, dtype=torch.float64)
        self.cell_width = torch.as_tensor(cell_width, device=device, dtype=torch.float64)
        self.shape = torch.as_tensor(shape, device=device)
        self.maximum_cell_count = int(cell_counts.max())
        self.cell_entry_count = len(seed_ids)
        self.last_candidate_pairs = 0

    def check_targets(self, means, covariances, gate_quantile: float, chunk_size: int):
        """Reject reuse with different immutable targets or gate/budget settings; return None."""
        if (means is not self.means or covariances is not self.covariances
                or gate_quantile != self.gate_quantile or chunk_size != self.chunk_size):
            raise ValueError("radar support index must match its immutable target tensors and settings")

    def support_masks(self, gaussian_means):
        """Return exact Gaussian/anchor masks at current means using device-side ragged joins.

        Args:
            gaussian_means: Validated static means [G,3] on the target device.

        Returns:
            Boolean [G], [S]. Gaussian/candidate blocks bound temporary memory.
            A pathological cell can still contain all anchors; it is processed in
            pieces rather than truncated, so the dense reference remains the spec.
        """
        import torch

        device = gaussian_means.device
        supported = torch.zeros(len(gaussian_means), device=device, dtype=torch.bool)
        seed_supported = torch.zeros(len(self.means), device=device, dtype=torch.bool)
        rows_per_batch = max(1, self.pair_budget // self.maximum_cell_count)
        self.last_candidate_pairs = 0
        with torch.no_grad():
            for row_start in range(0, len(gaussian_means), rows_per_batch):
                row_stop = min(len(gaussian_means), row_start + rows_per_batch)
                points = gaussian_means[row_start:row_stop]
                cells = torch.floor((points.double() - self.origin) / self.cell_width).long()
                in_bounds = ((cells >= 0) & (cells < self.shape)).all(dim=1)
                keys = (cells[:, 0] * self.shape[1] + cells[:, 1]) * self.shape[2] + cells[:, 2]
                keys = torch.where(in_bounds, keys, -torch.ones_like(keys))
                positions = torch.searchsorted(self.cell_keys, keys).clamp(max=len(self.cell_keys) - 1)
                found = in_bounds & (self.cell_keys[positions] == keys)
                starts = self.cell_offsets[positions]
                counts = torch.where(found, self.cell_offsets[positions + 1] - starts, torch.zeros_like(starts))
                candidate_count = int(counts.sum().item())
                self.last_candidate_pairs += candidate_count
                if candidate_count == 0:
                    continue
                row_ids = torch.repeat_interleave(torch.arange(row_stop - row_start, device=device), counts)
                row_offsets = torch.cumsum(counts, dim=0) - counts
                within_row = torch.arange(candidate_count, device=device) - torch.repeat_interleave(row_offsets, counts)
                candidate_offsets = torch.repeat_interleave(starts, counts) + within_row
                candidate_seeds = self.cell_seeds[candidate_offsets]
                for pair_start in range(0, candidate_count, self.pair_budget):
                    selected = slice(pair_start, pair_start + self.pair_budget)
                    gaussian_ids = row_start + row_ids[selected]
                    seed_ids = candidate_seeds[selected]
                    deltas = gaussian_means[gaussian_ids] - self.means[seed_ids]
                    distances = torch.einsum("ni,nij,nj->n", deltas, self.inverses[seed_ids], deltas)
                    inside = distances <= self.threshold
                    supported[gaussian_ids[inside]] = True
                    seed_supported[seed_ids[inside]] = True
        return supported, seed_supported
