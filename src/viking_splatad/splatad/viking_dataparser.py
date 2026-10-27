"""neurad-studio dataparser for the Viking Hill intermediate dataset (T27, framework half).

Needs torch + a neurad-studio checkout patched by ``patches/apply_ouster_patch.py``. When
those are not importable (plain project venv) the module still imports, with
``NERFSTUDIO_AVAILABLE = False`` and no classes, so package-wide imports (traceability test)
keep working. The numeric conversion lives in ``viking_conversions`` (numpy only, T27).

Register with ns-train without editing neurad's config files::

    NERFSTUDIO_DATAPARSER_CONFIGS=viking-data=viking_splatad.splatad.viking_dataparser:VIKING_DATAPARSER_SPEC

(neurad's ``discover_dataparsers`` reads that variable; ``install_into_neurad.sh`` prints it).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np

from viking_splatad.dataset.run_split import (
    WHOLE_BLOCK_CORE_FRACTION,
    middle_block_train_eval_indices,
    restrict_test_to_block_cores,
)
from viking_splatad.splatad.viking_conversions import (
    OUSTER_LIDAR_TYPE_NAME,
    DatasetView,
    build_sensor_arrays,
    load_lidar_sweep,
    no_return_path_for_sweep,
    no_return_placeholder_points,
    open_dataset_view,
    resolve_train_fraction,
)

try:
    import torch

    from nerfstudio.cameras import lidars as lidars_module
    from nerfstudio.cameras.cameras import Cameras, CameraType
    from nerfstudio.cameras.lidars import LIDAR_MODEL_TO_TYPE, Lidars
    from nerfstudio.data.dataparsers.ad_dataparser import (
        DUMMY_DISTANCE_VALUE,
        ADDataParser,
        ADDataParserConfig,
    )
    from nerfstudio.plugins.registry_dataparser import DataParserSpecification

    NERFSTUDIO_AVAILABLE = True
except ImportError:  # plain project venv: no torch / neurad-studio
    NERFSTUDIO_AVAILABLE = False

if NERFSTUDIO_AVAILABLE:

    @dataclass
    class VikingHillDataParserConfig(ADDataParserConfig):
        """Config for the Viking Hill dataset written by ``dataset/writer.py``."""

        _target: Type = field(default_factory=lambda: VikingHillDataParser)
        data: Path = Path("data/intermediate")
        """Directory containing one sub-directory per session."""
        session_name: str = ""
        """Session directory name under ``data`` (also used as ``sequence``)."""
        cameras: Tuple[str, ...] = ("ids_camera",)
        """Camera stream names (plan: one camera called ids_camera)."""
        lidars: Tuple[str, ...] = ("ouster",)
        """Lidar stream names (plan: one lidar called ouster)."""
        load_cuboids: bool = False
        """Ground-truth cuboids never enter SplatAD inputs (project rule 6)."""
        annotation_interval: float = 0.1
        """Required > 0 by ADDataParserConfig but unused without cuboids; 0.1 s = OS0 sweep period."""
        train_split_fraction_override: Optional[float] = None
        """None = use the manifest's session train fraction; else this value."""
        test_block_count: Optional[int] = None
        """None = neurad's per-sensor LINSPACE split (every ~1/(1-f)-th frame is test). K = the test
        set is K contiguous time blocks, the same for every sensor, centred at k/(K+1) of the
        window (T60). Set from the session config's split.test_block_count by submit_run.py."""
        test_block_core_fraction: float = WHOLE_BLOCK_CORE_FRACTION
        """Fraction c of each T60 block, around its centre, that is the eval split (T62). The rest
        of the block is guard: not trained on, not evaluated or rendered. 1.0 = the whole block.
        Needs test_block_count. Set from the session config's split.test_block_core_fraction."""
        time_to_center_fraction_of_readout: float = 0.5
        """Centre row is exposed this fraction of the readout after the stamp: 0.5 if the stamp
        marks readout start (assumed; unverified), 0.0 if it marks mid-readout."""
        min_lidar_dist: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Base-class box test disabled (|v| >= 0 always holds): ego returns are removed by the
        manifest's radius in ``_read_lidars`` instead."""
        horizontal_beam_divergence: float = 3e-3
        """Radians at 1 m; same public default as the PandaSet/py123d parsers (OS0 value unmeasured)."""
        vertical_beam_divergence: float = 1.5e-3
        """Radians at 1 m; same public default as the PandaSet/py123d parsers (OS0 value unmeasured)."""

        def __post_init__(self) -> None:
            """Mirror ``session_name`` into ``sequence`` then run the base validation."""
            self.sequence = self.session_name
            super().__post_init__()

    @dataclass
    class VikingHillDataParser(ADDataParser):
        """Reads the intermediate dataset and feeds neurad's ``ADDataParser`` pipeline."""

        config: VikingHillDataParserConfig

        def __post_init__(self) -> None:
            """Initialise lazily filled state."""
            self._view: Optional[DatasetView] = None
            self._camera_arrays = None
            self._lidar_arrays = None
            # T60 positions per sensor kind ("camera"/"lidar" -> (train, test)), set after the window.
            self._block_split_positions: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None

        def _generate_dataparser_outputs(self, split="train"):
            """Open the dataset, register the lidar model, then run the base pipeline.

            Args:
                split: "train" or "eval".

            Returns:
                ``DataparserOutputs`` from ``ADDataParser``.
            """
            session_root = Path(self.config.data) / self.config.session_name
            if not self.config.session_name or not session_root.is_dir():
                raise FileNotFoundError(f"set session_name to an existing session dir: {session_root}")
            # 1) cheap view (no pixels), then numeric conversion (T27).
            self._view = open_dataset_view(session_root)
            self._camera_arrays, self._lidar_arrays = build_sensor_arrays(
                self._view, self.config.cameras, self.config.lidars,
                self.config.time_to_center_fraction_of_readout,
            )
            # 2) the split code in ADDataParser reads the fraction from the config, so write the
            #    resolved value there (manifest unless overridden).
            self.config.train_split_fraction = resolve_train_fraction(
                self._view, self.config.train_split_fraction_override
            )
            # 3) register the data-derived lidar model with the patched lidars module BEFORE the
            #    datamanager asks for it, and mirror it into the config fields used by
            #    _get_missing_points (add_missing_points).
            model = self._lidar_arrays.model
            if model.type_name == OUSTER_LIDAR_TYPE_NAME:
                if not hasattr(lidars_module, "register_lidar_model"):
                    raise RuntimeError("neurad-studio is not patched; run install_into_neurad.sh")
                lidars_module.register_lidar_model(
                    LIDAR_MODEL_TO_TYPE[model.type_name],
                    model.elevation_mapping_degrees,
                    model.azimuth_resolution_degrees,
                    model.revolution_time_seconds,
                )
            for lidar_name in self.config.lidars:
                self.config.lidar_elevation_mapping = {
                    **(self.config.lidar_elevation_mapping or {}), lidar_name: model.elevation_mapping_degrees}
                self.config.lidar_azimuth_resolution = {
                    **(self.config.lidar_azimuth_resolution or {}), lidar_name: model.azimuth_resolution_degrees}
                # No channel is skipped: every ring is a real beam.
                self.config.skip_elevation_channels = {
                    **(self.config.skip_elevation_channels or {}), lidar_name: ()}
            return super()._generate_dataparser_outputs(split)

        def _filter_based_on_time(self, cameras, img_filenames, lidars, pc_filenames, trajectories):
            """Base time window, then the T60 block split (and T62 cores) on the windowed float64 times.

            The blocks are computed here, not in ``_get_train_eval_indices``: that runs after
            ``_adjust_times`` has cast times to float32 and sees one sensor kind at a time, while
            T60 needs every sensor's times together, unrounded (equal to poses.npz, which is what
            scripts/09 rebuilds the split from).

            Args:
                cameras, img_filenames, lidars, pc_filenames, trajectories: As in ``ADDataParser``.

            Returns:
                The base class's filtered (cameras, img_filenames, lidars, pc_filenames, trajectories).
            """
            filtered = super()._filter_based_on_time(cameras, img_filenames, lidars, pc_filenames, trajectories)
            self._block_split_positions = None
            if self.config.test_block_count is None:
                # Cores are parts of T60 blocks; refuse c < 1 rather than ignore it silently.
                if self.config.test_block_core_fraction != WHOLE_BLOCK_CORE_FRACTION:
                    raise ValueError("test_block_core_fraction < 1 needs test_block_count")
                return filtered
            windowed_cameras, windowed_lidars = filtered[0], filtered[2]
            windowed_times = {"camera": windowed_cameras.times.reshape(-1).double().numpy(),
                              "lidar": windowed_lidars.times.reshape(-1).double().numpy()}
            block_positions, block_bounds = middle_block_train_eval_indices(
                windowed_times, self.config.train_split_fraction, self.config.test_block_count)
            # T62: eval keeps only each block's core; guard frames drop out of both splits.
            self._block_split_positions, _, _ = restrict_test_to_block_cores(
                windowed_times, block_positions, block_bounds, self.config.test_block_core_fraction)
            return filtered

        def _get_train_eval_indices(self, sensors: Union["Cameras", "Lidars"]) -> Tuple["torch.Tensor", "torch.Tensor"]:
            """neurad's LINSPACE split, or the T60 block positions when ``test_block_count`` is set.

            Args:
                sensors: The windowed cameras or lidars (same order as in ``_filter_based_on_time``).

            Returns:
                (train_indices, eval_indices) int64 tensors.

            Raises:
                ValueError: If ``max_eval_frames`` is set with blocks (it would subsample them randomly).
            """
            if self.config.test_block_count is None:
                return super()._get_train_eval_indices(sensors)
            if self.config.max_eval_frames is not None:
                raise ValueError("max_eval_frames subsamples eval frames with an RNG; unset it for the block split")
            sensor_kind = "camera" if isinstance(sensors, Cameras) else "lidar"
            train_positions, test_positions = self._block_split_positions[sensor_kind]
            return torch.from_numpy(train_positions), torch.from_numpy(test_positions)

        def _get_cameras(self) -> Tuple["Cameras", List[Path]]:
            """Build ``Cameras`` from the precomputed arrays (flip already applied).

            Returns:
                (cameras, image paths).
            """
            arrays = self._camera_arrays
            cameras = Cameras(
                fx=torch.from_numpy(arrays.focal_x),
                fy=torch.from_numpy(arrays.focal_y),
                cx=torch.from_numpy(arrays.principal_x),
                cy=torch.from_numpy(arrays.principal_y),
                width=torch.from_numpy(arrays.widths),
                height=torch.from_numpy(arrays.heights),
                distortion_params=None,  # images are pre-undistorted in the extractor
                camera_to_worlds=torch.from_numpy(arrays.camera_to_worlds),
                camera_type=CameraType.PERSPECTIVE,
                times=torch.from_numpy(arrays.times),
                metadata={"sensor_idxs": torch.from_numpy(arrays.sensor_indices)},
            )
            return cameras, list(arrays.image_paths)

        def _get_lidars(self) -> Tuple["Lidars", List[Path]]:
            """Build ``Lidars`` (type from the manifest) from the precomputed arrays.

            Returns:
                (lidars, sweep paths).
            """
            arrays = self._lidar_arrays
            lidars = Lidars(
                lidar_to_worlds=torch.from_numpy(arrays.lidar_to_worlds),
                lidar_type=LIDAR_MODEL_TO_TYPE[arrays.model.type_name],
                assume_ego_compensated=True,  # points are deskewed to the reference pose (contract)
                times=torch.from_numpy(arrays.times),
                metadata={"sensor_idxs": torch.from_numpy(arrays.sensor_indices)},
                horizontal_beam_divergence=self.config.horizontal_beam_divergence,
                vertical_beam_divergence=self.config.vertical_beam_divergence,
            )
            return lidars, list(arrays.sweep_paths)

        def _read_lidars(self, lidars: "Lidars", filepaths: List[Path]) -> List["torch.Tensor"]:
            """Load sweeps (x,y,z,intensity,t[,channel]) minus ego returns, plus no-returns if asked.

            With ``add_missing_points`` each sweep gets its recorded no-returns (Phase B sidecar,
            ``lidar_no_return/``) appended as placeholders at neurad's ``DUMMY_DISTANCE_VALUE``
            (T55), which SplatAD treats as "did not return" rays (ray-drop loss only). This
            replaces neurad's ``_get_missing_points`` guess from gaps between neighbouring returns.

            Args:
                lidars: Lidars (unused except for ordering).
                filepaths: Sweep files, aligned with ``lidars``.

            Returns:
                One float32 tensor per sweep.

            Raises:
                FileNotFoundError: With ``add_missing_points``, if a sweep has no no-return file
                    (dataset extracted before Phase B: re-extract with ``--force``).
            """
            # Ego filter from the manifest: sphere radius or box (one of them is the identity).
            model = self._lidar_arrays.model
            point_clouds = []
            for path in filepaths:
                sweep = load_lidar_sweep(path, model.min_lidar_radius, model.ego_box_half_extents)
                if self.config.add_missing_points:
                    no_return_path = no_return_path_for_sweep(path)
                    if not no_return_path.is_file():
                        raise FileNotFoundError(
                            f"{no_return_path} missing: add_missing_points needs the no-return sidecar "
                            "(re-extract the dataset with the current converter, --force)"
                        )
                    placeholders = no_return_placeholder_points(
                        np.load(no_return_path), DUMMY_DISTANCE_VALUE, sweep.shape[1])
                    sweep = np.concatenate([sweep, placeholders], axis=0)
                point_clouds.append(torch.from_numpy(sweep))
            return point_clouds

        def _add_sensor_velocities(self, cameras, lidars) -> None:
            """Base velocities, then overwrite rolling-shutter metadata from the manifest."""
            super()._add_sensor_velocities(cameras, lidars)
            frame_count = len(cameras)
            cameras.metadata["rolling_shutter_time"] = torch.full(
                (frame_count, 1), self._camera_arrays.rolling_shutter_seconds)
            cameras.metadata["time_to_center_pixel"] = torch.full(
                (frame_count, 1), self._camera_arrays.time_to_center_pixel_seconds)

        def _get_actor_trajectories(self) -> List[Dict]:
            """No actors: cuboids stay out of SplatAD inputs."""
            return []

    # Object neurad's plugin loader expects (it rejects a bare config class).
    VIKING_DATAPARSER_SPEC = DataParserSpecification(
        config=VikingHillDataParserConfig(),
        description="Viking Hill intermediate dataset (Ouster OS0-32 + one undistorted camera)",
    )
