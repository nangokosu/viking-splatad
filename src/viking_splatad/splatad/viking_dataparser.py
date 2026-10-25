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
from typing import Dict, List, Optional, Tuple, Type

import numpy as np

from viking_splatad.splatad.viking_conversions import (
    OUSTER_LIDAR_TYPE_NAME,
    DatasetView,
    build_sensor_arrays,
    load_lidar_sweep,
    open_dataset_view,
    resolve_train_fraction,
)

try:
    import torch

    from nerfstudio.cameras import lidars as lidars_module
    from nerfstudio.cameras.cameras import Cameras, CameraType
    from nerfstudio.cameras.lidars import LIDAR_MODEL_TO_TYPE, Lidars
    from nerfstudio.data.dataparsers.ad_dataparser import ADDataParser, ADDataParserConfig
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
            """Load sweeps (x,y,z,intensity,t[,channel]) minus ego returns.

            Args:
                lidars: Lidars (unused except for ordering).
                filepaths: Sweep files, aligned with ``lidars``.

            Returns:
                One float32 tensor per sweep.
            """
            # Ego filter from the manifest: sphere radius or box (one of them is the identity).
            model = self._lidar_arrays.model
            point_clouds = [
                torch.from_numpy(load_lidar_sweep(path, model.min_lidar_radius, model.ego_box_half_extents))
                for path in filepaths
            ]
            if self.config.add_missing_points:
                raise NotImplementedError(
                    "add_missing_points (Phase B) is not implemented for the Viking parser yet"
                )
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
