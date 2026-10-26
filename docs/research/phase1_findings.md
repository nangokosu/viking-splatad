# Viking Hills → SplatAD: render a forest (rev 2: contract-driven)

## Context
Goal: train SplatAD (neurad-studio) on the Viking Hill dataset and render the forest.
**Primary success criterion:** SplatAD-rendered RGB (and lidar range) images of Viking Hills on held-out frames that visibly show forest structure.
Secondary: lidar/camera/radar alignment, and ground-truth cuboids extracted separately for occupancy evaluation.

Decided: SplatAD trains/renders on a **remote NVIDIA GPU box over SSH** (Mac has no CUDA: the SplatAD datamanager calls `.cuda()` unconditionally). The Mac is for authoring and CPU tests.

**Design principle (user requirement):** every transform in the Viking Hills → intermediate-dataset conversion is derived from, and checked against, SplatAD's actual input contract (table below, read from neurad-studio source). A machine-checkable `splatad_contract` validator gates all converter output before it ever reaches a GPU.

## Source facts
**Viking Hills** (research; message types/contents unverified until Phase 1): ROS2 Jazzy MCAP, Zenodo 10.5281/zenodo.20795213, ~160 GB (15 x 7z), two ~1 h sessions (short/tall grass). Sensors: 1 camera IDS uEye 2056x1542 (`/ids_camera/image_raw/compressed`, `.../camera_info`); Ouster OS0-32 10 Hz (`/ouster/points`, `/point_cloud_deskewed`); Sensrad Hugin 4D radar 16 Hz (`/hugin_raf_1/radar_data`); poses `/icp_odom` (base_link in map, SLAM-derived) + `/tf`. Calibration: `calibration/extrinsics/extrinsics.txt`, `intrinsics/camera_calibration.txt`. GT: `cuboid_labels/short_and_tall_grass_labels.json` (Segments.ai static cuboids on per-session *map*, no timestamps; classes trunk/canopy/rock/bush/car/building/lamp/ignore). Radar clock unsynced.

## SplatAD input contract → required transform (the spine of the project)
| SplatAD requirement (source) | Converter / dataparser action for Viking Hills |
|---|---|
| Dataparser = `ADDataParser` subclass overriding `_get_cameras/_get_lidars/_read_lidars/_get_actor_trajectories`; register in `configs/dataparser_configs.py`; template `py123d_dataparser.py` | `VikingHillDataParser` + config `viking-data`, installed into a neurad-studio checkout by a patch script |
| Camera: `camera_to_worlds [N,3,4]` = `T_world_base(t) @ T_base_cam` in **OpenCV** axes, then `R @= OPENCV_TO_NERFSTUDIO` (`ad_dataparser.py:45-51`) | Compose from `/icp_odom` interpolated to image time and calibrated `T_base_cam`; apply the flip exactly once, in the dataparser only. Unit test with a known pose. |
| Camera `fx,fy,cx,cy` float px; `width,height` **int** tensors; `distortion_params` `[N,6]` (k1,k2,k3,k4,p1,p2) with k4==0, or None | Pre-undistort offline with `cv2.initUndistortRectifyMap` (alpha=0 crop from `camera_info`), write new intrinsics, pass `distortion_params=None`. Validator checks the file size equals declared w/h. |
| Images uint8 RGB(A), 3 or 4 ch; model downscales itself (`num_downscales=2`) | Write undistorted JPEG/PNG; optional pre-resize factor from config (derived from GPU memory budget, not hardcoded) |
| Times: float64 **seconds**, one common clock for all sensors, strictly increasing per sensor (velocities divide by Δt → NaN on duplicates) | Single reference clock (ROS header stamps, PTP-synced lidar); drop/dedupe duplicate stamps; validator asserts monotone and finite |
| Camera metadata `sensor_idxs [N,1]`, contiguous 0..n-1; `rolling_shutter_time` (readout seconds), `time_to_center_pixel` (centre-row offset from stamp) are per-sensor config floats | `sensor_idx=0` for the one camera. Set the two floats from camera readout spec / exposure-stamp convention found in Phase 1; if unknown start at 0 (disables RS), then ablate. Constants justified in config comments. |
| Lidar `lidar_to_worlds [N,3,4]` = pose of lidar **at `lidars.times[i]`**; points in **sensor frame**, z-up, azimuth=`atan2(y,x)`, elevation=`asin(z/d)` | Ouster sensor frame is x-fwd,y-left,z-up → compatible; verify `frame_id` of the chosen topic and convert if it is in base_link |
| Lidar points `[n,5]` float32 = `x,y,z,intensity,t[,channel]`; intensity ∈ [0,1] (sigmoid/MSE) | Normalise Ouster intensity/reflectivity by a data-derived scale stored in manifest (dtype max or global percentile); validator checks range |
| Per-point `t`: signed seconds relative to that scan's `lidars.times`; datamanager does `xyz_rs = xyz − v·t`, i.e. **points must already be ego-motion-compensated to the scan reference pose** | Define reference time = scan end (nuScenes-style) or mid (config). If `/ouster/points` is raw with per-point `t` field: **own deskew module** compensates with interpolated poses and emits `t − t_ref`. If only the deskewed topic is usable, synthesise linear `t` across the sweep. Decision made in Phase 1 from real fields. Test: synthetic moving-lidar scene deskews to a static wall. |
| `LidarType` has no Ouster member; datamanager needs `get_lidar_elevation_mapping` (dict channel→elev°; length multiple of 8 for 8-ch tiles) and `get_lidar_azimuth_resolution`, else `ValueError` | Add `OUSTER_OS0_32` to `LidarType`, `LIDAR_MODEL_TO_TYPE`, `data/utils/lidar_elevation_mappings.py` and the azimuth branch (3 small edits, shipped as a patch). Elevations **derived from data** (per-ring elevation medians from accumulated points or Ouster beam metadata if in bag), azimuth resolution = 360/(columns per ring measured). 32 channels satisfies the multiple-of-8 rule. Tile math: `tile_width=ceil(360/(res·32))`. |
| `add_missing_points` needs mapping/resolution/skip dicts + channel column; creates 2000 m dummy points → ray-drop labels | Phase A: `add_missing_points=False` (py123d default). Phase B (quality): enable after channel/ring column + mapping validated |
| `min_lidar_dist` box removes ego points in sensor frame | Derive from points clustering near origin / Husky extents in calibration; store in manifest |
| Points beyond `valid_lidar_distance_threshold` (1000 m) are "no return" | Clip nothing; Ouster no-return points (range 0) are dropped |
| Poses: z-up metric world; parser re-centres on first lidar's mean trajectory (needs ≥2 poses; std<0.1 m falls back to first pose); scene box needs lidars | Use `/icp_odom` map frame (check z-up). Choose window with enough motion. |
| Radars unsupported (assert) | Radar goes only to a sidecar dir for alignment diagnostics, never to the dataparser |
| Actors: `load_cuboids=False` → `trajectories=[]`; traced as supported, **runtime unverified** (torch_compile on empty tensors) | `load_cuboids=False`; **Phase 0 synthetic zero-actor GPU smoke test** before real data |
| Static Gaussians seed from lidar train scans (`norm<1000`) + painted colours from nearest-in-time cameras (`paint_points`, topk=2) | Camera and lidar need not be time-paired; keep camera rate/lidar overlap sufficient so points get coloured |
| Train/eval split LINSPACE per sensor, `train_split_fraction` default 0.5, ≥2 frames/sensor, non-empty train sets | Config: higher train fraction for a single-camera trajectory; validator checks non-empty splits |
| `sensor_idx_to_name = cameras + lidars`; lidar idx offset by n_cam if overlapping | Name them `ids_camera`, `ouster`; validator asserts contiguous |
| Trajectory-asserts, masks (`mask_filenames=None`), `dataparser_scale=1` | No masks in v1 |

## Approach
Two decoupled stages: (1) standalone MCAP → intermediate dataset (pure-python `mcap`/`mcap-ros2-support`, no ROS install); (2) thin `VikingHillDataParser` that reads the intermediate dataset and is the only code touching neurad-studio internals. Rejected: py123d export (drops per-point lidar time, extra dependency). One time window of one session (default short-grass), selected programmatically (pose quality + motion + camera/lidar overlap); only the needed bag chunks are extracted.

## Project layout (repository root)
```
CLAUDE.md  README.md  pyproject.toml
configs/default.yaml  configs/remote.yaml
src/viking_splatad/
  io/mcap_reader.py             # topic iteration + decode, chunk selection
  extract/pose.py camera.py lidar.py deskew.py radar.py calibration.py
  contract/splatad_contract.py  # machine-checkable table above (validator + report)
  contract/lidar_model.py       # derive elevation mapping / azimuth res / min_lidar_dist from data
  align/time_sync.py spatial_check.py
  labels/cuboids.py occupancy.py eval_occupancy.py   # GT, separate from SplatAD data
  dataset/schema.py writer.py   # intermediate dataset + manifest
  splatad/viking_dataparser.py patches/ (LidarType OS0-32, dataparser registration) install_into_neurad.sh
scripts/00_download_window.sh 01_inspect_bags.py 02_select_window.py 03_extract.py
        04_align_report.py 05_extract_cuboids.py 06_validate_contract.py
        07_sync_to_gpu.sh 08_train_render.sh 09_eval.py
docker/Dockerfile.splatad       # CUDA 11.8 per neurad-studio README
tests/unit integration contract gpu fixtures/   ;  reports/  data/(gitignored)
```
Intermediate dataset (`data/intermediate/<session>/`): `manifest.json` (sensor list, intrinsics, extrinsics, reference-clock info, RS params, lidar type + mapping, intensity scale, min_lidar_dist), `poses.npz`, `camera/<ts>.jpg` (undistorted), `lidar/<ts>.npy` ([n,5]/[n,6], contract-conformant), `sidecar/radar/`. GT is separate: `data/gt/cuboids.json`, `occupancy.npz`.

## Execution plan (each phase gated by its tests)
0. **Scaffold + toolchain proof (GPU box):** project, git, pytest; build Docker env; `ns-train splatad --help`. Generate a **synthetic scene** (analytic walls/boxes, moving rig, known distortion) through our writer + dataparser and run a 200-step SplatAD smoke → proves (a) our contract output trains, (b) the zero-actor path runs, (c) the `OUSTER_OS0_32` patch works.
1. **Inspect real bags** (download chunk 0 + calibration + labels JSON): record true message types, camera fps, `camera_info` distortion model, lidar fields (`t`, ring, reflectivity, frame_id), pose rate/frame, extrinsics frames, MCAP encoding. Resolve every "unverified" item in the table; update config. *Checkpoint with user if: camera fps very low, no per-point lidar time, fisheye, or selective 7z extraction impossible.*
2. **Extract** pose / camera (undistort) / lidar (+deskew, ring, intensity scale) / radar / calibration.
3. **Align** (time offsets incl. radar drift; lidar→image reprojection score; radar vs lidar overlay; file extrinsics vs `/tf`). Alignment must pass before SplatAD training.
4. **Contract validation + dataparser:** `06_validate_contract.py` runs the full table as asserts and emits a report; then `VikingHillDataParser` loads the dataset through the real `ADDataParser` pipeline.
5. **Train + render on GPU:** short run, then full 30001 steps; `ns-render` held-out RGB + lidar range images. If bad, debug in order: contract validator → pose convention → undistortion → time offsets/RS params → deskew/t sign → intensity scale → scene box. Phase B: enable `add_missing_points`.
6. **GT cuboids + occupancy** (parallel, independent of 4–5): cuboids→voxel occupancy in map frame (per class), per-frame crop via pose chain; occupancy-eval harness (IoU/precision/recall) vs lidar-accumulated and SplatAD-derived occupancy.

## Test suite

### Rule: every transformation and calibration has its own unit test (enforced)
- Every conversion/calibration function is registered with `@transform("T<n>", ...)` in `src/viking_splatad/registry.py`.
- Every unit test is tagged `@pytest.mark.covers("T<n>")`.
- `tests/test_traceability.py` fails the suite if any registered transform has no covering test (and if a test names an unknown id). Adding a transform without a test cannot go green.
- Each test uses synthetic known-answer input (analytic geometry, injected offset, or round-trip with the inverse), never "runs without error". Expected values are computed in the test from first principles, not copied from the implementation.

| ID | Transformation / calibration | Unit test (known answer) |
|---|---|---|
| T1 | Time-window + bag-chunk selection | Synthetic chunk metadata; selected chunks cover exactly the window, none extra |
| T2 | Stamp → common-clock float64 seconds, dedupe, monotone | Duplicate/unsorted stamps in → strictly increasing out; ns→s exact |
| T3 | `/icp_odom` → SE3 trajectory + interpolation | Constant-velocity + constant-yaw-rate path: interpolated pose equals analytic pose; slerp unit-norm |
| T4 | Frame chain via `/tf` (map→odom→base_link), z-up check | Hand-built TF tree composes to expected matrix; tilted world flagged |
| T5 | `extrinsics.txt` parse | Fixture text → expected 4x4 per sensor pair |
| T6 | Calibration file vs `/tf` cross-check | Matching → pass; perturbed by known δ → reports δ |
| T7a | Camera intrinsics parse (`camera_info`/checkerboard) | Fixture → expected K and distortion vector, model name |
| T7b | Distortion conversion to nerfstudio order (k1,k2,k3,k4=0,p1,p2) | OpenCV (k1,k2,p1,p2,k3) permuted correctly; k4==0 |
| T7c | Image undistortion + new intrinsics | Synthetic distorted grid → straight lines; new K reprojects a known 3D point to the expected pixel |
| T8 | Image decode / optional resize / size consistency | Resize factor f → K scales by f; declared w/h == array shape |
| T9a | Camera c2w composition `T_world_base @ T_base_cam` | Known base pose + extrinsic → analytic c2w |
| T9b | OPENCV→nerfstudio axis flip | `diag(1,-1,-1)` applied once; applying twice returns original; a point 1 m ahead lands on −z |
| T10 | Rolling-shutter params (`rolling_shutter_time`, `time_to_center_pixel`) | Start-of-exposure stamp → +readout/2; centre stamp → 0; end stamp → −readout/2 |
| T11 | PointCloud2 decode | Encode synthetic structured cloud → decode equals input (xyz, intensity, t, ring) |
| T12 | Lidar frame conversion (`frame_id` → sensor frame) | Cloud in base_link + known extrinsic → sensor-frame points equal analytic |
| T13 | Deskew + per-point `t` | Rig moving at v past a static wall: deskewed points collinear (residual < tolerance derived from noise-free case); `t` signed, bounded by sweep period; sign flip detected |
| T14 | Intensity normalisation to [0,1] | Values map to [0,1], monotone, scale stored in manifest; out-of-range clipped and counted |
| T15 | `min_lidar_dist` derivation | Synthetic cloud with a box-shaped cluster at origin → derived thresholds enclose it, exclude far points |
| T16 | Elevation-mapping derivation (32 channels) | Synthetic OS0-32 scan with known beam angles → recovered mapping within angular tolerance; len % 8 == 0 |
| T17 | Azimuth resolution + tile math | 1024 columns → 0.3516°, `tile_width=32`; 512 columns → 16; non-integer tiles rejected |
| T18 | `lidar_to_worlds` composition at scan reference time | Known pose + lidar extrinsic → analytic matrix; reference-time choice (start/mid/end) changes pose as expected |
| T19 | Channel/ring column assignment | Points on known rings → channel indices match mapping keys |
| T20 | `sensor_idxs` assignment | Contiguous from 0; lidar offset by n_cam rule reproduced |
| T21 | Train/eval LINSPACE split sanity | Both splits non-empty, disjoint, interleaved for N frames |
| T22 | Camera/lidar/pose time-offset estimation | Inject known offset δ into one stream → estimator recovers δ |
| T23 | Radar decode | Synthetic radar message → expected xyz, power, Doppler |
| T24 | Radar clock offset + radar→lidar extrinsic check | Injected drift/offset recovered; identity extrinsic yields zero residual |
| T25 | Lidar→camera reprojection score | Synthetic point lands at analytic pixel; perturbed extrinsic lowers score monotonically |
| T26 | Dataset writer ↔ reader (manifest/schema) | Write then read round-trip equals input; schema version mismatch raises |
| T27 | `VikingHillDataParser` → `DataparserOutputs` | Tiny written dataset → counts, dtypes, monotone times, finite poses, `trajectories==[]` |
| T28 | `OUSTER_OS0_32` LidarType patch | `get_lidar_elevation_mapping` / `get_lidar_azimuth_resolution` return derived values, no `ValueError` |
| T29 | Cuboid JSON parse | Fixture → typed boxes, class ids, full-size dims |
| T30 | Cuboid pose conversion (quaternion/yaw → R, full-size → half-extent) | Axis-aligned and 90°-rotated boxes give analytic corners |
| T31 | Cuboid voxelisation → occupancy | Analytic box volume ≈ occupied voxel count × voxel volume (within discretisation bound); class priority respected |
| T32 | Per-frame GT crop via pose chain | Known pose → voxels inside sensor FOV/range equal analytic set |
| T33 | Occupancy eval metrics (IoU/precision/recall) | Hand-built grids with known overlap → exact values |

### Additional suites
- **Contract (CPU, run on every extract):** each row of the table is a test/assert: dtypes/shapes (`int` w/h, `float32 [N,3,4]`, `[n,5]`), intensity∈[0,1], strictly increasing finite float64 times, per-point `t` range within sweep period, contiguous `sensor_idxs`, `distortion_params[3]==0` or None, image size == declared, non-empty train/eval, mapping length multiple of 8, azimuth-res tile math, z-up, finite poses ≥2.
- **Unit:** SE3 interpolation; PointCloud2 decode round-trip; undistortion on synthetic grid; calibration parser; quaternion↔matrix; deskew recovers a static wall from a moving rig; elevation-mapping/azimuth derivation on synthetic Ouster-like scan; time-offset estimator recovers injected offset; cuboid containment/voxelisation vs analytic boxes; occupancy IoU on hand-built grids.
- **Convention:** OPENCV→nerfstudio flip on known pose; lidar→world→camera reprojection of a synthetic point lands at expected pixel; box dims/axis order.
- **Integration (small real fixture, ~2 s of bag, generated by script):** extract→manifest→contract report all pass; `VikingHillDataParser` through real `ADDataParser` returns consistent counts, monotone times, scene box finite. (Needs neurad-studio importable; runs on GPU box if the Mac install fails.)
- **Alignment regression:** reprojection score threshold derived from first baseline run (`reports/baseline.json`), not hardcoded.
- **GPU (`-m gpu`, over SSH):** synthetic-scene 200-step smoke without NaN (zero actors); real-window 200-step smoke; `ns-render` image non-constant; held-out PSNR > mean-colour baseline.

## Conventions
Follows `~/CLAUDE.md`: docstring per function, step comments, math explained, descriptive names (no `i`/`x`), no hardcoded numbers (derive from bag/manifest/config; justify unavoidable ones). Anonymous identifier on any network fetch. Work parallelised via subagents (extractors / aligners / labels / dataparser+patches are independent); `comment-reviewer` pass after each file.

## Risks
- Zero-actor runtime path and gsplat internals (rows within tile, out-of-range elevations) unverified → Phase 0 smoke.
- Ouster not a built-in type; wrong elevation mapping gives a distorted range image → derive from data, validate with synthetic + real range-image reprojection.
- Raw vs deskewed lidar topic may force own deskew (correctness-critical; tested).
- Single forward camera + SLAM (not RTK) poses → possible blur; pick clean window, enable pose optimisation if needed.
- Foliage is high-frequency/view-sparse → judge qualitatively + vs baseline.
- 160 GB data; CTU mirror was down; 7z multi-part may prevent selective extraction → budget full download on the GPU box.
- Cuboids are map-frame, two maps differ ~±30 cm.

## Verification (end to end)
1. `pytest -m "not gpu"` green on Mac, including the contract suite.
2. `scripts/01..06` on the GPU box produce the intermediate dataset, alignment report and a passing contract report.
3. `pytest -m gpu` smoke passes (synthetic zero-actor + real window).
4. Full `ns-train splatad viking-data` completes; `ns-render` produces eval-frame images; user visually confirms forest; `09_eval.py` writes image/lidar/occupancy metrics to `reports/`.

## Phase 1 findings (May chunk 0, `reports/may_chunk0_inspection.md`) — resolves the "unverified" items
- **Bag**: MCAP/CDR; chunk 0 = 212k msgs, 99.8 s. The bag's CameraInfo schema uses uppercase fields (D,K,R,P) → `io/mcap_reader.make_decoded_reader` patches the decoder (tested). Use ROS *header stamps*; bag log_time lags them by 0.09–0.13 s.
- **Lidar** `/ouster/points` and `/point_cloud_deskewed` (frame `os_sensor`, 1024x32, point_step 48): fields x,y,z f32; intensity f32; **t u32 ns offset within the sweep (0..~1e8)**; reflectivity u16; **ring u16 (0 = top +44.9°, 31 = −42.2°, non-uniform ~2.7–3.0° spacing)**; ambient; range u32 mm. 21–58% of points valid (range>0); invalid raw points are zero; invalid *deskewed* points hold tiny non-zero junk → always mask by range>0. Deskewed has the same valid mask; reference time ≈ sweep start (inferred: |desk−raw| smallest at t≈0; still to confirm by running T13 on raw+icp_odom and comparing). 1024 columns → 0.3516°, 32 tiles (T17).
- **Camera**: 2056x1542, plumb_bob D=[k1,k2,p1,p2,k3]=[-0.0366,0.0639,0.0006,-0.0009,0], K fx≈1461.6, ~10 Hz. JPEG decodes with `cv2.imdecode` to correct-colour BGR (verified visually). `base_link→camera` quaternion (-.5,.5,-.5,.5) is already an OpenCV optical frame.
- **Extrinsics verified visually**: lidar→camera overlay with `extrinsics.txt` (base_link→os_sensor t=(0.02,0,0.225), identity rot; base_link→camera t=(0.18,-0.01,0.14)) puts ground rings on the road, car/building/trunk returns on the objects (`reports/may_chunk0_lidar_overlay.jpg`).
- **Poses**: `/icp_odom` (map→base_link, z-up, ~10 Hz); 73 m path in chunk 0 (slow first 30 s). Camera and lidar stamps are each 10 Hz, ≤50 ms apart.
- **Radar** `/hugin_raf_1/radar_data` PointCloud2 (x,y,z,range,elevation,azimuth,power,doppler; frame `hugin_radar`, ~3.3k pts, ~18 Hz). Radar extrinsic differs between May and June → per-session calibration.
- **Labels**: 2 samples (May map 467 cuboids, June map 549), annotations at `dataset.samples[i].labels["ground-truth"].attributes.annotations`; ids 1–8 as planned, file names capitalised; yaw-only; **no frame info in the file** → whether the label map frame equals the bag's `/icp_odom` map frame is UNVERIFIED (needed for occupancy eval). Existing loader works with `sample_key="2024_05"`.
- **Scene content**: chunk 0 is a campus road (buildings, parked cars, pines on one side), NOT dense forest. Forest chunks must be located before choosing the render window.

## Status (after unpack-pipeline milestone)
- Built + committed: config-driven unpack (`scripts/03_extract.py`), one-command chunk fetch (`00_fetch_chunk.py`, each chunk = independent ~1.6 GB block), contract validator (06), alignment report (04), GT cuboids/occupancy (05), SplatAD dataparser + `OUSTER_OS0_32` patch (T27/T28, CPU-tested in a separate neurad venv; NOT yet run under CUDA/`ns-train`). Runbook in `README.md`.
- Verified on real May data: chunk 0 full (995 cam / 997 lidar / 1766 radar, contract PASS, 36 s); chunk 1 added incrementally (chunk 0 skipped, `derived.json` byte-identical, stamps continuous at the join, contract PASS). Own deskew (T13, raw + /icp_odom) matches the provided deskewed cloud to 1.7 mm median. Dataset-only lidar→camera overlays align at frames 150/600/850.
- **Forest is in May chunk 1** (≈ +45 s onward, camera frames ~1445–1821): birch/pine woodland over grass. Chunk 0 = campus road.
- Open items: (1) camera↔lidar time offset not measurable by the edge metric (alignment report INCONCLUSIVE on this scene; rely on overlays; SplatAD has per-sensor time-offset optimisation); (2) labels' map frame vs bag `/icp_odom` map frame UNVERIFIED (blocks occupancy eval); (3) `add_missing_points` (Phase B) not implemented; (4) rolling-shutter params unknown (set to 0); (5) manifest stores `min_lidar_dist` as a list while schema annotates float; (6) 9/199 frames differed in point count between own and provided deskew (unexplained); (7) GPU box not yet specified.
