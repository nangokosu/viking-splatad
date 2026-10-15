# Bag inspection: `short_grass__ros2_0.mcap`

## Topics

### /ouster/points
- frame_id: `os_sensor`, width x height: 1024 x 32, point_step: 48, is_dense: True, big_endian: False
- fields: x(off=0, dtype=7, n=1), y(off=4, dtype=7, n=1), z(off=8, dtype=7, n=1), intensity(off=16, dtype=7, n=1), t(off=20, dtype=6, n=1), reflectivity(off=24, dtype=4, n=1), ring(off=26, dtype=4, n=1), ambient(off=28, dtype=4, n=1), range(off=32, dtype=6, n=1)
- header stamp[0]: 1714571311.141675s; median spacing: 0.1000s; median (log_time - header_stamp): 0.1246s
- points in first cloud: 32768
  - x: min -28.58, max 1.961, nonzero 6995/32768
  - y: min -22.13, max 0.01044, nonzero 6995/32768
  - z: min -1.034, max 12.21, nonzero 6995/32768
  - intensity: min 0, max 2595, nonzero 11024/32768
  - t: min 0, max 9.992e+07, nonzero 11264/32768
  - reflectivity: min 0, max 6305, nonzero 11264/32768
  - ring: min 0, max 31, nonzero 31744/32768
  - ambient: min 0, max 144, nonzero 6963/32768
  - range: min 0, max 3.232e+04, nonzero 6995/32768

### /point_cloud_deskewed
- frame_id: `os_sensor`, width x height: 1024 x 32, point_step: 48, is_dense: True, big_endian: False
- fields: x(off=0, dtype=7, n=1), y(off=4, dtype=7, n=1), z(off=8, dtype=7, n=1), intensity(off=16, dtype=7, n=1), t(off=20, dtype=6, n=1), reflectivity(off=24, dtype=4, n=1), ring(off=26, dtype=4, n=1), ambient(off=28, dtype=4, n=1), range(off=32, dtype=6, n=1)
- header stamp[0]: 1714571311.141694s; median spacing: 0.1000s; median (log_time - header_stamp): 0.1277s
- points in first cloud: 32768
  - x: min -28.58, max 61.51, nonzero 32768/32768
  - y: min -61.23, max 49.57, nonzero 32768/32768
  - z: min -1.58, max 16.78, nonzero 32768/32768
  - intensity: min 0, max 6116, nonzero 32114/32768
  - t: min 0, max 9.991e+07, nonzero 32736/32768
  - reflectivity: min 168, max 6.554e+04, nonzero 32768/32768
  - ring: min 0, max 31, nonzero 31744/32768
  - ambient: min 0, max 255, nonzero 19600/32768
  - range: min 0, max 8.461e+04, nonzero 19645/32768

### /hugin_raf_1/radar_data
- frame_id: `hugin_radar`, width x height: 3314 x 1, point_step: 32, is_dense: False, big_endian: False
- fields: x(off=0, dtype=7, n=1), y(off=4, dtype=7, n=1), z(off=8, dtype=7, n=1), range(off=12, dtype=7, n=1), elevation(off=16, dtype=7, n=1), azimuth(off=20, dtype=7, n=1), power(off=24, dtype=7, n=1), doppler(off=28, dtype=7, n=1)
- header stamp[0]: 1714571311.133000s; median spacing: 0.0560s; median (log_time - header_stamp): 0.0938s
- points in first cloud: 3314
  - x: min -20.83, max 34.77, nonzero 3294/3314
  - y: min 1.012, max 50.32, nonzero 3314/3314
  - z: min -8.767, max 8.812, nonzero 3274/3314
  - range: min 1.013, max 50.47, nonzero 3314/3314
  - elevation: min -0.3001, max 0.3001, nonzero 3274/3314
  - azimuth: min -0.7936, max 1.059, nonzero 3294/3314
  - power: min 64.88, max 106.9, nonzero 3314/3314
  - doppler: min -0.6411, max 0.3561, nonzero 830/3314

### /ids_camera/image_raw/camera_info
- frame_id `camera`, size 2056x1542, distortion_model `plumb_bob`
- D = [-0.036581, 0.063865, 0.000594, -0.000878, 0.0]
- K = [1461.57941, 0.0, 1028.16795, 0.0, 1461.83192, 820.47617, 0.0, 0.0, 1.0]
- P = [1476.23096, 0.0, 1025.56845, 0.0, 0.0, 1475.85815, 819.54816, 0.0, 0.0, 0.0, 1.0, 0.0]

### /ids_camera/image_raw/compressed
- format `rgb8; jpeg compressed bgr8`, bytes 759224, frame_id `camera`, magic ffd8ffe0
- median spacing 0.1000s

### /icp_odom
- header.frame_id `map`, child_frame_id `base_link`
- first position (1.194, -0.148, -0.006), quat xyzw (-0.0001, -0.0002, -0.9999, 0.0107)
- path length over first 50 msgs: 0.22 m

### /husky_udp_bridge/odom
- header.frame_id `odom`, child_frame_id `base_link`
- first position (118.925, -8.081, 0.000), quat xyzw (0.0000, 0.0000, 1.0000, 0.0003)
- path length over first 50 msgs: 0.00 m

### /imu_odom
- header.frame_id `odom`, child_frame_id `base_link`
- first position (-0.000, -0.000, -0.000), quat xyzw (0.0044, -0.0045, 0.6978, 0.7163)
- path length over first 50 msgs: 0.00 m

### /tf_static
- `os_sensor` -> `os_lidar`: t=(0.0000, 0.0000, 0.0362) q_xyzw=(0.0000, 0.0000, 1.0000, 0.0000)
- `os_sensor` -> `os_imu`: t=(0.0063, -0.0118, 0.0076) q_xyzw=(0.0000, 0.0000, 0.0000, 1.0000)
- `os_sensor` -> `os_lidar`: t=(0.0000, 0.0000, 0.0362) q_xyzw=(0.0000, 0.0000, 1.0000, 0.0000)
- `os_sensor` -> `os_imu`: t=(0.0063, -0.0118, 0.0076) q_xyzw=(0.0000, 0.0000, 0.0000, 1.0000)
- `os_sensor` -> `os_lidar`: t=(0.0000, 0.0000, 0.0362) q_xyzw=(0.0000, 0.0000, 1.0000, 0.0000)
- `os_sensor` -> `os_imu`: t=(0.0063, -0.0118, 0.0076) q_xyzw=(0.0000, 0.0000, 0.0000, 1.0000)
- `os_sensor` -> `os_lidar`: t=(0.0000, 0.0000, 0.0362) q_xyzw=(0.0000, 0.0000, 1.0000, 0.0000)
- `os_sensor` -> `os_imu`: t=(0.0063, -0.0118, 0.0076) q_xyzw=(0.0000, 0.0000, 0.0000, 1.0000)
- `os_sensor` -> `os_lidar`: t=(0.0000, 0.0000, 0.0362) q_xyzw=(0.0000, 0.0000, 1.0000, 0.0000)
- `os_sensor` -> `os_imu`: t=(0.0063, -0.0118, 0.0076) q_xyzw=(0.0000, 0.0000, 0.0000, 1.0000)
- `os_sensor` -> `os_lidar`: t=(0.0000, 0.0000, 0.0362) q_xyzw=(0.0000, 0.0000, 1.0000, 0.0000)
- `os_sensor` -> `os_imu`: t=(0.0063, -0.0118, 0.0076) q_xyzw=(0.0000, 0.0000, 0.0000, 1.0000)
- `os_sensor` -> `os_lidar`: t=(0.0000, 0.0000, 0.0362) q_xyzw=(0.0000, 0.0000, 1.0000, 0.0000)
- `os_sensor` -> `os_imu`: t=(0.0063, -0.0118, 0.0076) q_xyzw=(0.0000, 0.0000, 0.0000, 1.0000)

### /tf (dynamic) frame pairs seen
- `base_link` -> `camera`
- `base_link` -> `flir_camera`
- `base_link` -> `hugin_radar`
- `base_link` -> `navtech`
- `base_link` -> `os_sensor`
- `base_link` -> `radar_pose_can0`
- `base_link` -> `radar_pose_can1`
- `base_link` -> `sick`
- `base_link` -> `ti_mmwave_0`
- `base_link` -> `vertical_os_sensor`
- `map` -> `odom`
- `odom` -> `base_link`

## Calibration files

### extrinsics.txt
```
#------------------------------------------------------------------
# Transformations
#------------------------------------------------------------------

All of the following transforms are also available in the bagfiles.
The bagfiles contain additional static TFs related to other sensors not available in this
dataset, please ignore those.
#------------------------------------------------------------------

base_link -> camera

- Translation: [0.180, -0.010, 0.140]
- Rotation: in Quaternion [-0.500, 0.500, -0.500, 0.500]
            in RPY (radian) [-1.572, -0.000, -1.572]
            in RPY (degree) [-90.046, -0.000, -90.046]

#------------------------------------------------------------------

base_link -> emlid_gps

- Translation: [-0.280, 0.000, 0.805]
- Rotation: in Quaternion [0.000, 0.000, 0.000, 1.000]
            in RPY (radian) [0.000, -0.000, 0.000]
            in RPY (degree) [0.000, -0.000, 0.000]

#------------------------------------------------------------------

base_link -> hugin_radar

Attention, hugin radar TF differs between the "short_grass__ros1__" and "high_grass__ros1__" runs!

"short_grass__ros1__" bags (2024-05):

- Translation: [0.200, 0.170, 0.080]
- Rotation: in Quaternion [0.701, 0.712, 0.028, 0.027]
            in RPY (radian) [3.064, 0.000, 1.586]
            in RPY (degree) [175.554, 0.000, 90.871]

"high_grass__ros1__" bags (2024-06):

- Translation: [0.200, 0.170, 0.080]
- Rotation: in Quaternion [0.703, 0.711, 0.004, 0.004]
            in RPY (radian) [3.129, -0.000, 1.581]
            in RPY (degree) [179.278, -0.000, 90.585]

#------------------------------------------------------------------

base_link -> imu

- Translation: [0.040, 0.120, 0.010]
- Rotation: in Quaternion [0.000, 0.000, -0.707, 0.707]
            in RPY (radian) [0.000, 0.000, -1.571]
            in RPY (degree) [0.000, 0.000, -90.000]
            
#------------------------------------------------------------------

base_link -> os_sensor

- Translation: [0.020, 0.000, 0.225]
- Rotation: in Quaternion [0.000, 0.000, 0.000, 1.000]
            in RPY (radian) [0.000, -0.000, 0.000]
            in RPY (degree) [0.000, -0.000, 0.000]

os_sensor -> os_lidar

- Translation: [0.000, 0.000, 0.036]
- Rotation: in Quaternion [0.000, 0.000, 1.000, 0.000]
            in RPY (radian) [0.000, -0.000, 3.142]
            in RPY (degree) [0.000, -0.000, 180.000]

os_sensor -> os_imu

- Translation: [0.006, -0.012, 0.008]
- Rotation: in Quaternion [0.000, 0.000, 0.000, 1.000]
            in RPY (radian) [0.000, -0.000, 0.000]
            in RPY (degree) [0.000, -0.000, 0.000]

#------------------------------------------------------------------
```

### camera_calibration.txt
```
height: 1542
width: 2056
distortion_model: "plumb_bob"
D: [-0.036581, 0.063865, 0.000594, -0.000878, 0.0]
K: [1461.57941, 0.0, 1028.16795, 0.0, 1461.83192, 820.47617, 0.0, 0.0, 1.0]
R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
P: [1476.23096, 0.0, 1025.56845, 0.0, 0.0, 1475.85815, 819.54816, 0.0, 0.0, 0.0, 1.0, 0.0]
binning_x: 0
binning_y: 0
roi: 
  x_offset: 0
  y_offset: 0
  height: 0
  width: 0
  do_rectify: False
```

### hugin_radar_startup_params.txt
```
Sequence type: 
   FineShortSeq = 4
   
Used thresholds:
   {static_threshold: 5.0, dynamic_azimuth: 14.0, dynamic_elevation: 6.0}
```

## Cuboid labels `short_and_tall_grass_labels.json`
- `$` dict keys: ['name', 'description', 'created_at', 'dataset']
- `$.name` = 'v0.3'
- `$.description` = 'Added labels for radar data outside the reach of the lidar, "ignore" class introduced to cover data we are not able to label.'
- `$.created_at` = '2025-09-04 09:54:58.505292+00:00'
- `$.dataset` dict keys: ['name', 'description', 'category', 'created_at', 'owner', 'task_type', 'task_attributes', 'tasks', 'labelsets', 'samples']
- `$.dataset.name` = 'Big_map_high_grass'
- `$.dataset.description` = 'Pointcloud uploaded as one complete map. Contains the forested campus area.'
- `$.dataset.category` = 'Garden & vegetation'
- `$.dataset.created_at` = '2025-07-21 08:45:58.511956+00:00'
- `$.dataset.owner` = 'robot'
- `$.dataset.task_type` = 'pointcloud-cuboid'
