# P1/P2 over repeats: may_short_grass

Descriptive only: a variant gap counts only when it exceeds the spread between repeats without losing precision_H. Repeats share one seed, so they are not paired. Cells are mean ± sample std (n = repeats).

- **may_scene_a** (`splatad`): may_short_grass-may_scene_a-20261116-133928-17bb7a1 [17bb7a1; split identical to the reference record], may_short_grass-may_scene_a-20261116-135425-17bb7a1 [17bb7a1; split identical to the reference record], may_short_grass-may_scene_a-20261116-135428-17bb7a1 [17bb7a1; split identical to the reference record], may_short_grass-may_scene_a-20261116-135432-17bb7a1 [17bb7a1; split identical to the reference record], may_short_grass-may_scene_a-20261116-135435-17bb7a1 [17bb7a1; split identical to the reference record]
- **may_scene_a_radar_init** (`splatad-radar-init`): may_short_grass-may_scene_a_radar_init-20261116-134527-17bb7a1 [17bb7a1; split identical to the reference record], may_short_grass-may_scene_a_radar_init-20261116-135426-17bb7a1 [17bb7a1; split identical to the reference record], may_short_grass-may_scene_a_radar_init-20261116-135429-17bb7a1 [17bb7a1; split identical to the reference record], may_short_grass-may_scene_a_radar_init-20261116-135433-17bb7a1 [17bb7a1; split identical to the reference record], may_short_grass-may_scene_a_radar_init-20261116-135436-17bb7a1 [17bb7a1; split identical to the reference record]
- **may_scene_a_radar_init_loss** (`splatad-radar-init-lidar`): may_short_grass-may_scene_a_radar_init_loss-20261116-135125-17bb7a1 [17bb7a1; split identical to the reference record], may_short_grass-may_scene_a_radar_init_loss-20261116-135427-17bb7a1 [17bb7a1; split identical to the reference record], may_short_grass-may_scene_a_radar_init_loss-20261116-135431-17bb7a1 [17bb7a1; split identical to the reference record], may_short_grass-may_scene_a_radar_init_loss-20261116-135434-17bb7a1 [17bb7a1; split identical to the reference record], may_short_grass-may_scene_a_radar_init_loss-20261116-135437-17bb7a1 [17bb7a1; split identical to the reference record]

## obstacles, P2 occupied at >= 0.5

| metric | may_scene_a | may_scene_a_radar_init | may_scene_a_radar_init_loss |
|---|---|---|---|
| P2@0.5/recall_Hnear/pooled | 0.099 ± 0.104 (n=5) | 0.041 ± 0.004 (n=5) | 0.136 ± 0.080 (n=5) |
| P2@0.5/recall_Hnear/mean | 0.082 ± 0.046 (n=5) | 0.056 ± 0.038 (n=5) | 0.358 ± 0.090 (n=5) |
| P2@0.5/precision_Hnear/pooled | 0.495 ± 0.226 (n=5) | 0.434 ± 0.115 (n=5) | 0.359 ± 0.129 (n=5) |
| P2@0.5/precision_Hnear/mean | 0.388 ± 0.160 (n=5) | 0.337 ± 0.133 (n=5) | 0.250 ± 0.074 (n=5) |
| P2@0.5/iou_Hnear/pooled | 0.094 ± 0.101 (n=5) | 0.039 ± 0.004 (n=5) | 0.111 ± 0.066 (n=5) |
| P2@0.5/iou_Hnear/mean | 0.072 ± 0.047 (n=5) | 0.043 ± 0.031 (n=5) | 0.102 ± 0.040 (n=5) |
| P1/recall_Hnear/pooled | 0.015 ± 0.005 (n=5) | 0.013 ± 0.006 (n=5) | 0.019 ± 0.015 (n=5) |
| P1/recall_Hnear/mean | 0.108 ± 0.036 (n=5) | 0.186 ± 0.047 (n=5) | 0.194 ± 0.049 (n=5) |
| hidden_near/gt_cells | 137.000 ± 0.000 (n=5) | 137.000 ± 0.000 (n=5) | 137.000 ± 0.000 (n=5) |
| hidden_near/objects_with_hidden_gt | 7.000 ± 0.000 (n=5) | 7.000 ± 0.000 (n=5) | 7.000 ± 0.000 (n=5) |
| P2@0.5/recall_H/pooled | 0.013 ± 0.004 (n=5) | 0.003 ± 0.002 (n=5) | 0.088 ± 0.002 (n=5) |
| P2@0.5/recall_H/mean | 0.030 ± 0.004 (n=5) | 0.008 ± 0.004 (n=5) | 0.144 ± 0.007 (n=5) |
| P2@0.5/precision_H/pooled | 0.462 ± 0.061 (n=5) | 0.443 ± 0.124 (n=5) | 0.554 ± 0.019 (n=5) |
| P2@0.5/precision_H/mean | 0.402 ± 0.053 (n=5) | 0.387 ± 0.116 (n=5) | 0.492 ± 0.036 (n=5) |
| P2@0.5/iou_H/pooled | 0.012 ± 0.004 (n=5) | 0.003 ± 0.002 (n=5) | 0.082 ± 0.002 (n=5) |
| P2@0.5/iou_H/mean | 0.026 ± 0.004 (n=5) | 0.008 ± 0.004 (n=5) | 0.091 ± 0.006 (n=5) |
| hidden/gt_cells | 5634.000 ± 0.000 (n=5) | 5634.000 ± 0.000 (n=5) | 5634.000 ± 0.000 (n=5) |
| hidden/objects_with_hidden_gt | 80.000 ± 0.000 (n=5) | 80.000 ± 0.000 (n=5) | 80.000 ± 0.000 (n=5) |
| P2@0.5/recall/pooled | 0.278 ± 0.003 (n=5) | 0.221 ± 0.003 (n=5) | 0.370 ± 0.003 (n=5) |
| P2@0.5/recall/mean | 0.421 ± 0.003 (n=5) | 0.341 ± 0.002 (n=5) | 0.509 ± 0.006 (n=5) |
| P2@0.5/precision/pooled | 0.838 ± 0.004 (n=5) | 0.810 ± 0.005 (n=5) | 0.758 ± 0.006 (n=5) |
| P2@0.5/precision/mean | 0.703 ± 0.012 (n=5) | 0.676 ± 0.020 (n=5) | 0.690 ± 0.011 (n=5) |
| P2@0.5/iou/pooled | 0.264 ± 0.003 (n=5) | 0.210 ± 0.003 (n=5) | 0.331 ± 0.002 (n=5) |
| P2@0.5/iou/mean | 0.343 ± 0.002 (n=5) | 0.275 ± 0.002 (n=5) | 0.375 ± 0.003 (n=5) |
| P1/recall/pooled | 0.299 ± 0.001 (n=5) | 0.273 ± 0.002 (n=5) | 0.298 ± 0.002 (n=5) |
| P1/recall/mean | 0.433 ± 0.003 (n=5) | 0.404 ± 0.002 (n=5) | 0.430 ± 0.002 (n=5) |
| P1/precision/pooled | 0.680 ± 0.004 (n=5) | 0.660 ± 0.004 (n=5) | 0.673 ± 0.003 (n=5) |
| P1/precision/mean | 0.590 ± 0.007 (n=5) | 0.586 ± 0.005 (n=5) | 0.604 ± 0.010 (n=5) |
| P1/recall_H/pooled | 0.006 ± 0.002 (n=5) | 0.003 ± 0.002 (n=5) | 0.004 ± 0.002 (n=5) |
| P1/recall_H/mean | 0.017 ± 0.004 (n=5) | 0.020 ± 0.003 (n=5) | 0.022 ± 0.007 (n=5) |


## obstacles, P2 occupied at >= 0.8

| metric | may_scene_a | may_scene_a_radar_init | may_scene_a_radar_init_loss |
|---|---|---|---|
| P2@0.8/recall_Hnear/pooled | 0.061 ± 0.069 (n=5) | 0.032 ± 0.004 (n=5) | 0.069 ± 0.044 (n=5) |
| P2@0.8/recall_Hnear/mean | 0.031 ± 0.019 (n=5) | 0.017 ± 0.013 (n=5) | 0.283 ± 0.049 (n=5) |
| P2@0.8/precision_Hnear/pooled | 0.478 ± 0.242 (n=5) | 0.420 ± 0.109 (n=5) | 0.306 ± 0.128 (n=5) |
| P2@0.8/precision_Hnear/mean | 0.385 ± 0.228 (n=5) | 0.287 ± 0.132 (n=5) | 0.254 ± 0.031 (n=5) |
| P2@0.8/iou_Hnear/pooled | 0.059 ± 0.067 (n=5) | 0.031 ± 0.004 (n=5) | 0.060 ± 0.039 (n=5) |
| P2@0.8/iou_Hnear/mean | 0.028 ± 0.019 (n=5) | 0.013 ± 0.009 (n=5) | 0.069 ± 0.029 (n=5) |
| P1/recall_Hnear/pooled | 0.015 ± 0.005 (n=5) | 0.013 ± 0.006 (n=5) | 0.019 ± 0.015 (n=5) |
| P1/recall_Hnear/mean | 0.108 ± 0.036 (n=5) | 0.186 ± 0.047 (n=5) | 0.194 ± 0.049 (n=5) |
| hidden_near/gt_cells | 137.000 ± 0.000 (n=5) | 137.000 ± 0.000 (n=5) | 137.000 ± 0.000 (n=5) |
| hidden_near/objects_with_hidden_gt | 7.000 ± 0.000 (n=5) | 7.000 ± 0.000 (n=5) | 7.000 ± 0.000 (n=5) |
| P2@0.8/recall_H/pooled | 0.007 ± 0.003 (n=5) | 0.003 ± 0.001 (n=5) | 0.065 ± 0.001 (n=5) |
| P2@0.8/recall_H/mean | 0.014 ± 0.003 (n=5) | 0.003 ± 0.001 (n=5) | 0.095 ± 0.006 (n=5) |
| P2@0.8/precision_H/pooled | 0.479 ± 0.099 (n=5) | 0.452 ± 0.128 (n=5) | 0.542 ± 0.008 (n=5) |
| P2@0.8/precision_H/mean | 0.424 ± 0.041 (n=5) | 0.324 ± 0.113 (n=5) | 0.476 ± 0.025 (n=5) |
| P2@0.8/iou_H/pooled | 0.007 ± 0.003 (n=5) | 0.003 ± 0.001 (n=5) | 0.062 ± 0.001 (n=5) |
| P2@0.8/iou_H/mean | 0.013 ± 0.003 (n=5) | 0.003 ± 0.001 (n=5) | 0.056 ± 0.004 (n=5) |
| hidden/gt_cells | 5634.000 ± 0.000 (n=5) | 5634.000 ± 0.000 (n=5) | 5634.000 ± 0.000 (n=5) |
| hidden/objects_with_hidden_gt | 80.000 ± 0.000 (n=5) | 80.000 ± 0.000 (n=5) | 80.000 ± 0.000 (n=5) |
| P2@0.8/recall/pooled | 0.244 ± 0.002 (n=5) | 0.190 ± 0.002 (n=5) | 0.326 ± 0.001 (n=5) |
| P2@0.8/recall/mean | 0.379 ± 0.002 (n=5) | 0.297 ± 0.003 (n=5) | 0.469 ± 0.003 (n=5) |
| P2@0.8/precision/pooled | 0.863 ± 0.003 (n=5) | 0.831 ± 0.002 (n=5) | 0.772 ± 0.003 (n=5) |
| P2@0.8/precision/mean | 0.716 ± 0.012 (n=5) | 0.685 ± 0.012 (n=5) | 0.714 ± 0.008 (n=5) |
| P2@0.8/iou/pooled | 0.234 ± 0.002 (n=5) | 0.183 ± 0.002 (n=5) | 0.297 ± 0.001 (n=5) |
| P2@0.8/iou/mean | 0.316 ± 0.002 (n=5) | 0.246 ± 0.003 (n=5) | 0.351 ± 0.003 (n=5) |
| P1/recall/pooled | 0.299 ± 0.001 (n=5) | 0.273 ± 0.002 (n=5) | 0.298 ± 0.002 (n=5) |
| P1/recall/mean | 0.433 ± 0.003 (n=5) | 0.404 ± 0.002 (n=5) | 0.430 ± 0.002 (n=5) |
| P1/precision/pooled | 0.680 ± 0.004 (n=5) | 0.660 ± 0.004 (n=5) | 0.673 ± 0.003 (n=5) |
| P1/precision/mean | 0.590 ± 0.007 (n=5) | 0.586 ± 0.005 (n=5) | 0.604 ± 0.010 (n=5) |
| P1/recall_H/pooled | 0.006 ± 0.002 (n=5) | 0.003 ± 0.002 (n=5) | 0.004 ± 0.002 (n=5) |
| P1/recall_H/mean | 0.017 ± 0.004 (n=5) | 0.020 ± 0.003 (n=5) | 0.022 ± 0.007 (n=5) |


## tree_trunk, P2 occupied at >= 0.5

| metric | may_scene_a | may_scene_a_radar_init | may_scene_a_radar_init_loss |
|---|---|---|---|
| P2@0.5/recall_Hnear/pooled | 0.200 ± 0.139 (n=5) | 0.133 ± 0.139 (n=5) | 0.200 ± 0.183 (n=5) |
| P2@0.5/recall_Hnear/mean | 0.175 ± 0.112 (n=5) | 0.150 ± 0.137 (n=5) | 0.250 ± 0.177 (n=5) |
| P2@0.5/precision_Hnear/pooled | 0.700 ± 0.447 (n=5) | 0.240 ± 0.277 (n=5) | 0.300 ± 0.183 (n=5) |
| P2@0.5/precision_Hnear/mean | 0.700 ± 0.447 (n=5) | 0.250 ± 0.250 (n=5) | 0.283 ± 0.162 (n=5) |
| P2@0.5/iou_Hnear/pooled | 0.195 ± 0.141 (n=5) | 0.102 ± 0.117 (n=5) | 0.142 ± 0.120 (n=5) |
| P2@0.5/iou_Hnear/mean | 0.167 ± 0.118 (n=5) | 0.092 ± 0.085 (n=5) | 0.145 ± 0.085 (n=5) |
| P1/recall_Hnear/pooled | 0.200 ± 0.075 (n=5) | 0.133 ± 0.139 (n=5) | 0.133 ± 0.139 (n=5) |
| P1/recall_Hnear/mean | 0.275 ± 0.056 (n=5) | 0.150 ± 0.163 (n=5) | 0.175 ± 0.168 (n=5) |
| hidden_near/gt_cells | 6.000 ± 0.000 (n=5) | 6.000 ± 0.000 (n=5) | 6.000 ± 0.000 (n=5) |
| hidden_near/objects_with_hidden_gt | 2.000 ± 0.000 (n=5) | 2.000 ± 0.000 (n=5) | 2.000 ± 0.000 (n=5) |
| P2@0.5/recall_H/pooled | 0.017 ± 0.002 (n=5) | 0.005 ± 0.004 (n=5) | 0.078 ± 0.005 (n=5) |
| P2@0.5/recall_H/mean | 0.036 ± 0.009 (n=5) | 0.010 ± 0.007 (n=5) | 0.138 ± 0.012 (n=5) |
| P2@0.5/precision_H/pooled | 0.506 ± 0.034 (n=5) | 0.477 ± 0.222 (n=5) | 0.679 ± 0.057 (n=5) |
| P2@0.5/precision_H/mean | 0.447 ± 0.059 (n=5) | 0.453 ± 0.154 (n=5) | 0.581 ± 0.047 (n=5) |
| P2@0.5/iou_H/pooled | 0.017 ± 0.002 (n=5) | 0.005 ± 0.004 (n=5) | 0.075 ± 0.004 (n=5) |
| P2@0.5/iou_H/mean | 0.032 ± 0.008 (n=5) | 0.010 ± 0.007 (n=5) | 0.110 ± 0.010 (n=5) |
| hidden/gt_cells | 2469.000 ± 0.000 (n=5) | 2469.000 ± 0.000 (n=5) | 2469.000 ± 0.000 (n=5) |
| hidden/objects_with_hidden_gt | 52.000 ± 0.000 (n=5) | 52.000 ± 0.000 (n=5) | 52.000 ± 0.000 (n=5) |
| P2@0.5/recall/pooled | 0.375 ± 0.002 (n=5) | 0.302 ± 0.001 (n=5) | 0.433 ± 0.004 (n=5) |
| P2@0.5/recall/mean | 0.491 ± 0.002 (n=5) | 0.400 ± 0.002 (n=5) | 0.548 ± 0.006 (n=5) |
| P2@0.5/precision/pooled | 0.849 ± 0.005 (n=5) | 0.828 ± 0.002 (n=5) | 0.811 ± 0.009 (n=5) |
| P2@0.5/precision/mean | 0.744 ± 0.014 (n=5) | 0.727 ± 0.020 (n=5) | 0.752 ± 0.011 (n=5) |
| P2@0.5/iou/pooled | 0.351 ± 0.001 (n=5) | 0.284 ± 0.001 (n=5) | 0.393 ± 0.004 (n=5) |
| P2@0.5/iou/mean | 0.406 ± 0.001 (n=5) | 0.328 ± 0.003 (n=5) | 0.432 ± 0.003 (n=5) |
| P1/recall/pooled | 0.379 ± 0.002 (n=5) | 0.343 ± 0.002 (n=5) | 0.376 ± 0.003 (n=5) |
| P1/recall/mean | 0.493 ± 0.002 (n=5) | 0.453 ± 0.002 (n=5) | 0.486 ± 0.003 (n=5) |
| P1/precision/pooled | 0.711 ± 0.004 (n=5) | 0.687 ± 0.006 (n=5) | 0.712 ± 0.004 (n=5) |
| P1/precision/mean | 0.649 ± 0.008 (n=5) | 0.643 ± 0.007 (n=5) | 0.668 ± 0.013 (n=5) |
| P1/recall_H/pooled | 0.005 ± 0.001 (n=5) | 0.002 ± 0.001 (n=5) | 0.005 ± 0.002 (n=5) |
| P1/recall_H/mean | 0.020 ± 0.006 (n=5) | 0.009 ± 0.006 (n=5) | 0.014 ± 0.011 (n=5) |


## tree_trunk, P2 occupied at >= 0.8

| metric | may_scene_a | may_scene_a_radar_init | may_scene_a_radar_init_loss |
|---|---|---|---|
| P2@0.8/recall_Hnear/pooled | 0.067 ± 0.091 (n=5) | 0.033 ± 0.075 (n=5) | 0.133 ± 0.139 (n=5) |
| P2@0.8/recall_Hnear/mean | 0.050 ± 0.068 (n=5) | 0.025 ± 0.056 (n=5) | 0.175 ± 0.168 (n=5) |
| P2@0.8/precision_Hnear/pooled | 0.500 ± 0.577 (n=4) | 0.100 ± 0.224 (n=5) | 0.213 ± 0.197 (n=5) |
| P2@0.8/precision_Hnear/mean | 0.500 ± 0.577 (n=4) | 0.100 ± 0.224 (n=5) | 0.208 ± 0.191 (n=5) |
| P2@0.8/iou_Hnear/pooled | 0.067 ± 0.091 (n=5) | 0.029 ± 0.064 (n=5) | 0.094 ± 0.095 (n=5) |
| P2@0.8/iou_Hnear/mean | 0.050 ± 0.068 (n=5) | 0.017 ± 0.037 (n=5) | 0.102 ± 0.093 (n=5) |
| P1/recall_Hnear/pooled | 0.200 ± 0.075 (n=5) | 0.133 ± 0.139 (n=5) | 0.133 ± 0.139 (n=5) |
| P1/recall_Hnear/mean | 0.275 ± 0.056 (n=5) | 0.150 ± 0.163 (n=5) | 0.175 ± 0.168 (n=5) |
| hidden_near/gt_cells | 6.000 ± 0.000 (n=5) | 6.000 ± 0.000 (n=5) | 6.000 ± 0.000 (n=5) |
| hidden_near/objects_with_hidden_gt | 2.000 ± 0.000 (n=5) | 2.000 ± 0.000 (n=5) | 2.000 ± 0.000 (n=5) |
| P2@0.8/recall_H/pooled | 0.011 ± 0.002 (n=5) | 0.003 ± 0.003 (n=5) | 0.041 ± 0.004 (n=5) |
| P2@0.8/recall_H/mean | 0.017 ± 0.003 (n=5) | 0.003 ± 0.002 (n=5) | 0.079 ± 0.008 (n=5) |
| P2@0.8/precision_H/pooled | 0.518 ± 0.086 (n=5) | 0.442 ± 0.287 (n=5) | 0.612 ± 0.023 (n=5) |
| P2@0.8/precision_H/mean | 0.452 ± 0.049 (n=5) | 0.354 ± 0.180 (n=5) | 0.535 ± 0.031 (n=5) |
| P2@0.8/iou_H/pooled | 0.011 ± 0.002 (n=5) | 0.003 ± 0.003 (n=5) | 0.040 ± 0.003 (n=5) |
| P2@0.8/iou_H/mean | 0.016 ± 0.003 (n=5) | 0.003 ± 0.002 (n=5) | 0.062 ± 0.005 (n=5) |
| hidden/gt_cells | 2469.000 ± 0.000 (n=5) | 2469.000 ± 0.000 (n=5) | 2469.000 ± 0.000 (n=5) |
| hidden/objects_with_hidden_gt | 52.000 ± 0.000 (n=5) | 52.000 ± 0.000 (n=5) | 52.000 ± 0.000 (n=5) |
| P2@0.8/recall/pooled | 0.331 ± 0.001 (n=5) | 0.263 ± 0.003 (n=5) | 0.379 ± 0.002 (n=5) |
| P2@0.8/recall/mean | 0.443 ± 0.001 (n=5) | 0.352 ± 0.003 (n=5) | 0.504 ± 0.004 (n=5) |
| P2@0.8/precision/pooled | 0.872 ± 0.005 (n=5) | 0.848 ± 0.003 (n=5) | 0.825 ± 0.003 (n=5) |
| P2@0.8/precision/mean | 0.751 ± 0.011 (n=5) | 0.738 ± 0.018 (n=5) | 0.770 ± 0.012 (n=5) |
| P2@0.8/iou/pooled | 0.316 ± 0.001 (n=5) | 0.251 ± 0.003 (n=5) | 0.351 ± 0.002 (n=5) |
| P2@0.8/iou/mean | 0.374 ± 0.001 (n=5) | 0.295 ± 0.003 (n=5) | 0.401 ± 0.003 (n=5) |
| P1/recall/pooled | 0.379 ± 0.002 (n=5) | 0.343 ± 0.002 (n=5) | 0.376 ± 0.003 (n=5) |
| P1/recall/mean | 0.493 ± 0.002 (n=5) | 0.453 ± 0.002 (n=5) | 0.486 ± 0.003 (n=5) |
| P1/precision/pooled | 0.711 ± 0.004 (n=5) | 0.687 ± 0.006 (n=5) | 0.712 ± 0.004 (n=5) |
| P1/precision/mean | 0.649 ± 0.008 (n=5) | 0.643 ± 0.007 (n=5) | 0.668 ± 0.013 (n=5) |
| P1/recall_H/pooled | 0.005 ± 0.001 (n=5) | 0.002 ± 0.001 (n=5) | 0.005 ± 0.002 (n=5) |
| P1/recall_H/mean | 0.020 ± 0.006 (n=5) | 0.009 ± 0.006 (n=5) | 0.014 ± 0.011 (n=5) |


## rock, P2 occupied at >= 0.5

| metric | may_scene_a | may_scene_a_radar_init | may_scene_a_radar_init_loss |
|---|---|---|---|
| P2@0.5/recall_Hnear/pooled | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| P2@0.5/recall_Hnear/mean | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| P2@0.5/precision_Hnear/pooled | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| P2@0.5/precision_Hnear/mean | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| P2@0.5/iou_Hnear/pooled | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| P2@0.5/iou_Hnear/mean | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| P1/recall_Hnear/pooled | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| P1/recall_Hnear/mean | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| hidden_near/gt_cells | 0.000 ± 0.000 (n=5) | 0.000 ± 0.000 (n=5) | 0.000 ± 0.000 (n=5) |
| hidden_near/objects_with_hidden_gt | 0.000 ± 0.000 (n=5) | 0.000 ± 0.000 (n=5) | 0.000 ± 0.000 (n=5) |
| P2@0.5/recall_H/pooled | 0.012 ± 0.022 (n=5) | 0.000 ± 0.000 (n=5) | 0.010 ± 0.018 (n=5) |
| P2@0.5/recall_H/mean | 0.011 ± 0.015 (n=5) | 0.000 ± 0.000 (n=5) | 0.019 ± 0.036 (n=5) |
| P2@0.5/precision_H/pooled | 0.857 ± 0.202 (n=2) | n/a (n=0) | 0.167 ± 0.236 (n=5) |
| P2@0.5/precision_H/mean | 0.708 ± 0.412 (n=2) | n/a (n=0) | 0.133 ± 0.183 (n=5) |
| P2@0.5/iou_H/pooled | 0.012 ± 0.022 (n=5) | 0.000 ± 0.000 (n=5) | 0.010 ± 0.017 (n=5) |
| P2@0.5/iou_H/mean | 0.010 ± 0.014 (n=5) | 0.000 ± 0.000 (n=5) | 0.013 ± 0.024 (n=5) |
| hidden/gt_cells | 97.000 ± 0.000 (n=5) | 97.000 ± 0.000 (n=5) | 97.000 ± 0.000 (n=5) |
| hidden/objects_with_hidden_gt | 12.000 ± 0.000 (n=5) | 12.000 ± 0.000 (n=5) | 12.000 ± 0.000 (n=5) |
| P2@0.5/recall/pooled | 0.253 ± 0.018 (n=5) | 0.206 ± 0.021 (n=5) | 0.284 ± 0.010 (n=5) |
| P2@0.5/recall/mean | 0.237 ± 0.016 (n=5) | 0.195 ± 0.019 (n=5) | 0.282 ± 0.022 (n=5) |
| P2@0.5/precision/pooled | 0.456 ± 0.011 (n=5) | 0.406 ± 0.034 (n=5) | 0.429 ± 0.020 (n=5) |
| P2@0.5/precision/mean | 0.507 ± 0.039 (n=5) | 0.470 ± 0.056 (n=5) | 0.387 ± 0.056 (n=5) |
| P2@0.5/iou/pooled | 0.194 ± 0.012 (n=5) | 0.158 ± 0.017 (n=5) | 0.206 ± 0.008 (n=5) |
| P2@0.5/iou/mean | 0.146 ± 0.013 (n=5) | 0.117 ± 0.011 (n=5) | 0.156 ± 0.017 (n=5) |
| P1/recall/pooled | 0.263 ± 0.023 (n=5) | 0.262 ± 0.005 (n=5) | 0.270 ± 0.016 (n=5) |
| P1/recall/mean | 0.242 ± 0.023 (n=5) | 0.246 ± 0.010 (n=5) | 0.253 ± 0.010 (n=5) |
| P1/precision/pooled | 0.304 ± 0.013 (n=5) | 0.299 ± 0.014 (n=5) | 0.308 ± 0.016 (n=5) |
| P1/precision/mean | 0.281 ± 0.031 (n=5) | 0.300 ± 0.031 (n=5) | 0.286 ± 0.026 (n=5) |
| P1/recall_H/pooled | 0.000 ± 0.000 (n=5) | 0.002 ± 0.005 (n=5) | 0.000 ± 0.000 (n=5) |
| P1/recall_H/mean | 0.000 ± 0.000 (n=5) | 0.008 ± 0.019 (n=5) | 0.000 ± 0.000 (n=5) |


## rock, P2 occupied at >= 0.8

| metric | may_scene_a | may_scene_a_radar_init | may_scene_a_radar_init_loss |
|---|---|---|---|
| P2@0.8/recall_Hnear/pooled | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| P2@0.8/recall_Hnear/mean | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| P2@0.8/precision_Hnear/pooled | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| P2@0.8/precision_Hnear/mean | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| P2@0.8/iou_Hnear/pooled | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| P2@0.8/iou_Hnear/mean | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| P1/recall_Hnear/pooled | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| P1/recall_Hnear/mean | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| hidden_near/gt_cells | 0.000 ± 0.000 (n=5) | 0.000 ± 0.000 (n=5) | 0.000 ± 0.000 (n=5) |
| hidden_near/objects_with_hidden_gt | 0.000 ± 0.000 (n=5) | 0.000 ± 0.000 (n=5) | 0.000 ± 0.000 (n=5) |
| P2@0.8/recall_H/pooled | 0.004 ± 0.006 (n=5) | 0.000 ± 0.000 (n=5) | 0.004 ± 0.009 (n=5) |
| P2@0.8/recall_H/mean | 0.007 ± 0.012 (n=5) | 0.000 ± 0.000 (n=5) | 0.008 ± 0.019 (n=5) |
| P2@0.8/precision_H/pooled | 0.750 ± 0.354 (n=2) | n/a (n=0) | 0.333 ± 0.577 (n=3) |
| P2@0.8/precision_H/mean | 0.750 ± 0.354 (n=2) | n/a (n=0) | 0.333 ± 0.577 (n=3) |
| P2@0.8/iou_H/pooled | 0.004 ± 0.006 (n=5) | 0.000 ± 0.000 (n=5) | 0.004 ± 0.009 (n=5) |
| P2@0.8/iou_H/mean | 0.007 ± 0.012 (n=5) | 0.000 ± 0.000 (n=5) | 0.008 ± 0.019 (n=5) |
| hidden/gt_cells | 97.000 ± 0.000 (n=5) | 97.000 ± 0.000 (n=5) | 97.000 ± 0.000 (n=5) |
| hidden/objects_with_hidden_gt | 12.000 ± 0.000 (n=5) | 12.000 ± 0.000 (n=5) | 12.000 ± 0.000 (n=5) |
| P2@0.8/recall/pooled | 0.228 ± 0.015 (n=5) | 0.165 ± 0.011 (n=5) | 0.267 ± 0.010 (n=5) |
| P2@0.8/recall/mean | 0.211 ± 0.017 (n=5) | 0.154 ± 0.011 (n=5) | 0.263 ± 0.010 (n=5) |
| P2@0.8/precision/pooled | 0.478 ± 0.022 (n=5) | 0.408 ± 0.025 (n=5) | 0.458 ± 0.011 (n=5) |
| P2@0.8/precision/mean | 0.526 ± 0.039 (n=5) | 0.477 ± 0.061 (n=5) | 0.460 ± 0.047 (n=5) |
| P2@0.8/iou/pooled | 0.183 ± 0.012 (n=5) | 0.133 ± 0.010 (n=5) | 0.203 ± 0.007 (n=5) |
| P2@0.8/iou/mean | 0.140 ± 0.014 (n=5) | 0.105 ± 0.009 (n=5) | 0.156 ± 0.009 (n=5) |
| P1/recall/pooled | 0.263 ± 0.023 (n=5) | 0.262 ± 0.005 (n=5) | 0.270 ± 0.016 (n=5) |
| P1/recall/mean | 0.242 ± 0.023 (n=5) | 0.246 ± 0.010 (n=5) | 0.253 ± 0.010 (n=5) |
| P1/precision/pooled | 0.304 ± 0.013 (n=5) | 0.299 ± 0.014 (n=5) | 0.308 ± 0.016 (n=5) |
| P1/precision/mean | 0.281 ± 0.031 (n=5) | 0.300 ± 0.031 (n=5) | 0.286 ± 0.026 (n=5) |
| P1/recall_H/pooled | 0.000 ± 0.000 (n=5) | 0.002 ± 0.005 (n=5) | 0.000 ± 0.000 (n=5) |
| P1/recall_H/mean | 0.000 ± 0.000 (n=5) | 0.008 ± 0.019 (n=5) | 0.000 ± 0.000 (n=5) |


## bush_or_small_tree, P2 occupied at >= 0.5

| metric | may_scene_a | may_scene_a_radar_init | may_scene_a_radar_init_loss |
|---|---|---|---|
| P2@0.5/recall_Hnear/pooled | 0.095 ± 0.110 (n=5) | 0.037 ± 0.003 (n=5) | 0.133 ± 0.076 (n=5) |
| P2@0.5/recall_Hnear/mean | 0.045 ± 0.044 (n=5) | 0.018 ± 0.006 (n=5) | 0.402 ± 0.098 (n=5) |
| P2@0.5/precision_Hnear/pooled | 0.478 ± 0.248 (n=5) | 0.493 ± 0.076 (n=5) | 0.363 ± 0.130 (n=5) |
| P2@0.5/precision_Hnear/mean | 0.304 ± 0.190 (n=5) | 0.382 ± 0.144 (n=5) | 0.227 ± 0.064 (n=5) |
| P2@0.5/iou_Hnear/pooled | 0.090 ± 0.106 (n=5) | 0.035 ± 0.003 (n=5) | 0.110 ± 0.063 (n=5) |
| P2@0.5/iou_Hnear/mean | 0.040 ± 0.040 (n=5) | 0.015 ± 0.006 (n=5) | 0.080 ± 0.030 (n=5) |
| P1/recall_Hnear/pooled | 0.006 ± 0.006 (n=5) | 0.008 ± 0.000 (n=5) | 0.014 ± 0.014 (n=5) |
| P1/recall_Hnear/mean | 0.041 ± 0.055 (n=5) | 0.200 ± 0.000 (n=5) | 0.201 ± 0.003 (n=5) |
| hidden_near/gt_cells | 131.000 ± 0.000 (n=5) | 131.000 ± 0.000 (n=5) | 131.000 ± 0.000 (n=5) |
| hidden_near/objects_with_hidden_gt | 5.000 ± 0.000 (n=5) | 5.000 ± 0.000 (n=5) | 5.000 ± 0.000 (n=5) |
| P2@0.5/recall_H/pooled | 0.009 ± 0.007 (n=5) | 0.003 ± 0.001 (n=5) | 0.099 ± 0.001 (n=5) |
| P2@0.5/recall_H/mean | 0.025 ± 0.022 (n=5) | 0.007 ± 0.002 (n=5) | 0.256 ± 0.028 (n=5) |
| P2@0.5/precision_H/pooled | 0.384 ± 0.151 (n=5) | 0.365 ± 0.123 (n=5) | 0.500 ± 0.012 (n=5) |
| P2@0.5/precision_H/mean | 0.269 ± 0.132 (n=5) | 0.304 ± 0.119 (n=5) | 0.343 ± 0.033 (n=5) |
| P2@0.5/iou_H/pooled | 0.009 ± 0.007 (n=5) | 0.003 ± 0.001 (n=5) | 0.090 ± 0.001 (n=5) |
| P2@0.5/iou_H/mean | 0.019 ± 0.016 (n=5) | 0.006 ± 0.002 (n=5) | 0.081 ± 0.011 (n=5) |
| hidden/gt_cells | 3068.000 ± 0.000 (n=5) | 3068.000 ± 0.000 (n=5) | 3068.000 ± 0.000 (n=5) |
| hidden/objects_with_hidden_gt | 16.000 ± 0.000 (n=5) | 16.000 ± 0.000 (n=5) | 16.000 ± 0.000 (n=5) |
| P2@0.5/recall/pooled | 0.127 ± 0.007 (n=5) | 0.093 ± 0.006 (n=5) | 0.274 ± 0.005 (n=5) |
| P2@0.5/recall/mean | 0.185 ± 0.012 (n=5) | 0.131 ± 0.008 (n=5) | 0.451 ± 0.006 (n=5) |
| P2@0.5/precision/pooled | 0.819 ± 0.012 (n=5) | 0.763 ± 0.020 (n=5) | 0.663 ± 0.007 (n=5) |
| P2@0.5/precision/mean | 0.547 ± 0.033 (n=5) | 0.481 ± 0.043 (n=5) | 0.490 ± 0.015 (n=5) |
| P2@0.5/iou/pooled | 0.124 ± 0.007 (n=5) | 0.091 ± 0.006 (n=5) | 0.241 ± 0.004 (n=5) |
| P2@0.5/iou/mean | 0.147 ± 0.008 (n=5) | 0.104 ± 0.007 (n=5) | 0.230 ± 0.003 (n=5) |
| P1/recall/pooled | 0.174 ± 0.002 (n=5) | 0.165 ± 0.003 (n=5) | 0.175 ± 0.001 (n=5) |
| P1/recall/mean | 0.252 ± 0.003 (n=5) | 0.250 ± 0.002 (n=5) | 0.260 ± 0.006 (n=5) |
| P1/precision/pooled | 0.615 ± 0.007 (n=5) | 0.607 ± 0.011 (n=5) | 0.589 ± 0.007 (n=5) |
| P1/precision/mean | 0.376 ± 0.021 (n=5) | 0.398 ± 0.020 (n=5) | 0.360 ± 0.021 (n=5) |
| P1/recall_H/pooled | 0.007 ± 0.003 (n=5) | 0.005 ± 0.003 (n=5) | 0.004 ± 0.003 (n=5) |
| P1/recall_H/mean | 0.022 ± 0.017 (n=5) | 0.067 ± 0.005 (n=5) | 0.066 ± 0.003 (n=5) |


## bush_or_small_tree, P2 occupied at >= 0.8

| metric | may_scene_a | may_scene_a_radar_init | may_scene_a_radar_init_loss |
|---|---|---|---|
| P2@0.8/recall_Hnear/pooled | 0.061 ± 0.073 (n=5) | 0.032 ± 0.006 (n=5) | 0.066 ± 0.041 (n=5) |
| P2@0.8/recall_Hnear/mean | 0.024 ± 0.027 (n=5) | 0.014 ± 0.008 (n=5) | 0.326 ± 0.007 (n=5) |
| P2@0.8/precision_Hnear/pooled | 0.476 ± 0.237 (n=5) | 0.489 ± 0.094 (n=5) | 0.313 ± 0.138 (n=5) |
| P2@0.8/precision_Hnear/mean | 0.347 ± 0.231 (n=5) | 0.370 ± 0.170 (n=5) | 0.248 ± 0.057 (n=5) |
| P2@0.8/iou_Hnear/pooled | 0.059 ± 0.072 (n=5) | 0.031 ± 0.006 (n=5) | 0.058 ± 0.037 (n=5) |
| P2@0.8/iou_Hnear/mean | 0.021 ± 0.025 (n=5) | 0.013 ± 0.008 (n=5) | 0.053 ± 0.007 (n=5) |
| P1/recall_Hnear/pooled | 0.006 ± 0.006 (n=5) | 0.008 ± 0.000 (n=5) | 0.014 ± 0.014 (n=5) |
| P1/recall_Hnear/mean | 0.041 ± 0.055 (n=5) | 0.200 ± 0.000 (n=5) | 0.201 ± 0.003 (n=5) |
| hidden_near/gt_cells | 131.000 ± 0.000 (n=5) | 131.000 ± 0.000 (n=5) | 131.000 ± 0.000 (n=5) |
| hidden_near/objects_with_hidden_gt | 5.000 ± 0.000 (n=5) | 5.000 ± 0.000 (n=5) | 5.000 ± 0.000 (n=5) |
| P2@0.8/recall_H/pooled | 0.004 ± 0.003 (n=5) | 0.002 ± 0.001 (n=5) | 0.086 ± 0.001 (n=5) |
| P2@0.8/recall_H/mean | 0.011 ± 0.008 (n=5) | 0.005 ± 0.002 (n=5) | 0.214 ± 0.010 (n=5) |
| P2@0.8/precision_H/pooled | 0.375 ± 0.157 (n=5) | 0.412 ± 0.121 (n=5) | 0.519 ± 0.006 (n=5) |
| P2@0.8/precision_H/mean | 0.322 ± 0.076 (n=5) | 0.284 ± 0.132 (n=5) | 0.395 ± 0.037 (n=5) |
| P2@0.8/iou_H/pooled | 0.004 ± 0.003 (n=5) | 0.002 ± 0.001 (n=5) | 0.080 ± 0.001 (n=5) |
| P2@0.8/iou_H/mean | 0.009 ± 0.007 (n=5) | 0.005 ± 0.002 (n=5) | 0.067 ± 0.002 (n=5) |
| hidden/gt_cells | 3068.000 ± 0.000 (n=5) | 3068.000 ± 0.000 (n=5) | 3068.000 ± 0.000 (n=5) |
| hidden/objects_with_hidden_gt | 16.000 ± 0.000 (n=5) | 16.000 ± 0.000 (n=5) | 16.000 ± 0.000 (n=5) |
| P2@0.8/recall/pooled | 0.106 ± 0.005 (n=5) | 0.076 ± 0.003 (n=5) | 0.244 ± 0.003 (n=5) |
| P2@0.8/recall/mean | 0.159 ± 0.003 (n=5) | 0.107 ± 0.007 (n=5) | 0.422 ± 0.004 (n=5) |
| P2@0.8/precision/pooled | 0.853 ± 0.011 (n=5) | 0.779 ± 0.020 (n=5) | 0.677 ± 0.004 (n=5) |
| P2@0.8/precision/mean | 0.598 ± 0.027 (n=5) | 0.487 ± 0.019 (n=5) | 0.531 ± 0.021 (n=5) |
| P2@0.8/iou/pooled | 0.104 ± 0.005 (n=5) | 0.075 ± 0.003 (n=5) | 0.218 ± 0.002 (n=5) |
| P2@0.8/iou/mean | 0.131 ± 0.002 (n=5) | 0.090 ± 0.007 (n=5) | 0.224 ± 0.002 (n=5) |
| P1/recall/pooled | 0.174 ± 0.002 (n=5) | 0.165 ± 0.003 (n=5) | 0.175 ± 0.001 (n=5) |
| P1/recall/mean | 0.252 ± 0.003 (n=5) | 0.250 ± 0.002 (n=5) | 0.260 ± 0.006 (n=5) |
| P1/precision/pooled | 0.615 ± 0.007 (n=5) | 0.607 ± 0.011 (n=5) | 0.589 ± 0.007 (n=5) |
| P1/precision/mean | 0.376 ± 0.021 (n=5) | 0.398 ± 0.020 (n=5) | 0.360 ± 0.021 (n=5) |
| P1/recall_H/pooled | 0.007 ± 0.003 (n=5) | 0.005 ± 0.003 (n=5) | 0.004 ± 0.003 (n=5) |
| P1/recall_H/mean | 0.022 ± 0.017 (n=5) | 0.067 ± 0.005 (n=5) | 0.066 ± 0.003 (n=5) |


## Arm pairs (gap = first - second; separated = the two arms' [min, max] do not overlap)

### may_scene_a_radar_init - may_scene_a

#### obstacles, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| obstacles/P2@0.5/recall_Hnear/pooled | -0.058 | 0.004 | 0.104 | 0.007 | 0.241 | False |
| obstacles/P2@0.5/recall_Hnear/mean | -0.027 | 0.038 | 0.046 | 0.071 | 0.129 | False |
| obstacles/P2@0.5/precision_Hnear/pooled | -0.061 | 0.115 | 0.226 | 0.287 | 0.565 | False |
| obstacles/P2@0.5/precision_Hnear/mean | -0.051 | 0.133 | 0.160 | 0.352 | 0.386 | False |
| obstacles/P2@0.5/iou_Hnear/pooled | -0.055 | 0.004 | 0.101 | 0.009 | 0.233 | False |
| obstacles/P2@0.5/iou_Hnear/mean | -0.029 | 0.031 | 0.047 | 0.066 | 0.128 | False |
| obstacles/P1/recall_Hnear/pooled | -0.001 | 0.006 | 0.005 | 0.015 | 0.015 | False |
| obstacles/P1/recall_Hnear/mean | 0.078 | 0.047 | 0.036 | 0.107 | 0.073 | False |
| obstacles/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/P2@0.5/recall_H/pooled | -0.009 | 0.002 | 0.004 | 0.004 | 0.009 | True |
| obstacles/P2@0.5/recall_H/mean | -0.022 | 0.004 | 0.004 | 0.011 | 0.010 | True |
| obstacles/P2@0.5/precision_H/pooled | -0.019 | 0.124 | 0.061 | 0.324 | 0.170 | False |
| obstacles/P2@0.5/precision_H/mean | -0.015 | 0.116 | 0.053 | 0.293 | 0.134 | False |
| obstacles/P2@0.5/iou_H/pooled | -0.009 | 0.002 | 0.004 | 0.004 | 0.009 | True |
| obstacles/P2@0.5/iou_H/mean | -0.019 | 0.004 | 0.004 | 0.011 | 0.010 | True |
| obstacles/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/P2@0.5/recall/pooled | -0.058 | 0.003 | 0.003 | 0.007 | 0.008 | True |
| obstacles/P2@0.5/recall/mean | -0.081 | 0.002 | 0.003 | 0.005 | 0.007 | True |
| obstacles/P2@0.5/precision/pooled | -0.027 | 0.005 | 0.004 | 0.012 | 0.009 | True |
| obstacles/P2@0.5/precision/mean | -0.027 | 0.020 | 0.012 | 0.052 | 0.035 | False |
| obstacles/P2@0.5/iou/pooled | -0.054 | 0.003 | 0.003 | 0.007 | 0.007 | True |
| obstacles/P2@0.5/iou/mean | -0.068 | 0.002 | 0.002 | 0.006 | 0.005 | True |
| obstacles/P1/recall/pooled | -0.026 | 0.002 | 0.001 | 0.005 | 0.003 | True |
| obstacles/P1/recall/mean | -0.030 | 0.002 | 0.003 | 0.005 | 0.008 | True |
| obstacles/P1/precision/pooled | -0.020 | 0.004 | 0.004 | 0.009 | 0.008 | True |
| obstacles/P1/precision/mean | -0.003 | 0.005 | 0.007 | 0.010 | 0.016 | False |
| obstacles/P1/recall_H/pooled | -0.003 | 0.002 | 0.002 | 0.004 | 0.004 | False |
| obstacles/P1/recall_H/mean | 0.003 | 0.003 | 0.004 | 0.007 | 0.011 | False |

#### obstacles, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| obstacles/P2@0.8/recall_Hnear/pooled | -0.029 | 0.004 | 0.069 | 0.007 | 0.168 | False |
| obstacles/P2@0.8/recall_Hnear/mean | -0.014 | 0.013 | 0.019 | 0.034 | 0.046 | False |
| obstacles/P2@0.8/precision_Hnear/pooled | -0.058 | 0.109 | 0.242 | 0.270 | 0.607 | False |
| obstacles/P2@0.8/precision_Hnear/mean | -0.098 | 0.132 | 0.228 | 0.300 | 0.551 | False |
| obstacles/P2@0.8/iou_Hnear/pooled | -0.029 | 0.004 | 0.067 | 0.008 | 0.164 | False |
| obstacles/P2@0.8/iou_Hnear/mean | -0.016 | 0.009 | 0.019 | 0.024 | 0.044 | False |
| obstacles/P1/recall_Hnear/pooled | -0.001 | 0.006 | 0.005 | 0.015 | 0.015 | False |
| obstacles/P1/recall_Hnear/mean | 0.078 | 0.047 | 0.036 | 0.107 | 0.073 | False |
| obstacles/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/P2@0.8/recall_H/pooled | -0.004 | 0.001 | 0.003 | 0.003 | 0.006 | True |
| obstacles/P2@0.8/recall_H/mean | -0.011 | 0.001 | 0.003 | 0.002 | 0.007 | True |
| obstacles/P2@0.8/precision_H/pooled | -0.026 | 0.128 | 0.099 | 0.335 | 0.246 | False |
| obstacles/P2@0.8/precision_H/mean | -0.100 | 0.113 | 0.041 | 0.289 | 0.100 | False |
| obstacles/P2@0.8/iou_H/pooled | -0.004 | 0.001 | 0.003 | 0.003 | 0.006 | True |
| obstacles/P2@0.8/iou_H/mean | -0.010 | 0.001 | 0.003 | 0.002 | 0.007 | True |
| obstacles/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/P2@0.8/recall/pooled | -0.053 | 0.002 | 0.002 | 0.006 | 0.006 | True |
| obstacles/P2@0.8/recall/mean | -0.082 | 0.003 | 0.002 | 0.007 | 0.005 | True |
| obstacles/P2@0.8/precision/pooled | -0.032 | 0.002 | 0.003 | 0.007 | 0.006 | True |
| obstacles/P2@0.8/precision/mean | -0.031 | 0.012 | 0.012 | 0.027 | 0.027 | True |
| obstacles/P2@0.8/iou/pooled | -0.051 | 0.002 | 0.002 | 0.006 | 0.005 | True |
| obstacles/P2@0.8/iou/mean | -0.069 | 0.003 | 0.002 | 0.007 | 0.004 | True |
| obstacles/P1/recall/pooled | -0.026 | 0.002 | 0.001 | 0.005 | 0.003 | True |
| obstacles/P1/recall/mean | -0.030 | 0.002 | 0.003 | 0.005 | 0.008 | True |
| obstacles/P1/precision/pooled | -0.020 | 0.004 | 0.004 | 0.009 | 0.008 | True |
| obstacles/P1/precision/mean | -0.003 | 0.005 | 0.007 | 0.010 | 0.016 | False |
| obstacles/P1/recall_H/pooled | -0.003 | 0.002 | 0.002 | 0.004 | 0.004 | False |
| obstacles/P1/recall_H/mean | 0.003 | 0.003 | 0.004 | 0.007 | 0.011 | False |

#### tree_trunk, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| tree_trunk/P2@0.5/recall_Hnear/pooled | -0.067 | 0.139 | 0.139 | 0.333 | 0.333 | False |
| tree_trunk/P2@0.5/recall_Hnear/mean | -0.025 | 0.137 | 0.112 | 0.250 | 0.250 | False |
| tree_trunk/P2@0.5/precision_Hnear/pooled | -0.460 | 0.277 | 0.447 | 0.667 | 1.000 | False |
| tree_trunk/P2@0.5/precision_Hnear/mean | -0.450 | 0.250 | 0.447 | 0.500 | 1.000 | False |
| tree_trunk/P2@0.5/iou_Hnear/pooled | -0.093 | 0.117 | 0.141 | 0.286 | 0.333 | False |
| tree_trunk/P2@0.5/iou_Hnear/mean | -0.075 | 0.085 | 0.118 | 0.167 | 0.250 | False |
| tree_trunk/P1/recall_Hnear/pooled | -0.067 | 0.139 | 0.075 | 0.333 | 0.167 | False |
| tree_trunk/P1/recall_Hnear/mean | -0.125 | 0.163 | 0.056 | 0.375 | 0.125 | False |
| tree_trunk/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/P2@0.5/recall_H/pooled | -0.013 | 0.004 | 0.002 | 0.010 | 0.004 | True |
| tree_trunk/P2@0.5/recall_H/mean | -0.026 | 0.007 | 0.009 | 0.017 | 0.022 | True |
| tree_trunk/P2@0.5/precision_H/pooled | -0.029 | 0.222 | 0.034 | 0.514 | 0.086 | False |
| tree_trunk/P2@0.5/precision_H/mean | 0.006 | 0.154 | 0.059 | 0.414 | 0.123 | False |
| tree_trunk/P2@0.5/iou_H/pooled | -0.012 | 0.004 | 0.002 | 0.010 | 0.004 | True |
| tree_trunk/P2@0.5/iou_H/mean | -0.022 | 0.007 | 0.008 | 0.016 | 0.020 | True |
| tree_trunk/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/P2@0.5/recall/pooled | -0.073 | 0.001 | 0.002 | 0.003 | 0.004 | True |
| tree_trunk/P2@0.5/recall/mean | -0.090 | 0.002 | 0.002 | 0.005 | 0.004 | True |
| tree_trunk/P2@0.5/precision/pooled | -0.021 | 0.002 | 0.005 | 0.004 | 0.011 | True |
| tree_trunk/P2@0.5/precision/mean | -0.017 | 0.020 | 0.014 | 0.047 | 0.038 | False |
| tree_trunk/P2@0.5/iou/pooled | -0.067 | 0.001 | 0.001 | 0.003 | 0.003 | True |
| tree_trunk/P2@0.5/iou/mean | -0.078 | 0.003 | 0.001 | 0.006 | 0.004 | True |
| tree_trunk/P1/recall/pooled | -0.037 | 0.002 | 0.002 | 0.005 | 0.005 | True |
| tree_trunk/P1/recall/mean | -0.039 | 0.002 | 0.002 | 0.007 | 0.006 | True |
| tree_trunk/P1/precision/pooled | -0.024 | 0.006 | 0.004 | 0.015 | 0.009 | True |
| tree_trunk/P1/precision/mean | -0.006 | 0.007 | 0.008 | 0.017 | 0.021 | False |
| tree_trunk/P1/recall_H/pooled | -0.004 | 0.001 | 0.001 | 0.002 | 0.002 | True |
| tree_trunk/P1/recall_H/mean | -0.011 | 0.006 | 0.006 | 0.015 | 0.015 | False |

#### tree_trunk, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| tree_trunk/P2@0.8/recall_Hnear/pooled | -0.033 | 0.075 | 0.091 | 0.167 | 0.167 | False |
| tree_trunk/P2@0.8/recall_Hnear/mean | -0.025 | 0.056 | 0.068 | 0.125 | 0.125 | False |
| tree_trunk/P2@0.8/precision_Hnear/pooled | -0.400 | 0.224 | 0.577 | 0.500 | 1.000 | False |
| tree_trunk/P2@0.8/precision_Hnear/mean | -0.400 | 0.224 | 0.577 | 0.500 | 1.000 | False |
| tree_trunk/P2@0.8/iou_Hnear/pooled | -0.038 | 0.064 | 0.091 | 0.143 | 0.167 | False |
| tree_trunk/P2@0.8/iou_Hnear/mean | -0.033 | 0.037 | 0.068 | 0.083 | 0.125 | False |
| tree_trunk/P1/recall_Hnear/pooled | -0.067 | 0.139 | 0.075 | 0.333 | 0.167 | False |
| tree_trunk/P1/recall_Hnear/mean | -0.125 | 0.163 | 0.056 | 0.375 | 0.125 | False |
| tree_trunk/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/P2@0.8/recall_H/pooled | -0.008 | 0.003 | 0.002 | 0.007 | 0.005 | True |
| tree_trunk/P2@0.8/recall_H/mean | -0.014 | 0.002 | 0.003 | 0.005 | 0.008 | True |
| tree_trunk/P2@0.8/precision_H/pooled | -0.076 | 0.287 | 0.086 | 0.607 | 0.230 | False |
| tree_trunk/P2@0.8/precision_H/mean | -0.098 | 0.180 | 0.049 | 0.457 | 0.114 | False |
| tree_trunk/P2@0.8/iou_H/pooled | -0.008 | 0.003 | 0.002 | 0.007 | 0.005 | True |
| tree_trunk/P2@0.8/iou_H/mean | -0.013 | 0.002 | 0.003 | 0.005 | 0.008 | True |
| tree_trunk/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/P2@0.8/recall/pooled | -0.068 | 0.003 | 0.001 | 0.007 | 0.004 | True |
| tree_trunk/P2@0.8/recall/mean | -0.091 | 0.003 | 0.001 | 0.008 | 0.003 | True |
| tree_trunk/P2@0.8/precision/pooled | -0.024 | 0.003 | 0.005 | 0.007 | 0.010 | True |
| tree_trunk/P2@0.8/precision/mean | -0.013 | 0.018 | 0.011 | 0.048 | 0.026 | False |
| tree_trunk/P2@0.8/iou/pooled | -0.064 | 0.003 | 0.001 | 0.007 | 0.002 | True |
| tree_trunk/P2@0.8/iou/mean | -0.079 | 0.003 | 0.001 | 0.007 | 0.003 | True |
| tree_trunk/P1/recall/pooled | -0.037 | 0.002 | 0.002 | 0.005 | 0.005 | True |
| tree_trunk/P1/recall/mean | -0.039 | 0.002 | 0.002 | 0.007 | 0.006 | True |
| tree_trunk/P1/precision/pooled | -0.024 | 0.006 | 0.004 | 0.015 | 0.009 | True |
| tree_trunk/P1/precision/mean | -0.006 | 0.007 | 0.008 | 0.017 | 0.021 | False |
| tree_trunk/P1/recall_H/pooled | -0.004 | 0.001 | 0.001 | 0.002 | 0.002 | True |
| tree_trunk/P1/recall_H/mean | -0.011 | 0.006 | 0.006 | 0.015 | 0.015 | False |

#### rock, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| rock/P2@0.5/recall_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.5/recall_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.5/precision_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.5/precision_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.5/iou_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.5/iou_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/P1/recall_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P1/recall_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P2@0.5/recall_H/pooled | -0.012 | 0.000 | 0.022 | 0.000 | 0.052 | False |
| rock/P2@0.5/recall_H/mean | -0.011 | 0.000 | 0.015 | 0.000 | 0.028 | False |
| rock/P2@0.5/precision_H/pooled | n/a | n/a | 0.202 | n/a | 0.286 | None |
| rock/P2@0.5/precision_H/mean | n/a | n/a | 0.412 | n/a | 0.583 | None |
| rock/P2@0.5/iou_H/pooled | -0.012 | 0.000 | 0.022 | 0.000 | 0.051 | False |
| rock/P2@0.5/iou_H/mean | -0.010 | 0.000 | 0.014 | 0.000 | 0.028 | False |
| rock/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P2@0.5/recall/pooled | -0.047 | 0.021 | 0.018 | 0.045 | 0.051 | True |
| rock/P2@0.5/recall/mean | -0.042 | 0.019 | 0.016 | 0.038 | 0.037 | True |
| rock/P2@0.5/precision/pooled | -0.050 | 0.034 | 0.011 | 0.088 | 0.031 | False |
| rock/P2@0.5/precision/mean | -0.037 | 0.056 | 0.039 | 0.138 | 0.093 | False |
| rock/P2@0.5/iou/pooled | -0.036 | 0.017 | 0.012 | 0.037 | 0.034 | True |
| rock/P2@0.5/iou/mean | -0.028 | 0.011 | 0.013 | 0.023 | 0.027 | True |
| rock/P1/recall/pooled | -0.001 | 0.005 | 0.023 | 0.011 | 0.051 | False |
| rock/P1/recall/mean | 0.003 | 0.010 | 0.023 | 0.024 | 0.054 | False |
| rock/P1/precision/pooled | -0.005 | 0.014 | 0.013 | 0.030 | 0.035 | False |
| rock/P1/precision/mean | 0.020 | 0.031 | 0.031 | 0.074 | 0.079 | False |
| rock/P1/recall_H/pooled | 0.002 | 0.005 | 0.000 | 0.010 | 0.000 | False |
| rock/P1/recall_H/mean | 0.008 | 0.019 | 0.000 | 0.042 | 0.000 | False |

#### rock, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| rock/P2@0.8/recall_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.8/recall_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.8/precision_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.8/precision_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.8/iou_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.8/iou_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/P1/recall_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P1/recall_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P2@0.8/recall_H/pooled | -0.004 | 0.000 | 0.006 | 0.000 | 0.010 | False |
| rock/P2@0.8/recall_H/mean | -0.007 | 0.000 | 0.012 | 0.000 | 0.028 | False |
| rock/P2@0.8/precision_H/pooled | n/a | n/a | 0.354 | n/a | 0.500 | None |
| rock/P2@0.8/precision_H/mean | n/a | n/a | 0.354 | n/a | 0.500 | None |
| rock/P2@0.8/iou_H/pooled | -0.004 | 0.000 | 0.006 | 0.000 | 0.010 | False |
| rock/P2@0.8/iou_H/mean | -0.007 | 0.000 | 0.012 | 0.000 | 0.028 | False |
| rock/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P2@0.8/recall/pooled | -0.063 | 0.011 | 0.015 | 0.022 | 0.034 | True |
| rock/P2@0.8/recall/mean | -0.057 | 0.011 | 0.017 | 0.027 | 0.046 | True |
| rock/P2@0.8/precision/pooled | -0.070 | 0.025 | 0.022 | 0.065 | 0.056 | True |
| rock/P2@0.8/precision/mean | -0.049 | 0.061 | 0.039 | 0.141 | 0.108 | False |
| rock/P2@0.8/iou/pooled | -0.049 | 0.010 | 0.012 | 0.022 | 0.029 | True |
| rock/P2@0.8/iou/mean | -0.035 | 0.009 | 0.014 | 0.025 | 0.038 | True |
| rock/P1/recall/pooled | -0.001 | 0.005 | 0.023 | 0.011 | 0.051 | False |
| rock/P1/recall/mean | 0.003 | 0.010 | 0.023 | 0.024 | 0.054 | False |
| rock/P1/precision/pooled | -0.005 | 0.014 | 0.013 | 0.030 | 0.035 | False |
| rock/P1/precision/mean | 0.020 | 0.031 | 0.031 | 0.074 | 0.079 | False |
| rock/P1/recall_H/pooled | 0.002 | 0.005 | 0.000 | 0.010 | 0.000 | False |
| rock/P1/recall_H/mean | 0.008 | 0.019 | 0.000 | 0.042 | 0.000 | False |

#### bush_or_small_tree, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| bush_or_small_tree/P2@0.5/recall_Hnear/pooled | -0.058 | 0.003 | 0.110 | 0.008 | 0.252 | False |
| bush_or_small_tree/P2@0.5/recall_Hnear/mean | -0.027 | 0.006 | 0.044 | 0.013 | 0.105 | False |
| bush_or_small_tree/P2@0.5/precision_Hnear/pooled | 0.016 | 0.076 | 0.248 | 0.187 | 0.590 | False |
| bush_or_small_tree/P2@0.5/precision_Hnear/mean | 0.078 | 0.144 | 0.190 | 0.350 | 0.414 | False |
| bush_or_small_tree/P2@0.5/iou_Hnear/pooled | -0.054 | 0.003 | 0.106 | 0.007 | 0.244 | False |
| bush_or_small_tree/P2@0.5/iou_Hnear/mean | -0.025 | 0.006 | 0.040 | 0.015 | 0.099 | False |
| bush_or_small_tree/P1/recall_Hnear/pooled | 0.002 | 0.000 | 0.006 | 0.000 | 0.015 | False |
| bush_or_small_tree/P1/recall_Hnear/mean | 0.159 | 0.000 | 0.055 | 0.000 | 0.102 | True |
| bush_or_small_tree/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/P2@0.5/recall_H/pooled | -0.006 | 0.001 | 0.007 | 0.003 | 0.016 | False |
| bush_or_small_tree/P2@0.5/recall_H/mean | -0.018 | 0.002 | 0.022 | 0.006 | 0.051 | False |
| bush_or_small_tree/P2@0.5/precision_H/pooled | -0.020 | 0.123 | 0.151 | 0.248 | 0.396 | False |
| bush_or_small_tree/P2@0.5/precision_H/mean | 0.035 | 0.119 | 0.132 | 0.282 | 0.349 | False |
| bush_or_small_tree/P2@0.5/iou_H/pooled | -0.006 | 0.001 | 0.007 | 0.003 | 0.016 | False |
| bush_or_small_tree/P2@0.5/iou_H/mean | -0.013 | 0.002 | 0.016 | 0.006 | 0.036 | False |
| bush_or_small_tree/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/P2@0.5/recall/pooled | -0.034 | 0.006 | 0.007 | 0.015 | 0.016 | True |
| bush_or_small_tree/P2@0.5/recall/mean | -0.054 | 0.008 | 0.012 | 0.020 | 0.034 | True |
| bush_or_small_tree/P2@0.5/precision/pooled | -0.056 | 0.020 | 0.012 | 0.052 | 0.028 | True |
| bush_or_small_tree/P2@0.5/precision/mean | -0.067 | 0.043 | 0.033 | 0.118 | 0.087 | False |
| bush_or_small_tree/P2@0.5/iou/pooled | -0.033 | 0.006 | 0.007 | 0.015 | 0.016 | True |
| bush_or_small_tree/P2@0.5/iou/mean | -0.043 | 0.007 | 0.008 | 0.017 | 0.023 | True |
| bush_or_small_tree/P1/recall/pooled | -0.009 | 0.003 | 0.002 | 0.007 | 0.004 | True |
| bush_or_small_tree/P1/recall/mean | -0.002 | 0.002 | 0.003 | 0.005 | 0.009 | False |
| bush_or_small_tree/P1/precision/pooled | -0.008 | 0.011 | 0.007 | 0.028 | 0.017 | False |
| bush_or_small_tree/P1/precision/mean | 0.022 | 0.020 | 0.021 | 0.053 | 0.051 | False |
| bush_or_small_tree/P1/recall_H/pooled | -0.002 | 0.003 | 0.003 | 0.008 | 0.007 | False |
| bush_or_small_tree/P1/recall_H/mean | 0.045 | 0.005 | 0.017 | 0.011 | 0.035 | True |

#### bush_or_small_tree, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| bush_or_small_tree/P2@0.8/recall_Hnear/pooled | -0.029 | 0.006 | 0.073 | 0.015 | 0.176 | False |
| bush_or_small_tree/P2@0.8/recall_Hnear/mean | -0.009 | 0.008 | 0.027 | 0.015 | 0.064 | False |
| bush_or_small_tree/P2@0.8/precision_Hnear/pooled | 0.012 | 0.094 | 0.237 | 0.261 | 0.585 | False |
| bush_or_small_tree/P2@0.8/precision_Hnear/mean | 0.023 | 0.170 | 0.231 | 0.417 | 0.563 | False |
| bush_or_small_tree/P2@0.8/iou_Hnear/pooled | -0.028 | 0.006 | 0.072 | 0.015 | 0.172 | False |
| bush_or_small_tree/P2@0.8/iou_Hnear/mean | -0.009 | 0.008 | 0.025 | 0.017 | 0.061 | False |
| bush_or_small_tree/P1/recall_Hnear/pooled | 0.002 | 0.000 | 0.006 | 0.000 | 0.015 | False |
| bush_or_small_tree/P1/recall_Hnear/mean | 0.159 | 0.000 | 0.055 | 0.000 | 0.102 | True |
| bush_or_small_tree/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/P2@0.8/recall_H/pooled | -0.001 | 0.001 | 0.003 | 0.003 | 0.008 | False |
| bush_or_small_tree/P2@0.8/recall_H/mean | -0.006 | 0.002 | 0.008 | 0.006 | 0.019 | False |
| bush_or_small_tree/P2@0.8/precision_H/pooled | 0.037 | 0.121 | 0.157 | 0.329 | 0.367 | False |
| bush_or_small_tree/P2@0.8/precision_H/mean | -0.037 | 0.132 | 0.076 | 0.310 | 0.183 | False |
| bush_or_small_tree/P2@0.8/iou_H/pooled | -0.001 | 0.001 | 0.003 | 0.003 | 0.008 | False |
| bush_or_small_tree/P2@0.8/iou_H/mean | -0.004 | 0.002 | 0.007 | 0.006 | 0.018 | False |
| bush_or_small_tree/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/P2@0.8/recall/pooled | -0.030 | 0.003 | 0.005 | 0.008 | 0.011 | True |
| bush_or_small_tree/P2@0.8/recall/mean | -0.051 | 0.007 | 0.003 | 0.018 | 0.007 | True |
| bush_or_small_tree/P2@0.8/precision/pooled | -0.074 | 0.020 | 0.011 | 0.045 | 0.025 | True |
| bush_or_small_tree/P2@0.8/precision/mean | -0.111 | 0.019 | 0.027 | 0.054 | 0.067 | True |
| bush_or_small_tree/P2@0.8/iou/pooled | -0.030 | 0.003 | 0.005 | 0.008 | 0.011 | True |
| bush_or_small_tree/P2@0.8/iou/mean | -0.041 | 0.007 | 0.002 | 0.017 | 0.005 | True |
| bush_or_small_tree/P1/recall/pooled | -0.009 | 0.003 | 0.002 | 0.007 | 0.004 | True |
| bush_or_small_tree/P1/recall/mean | -0.002 | 0.002 | 0.003 | 0.005 | 0.009 | False |
| bush_or_small_tree/P1/precision/pooled | -0.008 | 0.011 | 0.007 | 0.028 | 0.017 | False |
| bush_or_small_tree/P1/precision/mean | 0.022 | 0.020 | 0.021 | 0.053 | 0.051 | False |
| bush_or_small_tree/P1/recall_H/pooled | -0.002 | 0.003 | 0.003 | 0.008 | 0.007 | False |
| bush_or_small_tree/P1/recall_H/mean | 0.045 | 0.005 | 0.017 | 0.011 | 0.035 | True |

GT cells occupied in every repeat of one arm and none of the other: {"bush_or_small_tree": {"first_only": {"blocked": 2, "no_return": 0, "passed_through": 0, "seen": 0, "unobserved": 0}, "second_only": {"blocked": 6, "no_return": 1, "passed_through": 0, "seen": 18, "unobserved": 0}}, "obstacles": {"first_only": {"blocked": 7, "no_return": 0, "passed_through": 0, "seen": 1, "unobserved": 0}, "second_only": {"blocked": 48, "no_return": 4, "passed_through": 0, "seen": 62, "unobserved": 0}}, "rock": {"first_only": {"blocked": 0, "no_return": 0, "passed_through": 0, "seen": 0, "unobserved": 0}, "second_only": {"blocked": 0, "no_return": 0, "passed_through": 0, "seen": 0, "unobserved": 0}}, "tree_trunk": {"first_only": {"blocked": 5, "no_return": 0, "passed_through": 0, "seen": 1, "unobserved": 0}, "second_only": {"blocked": 42, "no_return": 3, "passed_through": 0, "seen": 44, "unobserved": 0}}}

### may_scene_a_radar_init_loss - may_scene_a

#### obstacles, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| obstacles/P2@0.5/recall_Hnear/pooled | 0.036 | 0.080 | 0.104 | 0.190 | 0.241 | False |
| obstacles/P2@0.5/recall_Hnear/mean | 0.276 | 0.090 | 0.046 | 0.231 | 0.129 | True |
| obstacles/P2@0.5/precision_Hnear/pooled | -0.136 | 0.129 | 0.226 | 0.308 | 0.565 | False |
| obstacles/P2@0.5/precision_Hnear/mean | -0.139 | 0.074 | 0.160 | 0.183 | 0.386 | False |
| obstacles/P2@0.5/iou_Hnear/pooled | 0.018 | 0.066 | 0.101 | 0.157 | 0.233 | False |
| obstacles/P2@0.5/iou_Hnear/mean | 0.029 | 0.040 | 0.047 | 0.098 | 0.128 | False |
| obstacles/P1/recall_Hnear/pooled | 0.004 | 0.015 | 0.005 | 0.036 | 0.015 | False |
| obstacles/P1/recall_Hnear/mean | 0.086 | 0.049 | 0.036 | 0.107 | 0.073 | False |
| obstacles/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/P2@0.5/recall_H/pooled | 0.075 | 0.002 | 0.004 | 0.005 | 0.009 | True |
| obstacles/P2@0.5/recall_H/mean | 0.114 | 0.007 | 0.004 | 0.014 | 0.010 | True |
| obstacles/P2@0.5/precision_H/pooled | 0.092 | 0.019 | 0.061 | 0.051 | 0.170 | False |
| obstacles/P2@0.5/precision_H/mean | 0.090 | 0.036 | 0.053 | 0.090 | 0.134 | False |
| obstacles/P2@0.5/iou_H/pooled | 0.070 | 0.002 | 0.004 | 0.005 | 0.009 | True |
| obstacles/P2@0.5/iou_H/mean | 0.064 | 0.006 | 0.004 | 0.016 | 0.010 | True |
| obstacles/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/P2@0.5/recall/pooled | 0.092 | 0.003 | 0.003 | 0.008 | 0.008 | True |
| obstacles/P2@0.5/recall/mean | 0.088 | 0.006 | 0.003 | 0.013 | 0.007 | True |
| obstacles/P2@0.5/precision/pooled | -0.079 | 0.006 | 0.004 | 0.014 | 0.009 | True |
| obstacles/P2@0.5/precision/mean | -0.014 | 0.011 | 0.012 | 0.026 | 0.035 | False |
| obstacles/P2@0.5/iou/pooled | 0.067 | 0.002 | 0.003 | 0.005 | 0.007 | True |
| obstacles/P2@0.5/iou/mean | 0.033 | 0.003 | 0.002 | 0.008 | 0.005 | True |
| obstacles/P1/recall/pooled | -0.001 | 0.002 | 0.001 | 0.005 | 0.003 | False |
| obstacles/P1/recall/mean | -0.003 | 0.002 | 0.003 | 0.005 | 0.008 | False |
| obstacles/P1/precision/pooled | -0.007 | 0.003 | 0.004 | 0.007 | 0.008 | False |
| obstacles/P1/precision/mean | 0.014 | 0.010 | 0.007 | 0.024 | 0.016 | False |
| obstacles/P1/recall_H/pooled | -0.002 | 0.002 | 0.002 | 0.005 | 0.004 | False |
| obstacles/P1/recall_H/mean | 0.005 | 0.007 | 0.004 | 0.017 | 0.011 | False |

#### obstacles, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| obstacles/P2@0.8/recall_Hnear/pooled | 0.007 | 0.044 | 0.069 | 0.109 | 0.168 | False |
| obstacles/P2@0.8/recall_Hnear/mean | 0.252 | 0.049 | 0.019 | 0.115 | 0.046 | True |
| obstacles/P2@0.8/precision_Hnear/pooled | -0.172 | 0.128 | 0.242 | 0.308 | 0.607 | False |
| obstacles/P2@0.8/precision_Hnear/mean | -0.131 | 0.031 | 0.228 | 0.068 | 0.551 | False |
| obstacles/P2@0.8/iou_Hnear/pooled | 0.001 | 0.039 | 0.067 | 0.097 | 0.164 | False |
| obstacles/P2@0.8/iou_Hnear/mean | 0.041 | 0.029 | 0.019 | 0.066 | 0.044 | False |
| obstacles/P1/recall_Hnear/pooled | 0.004 | 0.015 | 0.005 | 0.036 | 0.015 | False |
| obstacles/P1/recall_Hnear/mean | 0.086 | 0.049 | 0.036 | 0.107 | 0.073 | False |
| obstacles/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/P2@0.8/recall_H/pooled | 0.058 | 0.001 | 0.003 | 0.003 | 0.006 | True |
| obstacles/P2@0.8/recall_H/mean | 0.081 | 0.006 | 0.003 | 0.014 | 0.007 | True |
| obstacles/P2@0.8/precision_H/pooled | 0.064 | 0.008 | 0.099 | 0.016 | 0.246 | False |
| obstacles/P2@0.8/precision_H/mean | 0.052 | 0.025 | 0.041 | 0.061 | 0.100 | False |
| obstacles/P2@0.8/iou_H/pooled | 0.055 | 0.001 | 0.003 | 0.003 | 0.006 | True |
| obstacles/P2@0.8/iou_H/mean | 0.043 | 0.004 | 0.003 | 0.011 | 0.007 | True |
| obstacles/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/P2@0.8/recall/pooled | 0.082 | 0.001 | 0.002 | 0.003 | 0.006 | True |
| obstacles/P2@0.8/recall/mean | 0.091 | 0.003 | 0.002 | 0.007 | 0.005 | True |
| obstacles/P2@0.8/precision/pooled | -0.091 | 0.003 | 0.003 | 0.006 | 0.006 | True |
| obstacles/P2@0.8/precision/mean | -0.002 | 0.008 | 0.012 | 0.019 | 0.027 | False |
| obstacles/P2@0.8/iou/pooled | 0.063 | 0.001 | 0.002 | 0.002 | 0.005 | True |
| obstacles/P2@0.8/iou/mean | 0.036 | 0.003 | 0.002 | 0.006 | 0.004 | True |
| obstacles/P1/recall/pooled | -0.001 | 0.002 | 0.001 | 0.005 | 0.003 | False |
| obstacles/P1/recall/mean | -0.003 | 0.002 | 0.003 | 0.005 | 0.008 | False |
| obstacles/P1/precision/pooled | -0.007 | 0.003 | 0.004 | 0.007 | 0.008 | False |
| obstacles/P1/precision/mean | 0.014 | 0.010 | 0.007 | 0.024 | 0.016 | False |
| obstacles/P1/recall_H/pooled | -0.002 | 0.002 | 0.002 | 0.005 | 0.004 | False |
| obstacles/P1/recall_H/mean | 0.005 | 0.007 | 0.004 | 0.017 | 0.011 | False |

#### tree_trunk, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| tree_trunk/P2@0.5/recall_Hnear/pooled | 0.000 | 0.183 | 0.139 | 0.500 | 0.333 | False |
| tree_trunk/P2@0.5/recall_Hnear/mean | 0.075 | 0.177 | 0.112 | 0.500 | 0.250 | False |
| tree_trunk/P2@0.5/precision_Hnear/pooled | -0.400 | 0.183 | 0.447 | 0.500 | 1.000 | False |
| tree_trunk/P2@0.5/precision_Hnear/mean | -0.417 | 0.162 | 0.447 | 0.417 | 1.000 | False |
| tree_trunk/P2@0.5/iou_Hnear/pooled | -0.054 | 0.120 | 0.141 | 0.333 | 0.333 | False |
| tree_trunk/P2@0.5/iou_Hnear/mean | -0.022 | 0.085 | 0.118 | 0.225 | 0.250 | False |
| tree_trunk/P1/recall_Hnear/pooled | -0.067 | 0.139 | 0.075 | 0.333 | 0.167 | False |
| tree_trunk/P1/recall_Hnear/mean | -0.100 | 0.168 | 0.056 | 0.375 | 0.125 | False |
| tree_trunk/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/P2@0.5/recall_H/pooled | 0.061 | 0.005 | 0.002 | 0.011 | 0.004 | True |
| tree_trunk/P2@0.5/recall_H/mean | 0.102 | 0.012 | 0.009 | 0.033 | 0.022 | True |
| tree_trunk/P2@0.5/precision_H/pooled | 0.173 | 0.057 | 0.034 | 0.143 | 0.086 | True |
| tree_trunk/P2@0.5/precision_H/mean | 0.135 | 0.047 | 0.059 | 0.096 | 0.123 | True |
| tree_trunk/P2@0.5/iou_H/pooled | 0.058 | 0.004 | 0.002 | 0.010 | 0.004 | True |
| tree_trunk/P2@0.5/iou_H/mean | 0.078 | 0.010 | 0.008 | 0.027 | 0.020 | True |
| tree_trunk/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/P2@0.5/recall/pooled | 0.058 | 0.004 | 0.002 | 0.011 | 0.004 | True |
| tree_trunk/P2@0.5/recall/mean | 0.057 | 0.006 | 0.002 | 0.012 | 0.004 | True |
| tree_trunk/P2@0.5/precision/pooled | -0.038 | 0.009 | 0.005 | 0.022 | 0.011 | True |
| tree_trunk/P2@0.5/precision/mean | 0.008 | 0.011 | 0.014 | 0.027 | 0.038 | False |
| tree_trunk/P2@0.5/iou/pooled | 0.042 | 0.004 | 0.001 | 0.009 | 0.003 | True |
| tree_trunk/P2@0.5/iou/mean | 0.026 | 0.003 | 0.001 | 0.008 | 0.004 | True |
| tree_trunk/P1/recall/pooled | -0.003 | 0.003 | 0.002 | 0.008 | 0.005 | False |
| tree_trunk/P1/recall/mean | -0.007 | 0.003 | 0.002 | 0.009 | 0.006 | False |
| tree_trunk/P1/precision/pooled | 0.001 | 0.004 | 0.004 | 0.008 | 0.009 | False |
| tree_trunk/P1/precision/mean | 0.019 | 0.013 | 0.008 | 0.033 | 0.021 | False |
| tree_trunk/P1/recall_H/pooled | -0.001 | 0.002 | 0.001 | 0.005 | 0.002 | False |
| tree_trunk/P1/recall_H/mean | -0.006 | 0.011 | 0.006 | 0.027 | 0.015 | False |

#### tree_trunk, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| tree_trunk/P2@0.8/recall_Hnear/pooled | 0.067 | 0.139 | 0.091 | 0.333 | 0.167 | False |
| tree_trunk/P2@0.8/recall_Hnear/mean | 0.125 | 0.168 | 0.068 | 0.375 | 0.125 | False |
| tree_trunk/P2@0.8/precision_Hnear/pooled | -0.287 | 0.197 | 0.577 | 0.400 | 1.000 | False |
| tree_trunk/P2@0.8/precision_Hnear/mean | -0.292 | 0.191 | 0.577 | 0.375 | 1.000 | False |
| tree_trunk/P2@0.8/iou_Hnear/pooled | 0.028 | 0.095 | 0.091 | 0.222 | 0.167 | False |
| tree_trunk/P2@0.8/iou_Hnear/mean | 0.052 | 0.093 | 0.068 | 0.175 | 0.125 | False |
| tree_trunk/P1/recall_Hnear/pooled | -0.067 | 0.139 | 0.075 | 0.333 | 0.167 | False |
| tree_trunk/P1/recall_Hnear/mean | -0.100 | 0.168 | 0.056 | 0.375 | 0.125 | False |
| tree_trunk/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/P2@0.8/recall_H/pooled | 0.030 | 0.004 | 0.002 | 0.008 | 0.005 | True |
| tree_trunk/P2@0.8/recall_H/mean | 0.062 | 0.008 | 0.003 | 0.018 | 0.008 | True |
| tree_trunk/P2@0.8/precision_H/pooled | 0.094 | 0.023 | 0.086 | 0.054 | 0.230 | False |
| tree_trunk/P2@0.8/precision_H/mean | 0.082 | 0.031 | 0.049 | 0.083 | 0.114 | False |
| tree_trunk/P2@0.8/iou_H/pooled | 0.029 | 0.003 | 0.002 | 0.008 | 0.005 | True |
| tree_trunk/P2@0.8/iou_H/mean | 0.046 | 0.005 | 0.003 | 0.012 | 0.008 | True |
| tree_trunk/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/P2@0.8/recall/pooled | 0.048 | 0.002 | 0.001 | 0.006 | 0.004 | True |
| tree_trunk/P2@0.8/recall/mean | 0.061 | 0.004 | 0.001 | 0.008 | 0.003 | True |
| tree_trunk/P2@0.8/precision/pooled | -0.047 | 0.003 | 0.005 | 0.007 | 0.010 | True |
| tree_trunk/P2@0.8/precision/mean | 0.019 | 0.012 | 0.011 | 0.028 | 0.026 | False |
| tree_trunk/P2@0.8/iou/pooled | 0.035 | 0.002 | 0.001 | 0.005 | 0.002 | True |
| tree_trunk/P2@0.8/iou/mean | 0.027 | 0.003 | 0.001 | 0.007 | 0.003 | True |
| tree_trunk/P1/recall/pooled | -0.003 | 0.003 | 0.002 | 0.008 | 0.005 | False |
| tree_trunk/P1/recall/mean | -0.007 | 0.003 | 0.002 | 0.009 | 0.006 | False |
| tree_trunk/P1/precision/pooled | 0.001 | 0.004 | 0.004 | 0.008 | 0.009 | False |
| tree_trunk/P1/precision/mean | 0.019 | 0.013 | 0.008 | 0.033 | 0.021 | False |
| tree_trunk/P1/recall_H/pooled | -0.001 | 0.002 | 0.001 | 0.005 | 0.002 | False |
| tree_trunk/P1/recall_H/mean | -0.006 | 0.011 | 0.006 | 0.027 | 0.015 | False |

#### rock, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| rock/P2@0.5/recall_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.5/recall_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.5/precision_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.5/precision_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.5/iou_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.5/iou_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/P1/recall_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P1/recall_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P2@0.5/recall_H/pooled | -0.002 | 0.018 | 0.022 | 0.041 | 0.052 | False |
| rock/P2@0.5/recall_H/mean | 0.008 | 0.036 | 0.015 | 0.083 | 0.028 | False |
| rock/P2@0.5/precision_H/pooled | -0.690 | 0.236 | 0.202 | 0.500 | 0.286 | True |
| rock/P2@0.5/precision_H/mean | -0.575 | 0.183 | 0.412 | 0.333 | 0.583 | True |
| rock/P2@0.5/iou_H/pooled | -0.002 | 0.017 | 0.022 | 0.040 | 0.051 | False |
| rock/P2@0.5/iou_H/mean | 0.002 | 0.024 | 0.014 | 0.056 | 0.028 | False |
| rock/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P2@0.5/recall/pooled | 0.031 | 0.010 | 0.018 | 0.028 | 0.051 | False |
| rock/P2@0.5/recall/mean | 0.045 | 0.022 | 0.016 | 0.057 | 0.037 | True |
| rock/P2@0.5/precision/pooled | -0.027 | 0.020 | 0.011 | 0.050 | 0.031 | False |
| rock/P2@0.5/precision/mean | -0.120 | 0.056 | 0.039 | 0.131 | 0.093 | True |
| rock/P2@0.5/iou/pooled | 0.012 | 0.008 | 0.012 | 0.019 | 0.034 | False |
| rock/P2@0.5/iou/mean | 0.010 | 0.017 | 0.013 | 0.042 | 0.027 | False |
| rock/P1/recall/pooled | 0.007 | 0.016 | 0.023 | 0.039 | 0.051 | False |
| rock/P1/recall/mean | 0.010 | 0.010 | 0.023 | 0.022 | 0.054 | False |
| rock/P1/precision/pooled | 0.004 | 0.016 | 0.013 | 0.040 | 0.035 | False |
| rock/P1/precision/mean | 0.005 | 0.026 | 0.031 | 0.073 | 0.079 | False |
| rock/P1/recall_H/pooled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P1/recall_H/mean | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |

#### rock, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| rock/P2@0.8/recall_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.8/recall_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.8/precision_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.8/precision_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.8/iou_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.8/iou_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/P1/recall_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P1/recall_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P2@0.8/recall_H/pooled | 0.000 | 0.009 | 0.006 | 0.021 | 0.010 | False |
| rock/P2@0.8/recall_H/mean | 0.002 | 0.019 | 0.012 | 0.042 | 0.028 | False |
| rock/P2@0.8/precision_H/pooled | -0.417 | 0.577 | 0.354 | 1.000 | 0.500 | False |
| rock/P2@0.8/precision_H/mean | -0.417 | 0.577 | 0.354 | 1.000 | 0.500 | False |
| rock/P2@0.8/iou_H/pooled | 0.000 | 0.009 | 0.006 | 0.021 | 0.010 | False |
| rock/P2@0.8/iou_H/mean | 0.002 | 0.019 | 0.012 | 0.042 | 0.028 | False |
| rock/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P2@0.8/recall/pooled | 0.039 | 0.010 | 0.015 | 0.022 | 0.034 | True |
| rock/P2@0.8/recall/mean | 0.051 | 0.010 | 0.017 | 0.026 | 0.046 | True |
| rock/P2@0.8/precision/pooled | -0.021 | 0.011 | 0.022 | 0.028 | 0.056 | False |
| rock/P2@0.8/precision/mean | -0.066 | 0.047 | 0.039 | 0.121 | 0.108 | False |
| rock/P2@0.8/iou/pooled | 0.020 | 0.007 | 0.012 | 0.016 | 0.029 | True |
| rock/P2@0.8/iou/mean | 0.016 | 0.009 | 0.014 | 0.023 | 0.038 | False |
| rock/P1/recall/pooled | 0.007 | 0.016 | 0.023 | 0.039 | 0.051 | False |
| rock/P1/recall/mean | 0.010 | 0.010 | 0.023 | 0.022 | 0.054 | False |
| rock/P1/precision/pooled | 0.004 | 0.016 | 0.013 | 0.040 | 0.035 | False |
| rock/P1/precision/mean | 0.005 | 0.026 | 0.031 | 0.073 | 0.079 | False |
| rock/P1/recall_H/pooled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P1/recall_H/mean | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |

#### bush_or_small_tree, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| bush_or_small_tree/P2@0.5/recall_Hnear/pooled | 0.038 | 0.076 | 0.110 | 0.183 | 0.252 | False |
| bush_or_small_tree/P2@0.5/recall_Hnear/mean | 0.357 | 0.098 | 0.044 | 0.226 | 0.105 | True |
| bush_or_small_tree/P2@0.5/precision_Hnear/pooled | -0.115 | 0.130 | 0.248 | 0.313 | 0.590 | False |
| bush_or_small_tree/P2@0.5/precision_Hnear/mean | -0.076 | 0.064 | 0.190 | 0.149 | 0.414 | False |
| bush_or_small_tree/P2@0.5/iou_Hnear/pooled | 0.020 | 0.063 | 0.106 | 0.153 | 0.244 | False |
| bush_or_small_tree/P2@0.5/iou_Hnear/mean | 0.040 | 0.030 | 0.040 | 0.073 | 0.099 | False |
| bush_or_small_tree/P1/recall_Hnear/pooled | 0.008 | 0.014 | 0.006 | 0.031 | 0.015 | False |
| bush_or_small_tree/P1/recall_Hnear/mean | 0.161 | 0.003 | 0.055 | 0.007 | 0.102 | True |
| bush_or_small_tree/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/P2@0.5/recall_H/pooled | 0.090 | 0.001 | 0.007 | 0.003 | 0.016 | True |
| bush_or_small_tree/P2@0.5/recall_H/mean | 0.232 | 0.028 | 0.022 | 0.067 | 0.051 | True |
| bush_or_small_tree/P2@0.5/precision_H/pooled | 0.115 | 0.012 | 0.151 | 0.030 | 0.396 | False |
| bush_or_small_tree/P2@0.5/precision_H/mean | 0.074 | 0.033 | 0.132 | 0.082 | 0.349 | False |
| bush_or_small_tree/P2@0.5/iou_H/pooled | 0.081 | 0.001 | 0.007 | 0.002 | 0.016 | True |
| bush_or_small_tree/P2@0.5/iou_H/mean | 0.062 | 0.011 | 0.016 | 0.027 | 0.036 | True |
| bush_or_small_tree/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/P2@0.5/recall/pooled | 0.147 | 0.005 | 0.007 | 0.010 | 0.016 | True |
| bush_or_small_tree/P2@0.5/recall/mean | 0.265 | 0.006 | 0.012 | 0.013 | 0.034 | True |
| bush_or_small_tree/P2@0.5/precision/pooled | -0.156 | 0.007 | 0.012 | 0.018 | 0.028 | True |
| bush_or_small_tree/P2@0.5/precision/mean | -0.057 | 0.015 | 0.033 | 0.037 | 0.087 | False |
| bush_or_small_tree/P2@0.5/iou/pooled | 0.117 | 0.004 | 0.007 | 0.010 | 0.016 | True |
| bush_or_small_tree/P2@0.5/iou/mean | 0.083 | 0.003 | 0.008 | 0.006 | 0.023 | True |
| bush_or_small_tree/P1/recall/pooled | 0.001 | 0.001 | 0.002 | 0.003 | 0.004 | False |
| bush_or_small_tree/P1/recall/mean | 0.008 | 0.006 | 0.003 | 0.016 | 0.009 | False |
| bush_or_small_tree/P1/precision/pooled | -0.026 | 0.007 | 0.007 | 0.016 | 0.017 | True |
| bush_or_small_tree/P1/precision/mean | -0.016 | 0.021 | 0.021 | 0.050 | 0.051 | False |
| bush_or_small_tree/P1/recall_H/pooled | -0.002 | 0.003 | 0.003 | 0.008 | 0.007 | False |
| bush_or_small_tree/P1/recall_H/mean | 0.044 | 0.003 | 0.017 | 0.007 | 0.035 | True |

#### bush_or_small_tree, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| bush_or_small_tree/P2@0.8/recall_Hnear/pooled | 0.005 | 0.041 | 0.073 | 0.099 | 0.176 | False |
| bush_or_small_tree/P2@0.8/recall_Hnear/mean | 0.302 | 0.007 | 0.027 | 0.017 | 0.064 | True |
| bush_or_small_tree/P2@0.8/precision_Hnear/pooled | -0.163 | 0.138 | 0.237 | 0.339 | 0.585 | False |
| bush_or_small_tree/P2@0.8/precision_Hnear/mean | -0.099 | 0.057 | 0.231 | 0.150 | 0.563 | False |
| bush_or_small_tree/P2@0.8/iou_Hnear/pooled | -0.001 | 0.037 | 0.072 | 0.089 | 0.172 | False |
| bush_or_small_tree/P2@0.8/iou_Hnear/mean | 0.032 | 0.007 | 0.025 | 0.019 | 0.061 | False |
| bush_or_small_tree/P1/recall_Hnear/pooled | 0.008 | 0.014 | 0.006 | 0.031 | 0.015 | False |
| bush_or_small_tree/P1/recall_Hnear/mean | 0.161 | 0.003 | 0.055 | 0.007 | 0.102 | True |
| bush_or_small_tree/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/P2@0.8/recall_H/pooled | 0.083 | 0.001 | 0.003 | 0.002 | 0.008 | True |
| bush_or_small_tree/P2@0.8/recall_H/mean | 0.204 | 0.010 | 0.008 | 0.023 | 0.019 | True |
| bush_or_small_tree/P2@0.8/precision_H/pooled | 0.145 | 0.006 | 0.157 | 0.012 | 0.367 | False |
| bush_or_small_tree/P2@0.8/precision_H/mean | 0.073 | 0.037 | 0.076 | 0.101 | 0.183 | False |
| bush_or_small_tree/P2@0.8/iou_H/pooled | 0.076 | 0.001 | 0.003 | 0.002 | 0.008 | True |
| bush_or_small_tree/P2@0.8/iou_H/mean | 0.058 | 0.002 | 0.007 | 0.006 | 0.018 | True |
| bush_or_small_tree/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/P2@0.8/recall/pooled | 0.138 | 0.003 | 0.005 | 0.006 | 0.011 | True |
| bush_or_small_tree/P2@0.8/recall/mean | 0.264 | 0.004 | 0.003 | 0.007 | 0.007 | True |
| bush_or_small_tree/P2@0.8/precision/pooled | -0.176 | 0.004 | 0.011 | 0.010 | 0.025 | True |
| bush_or_small_tree/P2@0.8/precision/mean | -0.067 | 0.021 | 0.027 | 0.057 | 0.067 | True |
| bush_or_small_tree/P2@0.8/iou/pooled | 0.114 | 0.002 | 0.005 | 0.006 | 0.011 | True |
| bush_or_small_tree/P2@0.8/iou/mean | 0.093 | 0.002 | 0.002 | 0.006 | 0.005 | True |
| bush_or_small_tree/P1/recall/pooled | 0.001 | 0.001 | 0.002 | 0.003 | 0.004 | False |
| bush_or_small_tree/P1/recall/mean | 0.008 | 0.006 | 0.003 | 0.016 | 0.009 | False |
| bush_or_small_tree/P1/precision/pooled | -0.026 | 0.007 | 0.007 | 0.016 | 0.017 | True |
| bush_or_small_tree/P1/precision/mean | -0.016 | 0.021 | 0.021 | 0.050 | 0.051 | False |
| bush_or_small_tree/P1/recall_H/pooled | -0.002 | 0.003 | 0.003 | 0.008 | 0.007 | False |
| bush_or_small_tree/P1/recall_H/mean | 0.044 | 0.003 | 0.017 | 0.007 | 0.035 | True |

GT cells occupied in every repeat of one arm and none of the other: {"bush_or_small_tree": {"first_only": {"blocked": 555, "no_return": 75, "passed_through": 0, "seen": 29, "unobserved": 9}, "second_only": {"blocked": 7, "no_return": 0, "passed_through": 0, "seen": 1, "unobserved": 0}}, "obstacles": {"first_only": {"blocked": 817, "no_return": 187, "passed_through": 0, "seen": 129, "unobserved": 9}, "second_only": {"blocked": 88, "no_return": 2, "passed_through": 0, "seen": 61, "unobserved": 0}}, "rock": {"first_only": {"blocked": 4, "no_return": 0, "passed_through": 0, "seen": 0, "unobserved": 0}, "second_only": {"blocked": 0, "no_return": 0, "passed_through": 0, "seen": 0, "unobserved": 0}}, "tree_trunk": {"first_only": {"blocked": 258, "no_return": 112, "passed_through": 0, "seen": 100, "unobserved": 0}, "second_only": {"blocked": 81, "no_return": 2, "passed_through": 0, "seen": 60, "unobserved": 0}}}

### may_scene_a_radar_init_loss - may_scene_a_radar_init

#### obstacles, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| obstacles/P2@0.5/recall_Hnear/pooled | 0.095 | 0.080 | 0.004 | 0.190 | 0.007 | True |
| obstacles/P2@0.5/recall_Hnear/mean | 0.303 | 0.090 | 0.038 | 0.231 | 0.071 | True |
| obstacles/P2@0.5/precision_Hnear/pooled | -0.075 | 0.129 | 0.115 | 0.308 | 0.287 | False |
| obstacles/P2@0.5/precision_Hnear/mean | -0.088 | 0.074 | 0.133 | 0.183 | 0.352 | False |
| obstacles/P2@0.5/iou_Hnear/pooled | 0.073 | 0.066 | 0.004 | 0.157 | 0.009 | True |
| obstacles/P2@0.5/iou_Hnear/mean | 0.059 | 0.040 | 0.031 | 0.098 | 0.066 | False |
| obstacles/P1/recall_Hnear/pooled | 0.006 | 0.015 | 0.006 | 0.036 | 0.015 | False |
| obstacles/P1/recall_Hnear/mean | 0.008 | 0.049 | 0.047 | 0.107 | 0.107 | False |
| obstacles/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/P2@0.5/recall_H/pooled | 0.085 | 0.002 | 0.002 | 0.005 | 0.004 | True |
| obstacles/P2@0.5/recall_H/mean | 0.136 | 0.007 | 0.004 | 0.014 | 0.011 | True |
| obstacles/P2@0.5/precision_H/pooled | 0.111 | 0.019 | 0.124 | 0.051 | 0.324 | False |
| obstacles/P2@0.5/precision_H/mean | 0.105 | 0.036 | 0.116 | 0.090 | 0.293 | False |
| obstacles/P2@0.5/iou_H/pooled | 0.079 | 0.002 | 0.002 | 0.005 | 0.004 | True |
| obstacles/P2@0.5/iou_H/mean | 0.083 | 0.006 | 0.004 | 0.016 | 0.011 | True |
| obstacles/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/P2@0.5/recall/pooled | 0.150 | 0.003 | 0.003 | 0.008 | 0.007 | True |
| obstacles/P2@0.5/recall/mean | 0.168 | 0.006 | 0.002 | 0.013 | 0.005 | True |
| obstacles/P2@0.5/precision/pooled | -0.052 | 0.006 | 0.005 | 0.014 | 0.012 | True |
| obstacles/P2@0.5/precision/mean | 0.014 | 0.011 | 0.020 | 0.026 | 0.052 | False |
| obstacles/P2@0.5/iou/pooled | 0.121 | 0.002 | 0.003 | 0.005 | 0.007 | True |
| obstacles/P2@0.5/iou/mean | 0.101 | 0.003 | 0.002 | 0.008 | 0.006 | True |
| obstacles/P1/recall/pooled | 0.024 | 0.002 | 0.002 | 0.005 | 0.005 | True |
| obstacles/P1/recall/mean | 0.027 | 0.002 | 0.002 | 0.005 | 0.005 | True |
| obstacles/P1/precision/pooled | 0.013 | 0.003 | 0.004 | 0.007 | 0.009 | True |
| obstacles/P1/precision/mean | 0.017 | 0.010 | 0.005 | 0.024 | 0.010 | True |
| obstacles/P1/recall_H/pooled | 0.001 | 0.002 | 0.002 | 0.005 | 0.004 | False |
| obstacles/P1/recall_H/mean | 0.002 | 0.007 | 0.003 | 0.017 | 0.007 | False |

#### obstacles, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| obstacles/P2@0.8/recall_Hnear/pooled | 0.036 | 0.044 | 0.004 | 0.109 | 0.007 | False |
| obstacles/P2@0.8/recall_Hnear/mean | 0.265 | 0.049 | 0.013 | 0.115 | 0.034 | True |
| obstacles/P2@0.8/precision_Hnear/pooled | -0.114 | 0.128 | 0.109 | 0.308 | 0.270 | False |
| obstacles/P2@0.8/precision_Hnear/mean | -0.033 | 0.031 | 0.132 | 0.068 | 0.300 | False |
| obstacles/P2@0.8/iou_Hnear/pooled | 0.029 | 0.039 | 0.004 | 0.097 | 0.008 | False |
| obstacles/P2@0.8/iou_Hnear/mean | 0.056 | 0.029 | 0.009 | 0.066 | 0.024 | True |
| obstacles/P1/recall_Hnear/pooled | 0.006 | 0.015 | 0.006 | 0.036 | 0.015 | False |
| obstacles/P1/recall_Hnear/mean | 0.008 | 0.049 | 0.047 | 0.107 | 0.107 | False |
| obstacles/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/P2@0.8/recall_H/pooled | 0.063 | 0.001 | 0.001 | 0.003 | 0.003 | True |
| obstacles/P2@0.8/recall_H/mean | 0.092 | 0.006 | 0.001 | 0.014 | 0.002 | True |
| obstacles/P2@0.8/precision_H/pooled | 0.090 | 0.008 | 0.128 | 0.016 | 0.335 | False |
| obstacles/P2@0.8/precision_H/mean | 0.152 | 0.025 | 0.113 | 0.061 | 0.289 | False |
| obstacles/P2@0.8/iou_H/pooled | 0.059 | 0.001 | 0.001 | 0.003 | 0.003 | True |
| obstacles/P2@0.8/iou_H/mean | 0.053 | 0.004 | 0.001 | 0.011 | 0.002 | True |
| obstacles/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/P2@0.8/recall/pooled | 0.136 | 0.001 | 0.002 | 0.003 | 0.006 | True |
| obstacles/P2@0.8/recall/mean | 0.173 | 0.003 | 0.003 | 0.007 | 0.007 | True |
| obstacles/P2@0.8/precision/pooled | -0.058 | 0.003 | 0.002 | 0.006 | 0.007 | True |
| obstacles/P2@0.8/precision/mean | 0.029 | 0.008 | 0.012 | 0.019 | 0.027 | True |
| obstacles/P2@0.8/iou/pooled | 0.114 | 0.001 | 0.002 | 0.002 | 0.006 | True |
| obstacles/P2@0.8/iou/mean | 0.105 | 0.003 | 0.003 | 0.006 | 0.007 | True |
| obstacles/P1/recall/pooled | 0.024 | 0.002 | 0.002 | 0.005 | 0.005 | True |
| obstacles/P1/recall/mean | 0.027 | 0.002 | 0.002 | 0.005 | 0.005 | True |
| obstacles/P1/precision/pooled | 0.013 | 0.003 | 0.004 | 0.007 | 0.009 | True |
| obstacles/P1/precision/mean | 0.017 | 0.010 | 0.005 | 0.024 | 0.010 | True |
| obstacles/P1/recall_H/pooled | 0.001 | 0.002 | 0.002 | 0.005 | 0.004 | False |
| obstacles/P1/recall_H/mean | 0.002 | 0.007 | 0.003 | 0.017 | 0.007 | False |

#### tree_trunk, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| tree_trunk/P2@0.5/recall_Hnear/pooled | 0.067 | 0.183 | 0.139 | 0.500 | 0.333 | False |
| tree_trunk/P2@0.5/recall_Hnear/mean | 0.100 | 0.177 | 0.137 | 0.500 | 0.250 | False |
| tree_trunk/P2@0.5/precision_Hnear/pooled | 0.060 | 0.183 | 0.277 | 0.500 | 0.667 | False |
| tree_trunk/P2@0.5/precision_Hnear/mean | 0.033 | 0.162 | 0.250 | 0.417 | 0.500 | False |
| tree_trunk/P2@0.5/iou_Hnear/pooled | 0.040 | 0.120 | 0.117 | 0.333 | 0.286 | False |
| tree_trunk/P2@0.5/iou_Hnear/mean | 0.053 | 0.085 | 0.085 | 0.225 | 0.167 | False |
| tree_trunk/P1/recall_Hnear/pooled | 0.000 | 0.139 | 0.139 | 0.333 | 0.333 | False |
| tree_trunk/P1/recall_Hnear/mean | 0.025 | 0.168 | 0.163 | 0.375 | 0.375 | False |
| tree_trunk/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/P2@0.5/recall_H/pooled | 0.073 | 0.005 | 0.004 | 0.011 | 0.010 | True |
| tree_trunk/P2@0.5/recall_H/mean | 0.127 | 0.012 | 0.007 | 0.033 | 0.017 | True |
| tree_trunk/P2@0.5/precision_H/pooled | 0.202 | 0.057 | 0.222 | 0.143 | 0.514 | False |
| tree_trunk/P2@0.5/precision_H/mean | 0.128 | 0.047 | 0.154 | 0.096 | 0.414 | False |
| tree_trunk/P2@0.5/iou_H/pooled | 0.070 | 0.004 | 0.004 | 0.010 | 0.010 | True |
| tree_trunk/P2@0.5/iou_H/mean | 0.100 | 0.010 | 0.007 | 0.027 | 0.016 | True |
| tree_trunk/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/P2@0.5/recall/pooled | 0.131 | 0.004 | 0.001 | 0.011 | 0.003 | True |
| tree_trunk/P2@0.5/recall/mean | 0.148 | 0.006 | 0.002 | 0.012 | 0.005 | True |
| tree_trunk/P2@0.5/precision/pooled | -0.017 | 0.009 | 0.002 | 0.022 | 0.004 | True |
| tree_trunk/P2@0.5/precision/mean | 0.024 | 0.011 | 0.020 | 0.027 | 0.047 | False |
| tree_trunk/P2@0.5/iou/pooled | 0.109 | 0.004 | 0.001 | 0.009 | 0.003 | True |
| tree_trunk/P2@0.5/iou/mean | 0.104 | 0.003 | 0.003 | 0.008 | 0.006 | True |
| tree_trunk/P1/recall/pooled | 0.034 | 0.003 | 0.002 | 0.008 | 0.005 | True |
| tree_trunk/P1/recall/mean | 0.033 | 0.003 | 0.002 | 0.009 | 0.007 | True |
| tree_trunk/P1/precision/pooled | 0.024 | 0.004 | 0.006 | 0.008 | 0.015 | True |
| tree_trunk/P1/precision/mean | 0.026 | 0.013 | 0.007 | 0.033 | 0.017 | True |
| tree_trunk/P1/recall_H/pooled | 0.003 | 0.002 | 0.001 | 0.005 | 0.002 | True |
| tree_trunk/P1/recall_H/mean | 0.005 | 0.011 | 0.006 | 0.027 | 0.015 | False |

#### tree_trunk, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| tree_trunk/P2@0.8/recall_Hnear/pooled | 0.100 | 0.139 | 0.075 | 0.333 | 0.167 | False |
| tree_trunk/P2@0.8/recall_Hnear/mean | 0.150 | 0.168 | 0.056 | 0.375 | 0.125 | False |
| tree_trunk/P2@0.8/precision_Hnear/pooled | 0.113 | 0.197 | 0.224 | 0.400 | 0.500 | False |
| tree_trunk/P2@0.8/precision_Hnear/mean | 0.108 | 0.191 | 0.224 | 0.375 | 0.500 | False |
| tree_trunk/P2@0.8/iou_Hnear/pooled | 0.066 | 0.095 | 0.064 | 0.222 | 0.143 | False |
| tree_trunk/P2@0.8/iou_Hnear/mean | 0.085 | 0.093 | 0.037 | 0.175 | 0.083 | False |
| tree_trunk/P1/recall_Hnear/pooled | 0.000 | 0.139 | 0.139 | 0.333 | 0.333 | False |
| tree_trunk/P1/recall_Hnear/mean | 0.025 | 0.168 | 0.163 | 0.375 | 0.375 | False |
| tree_trunk/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/P2@0.8/recall_H/pooled | 0.038 | 0.004 | 0.003 | 0.008 | 0.007 | True |
| tree_trunk/P2@0.8/recall_H/mean | 0.076 | 0.008 | 0.002 | 0.018 | 0.005 | True |
| tree_trunk/P2@0.8/precision_H/pooled | 0.170 | 0.023 | 0.287 | 0.054 | 0.607 | False |
| tree_trunk/P2@0.8/precision_H/mean | 0.180 | 0.031 | 0.180 | 0.083 | 0.457 | False |
| tree_trunk/P2@0.8/iou_H/pooled | 0.037 | 0.003 | 0.003 | 0.008 | 0.007 | True |
| tree_trunk/P2@0.8/iou_H/mean | 0.059 | 0.005 | 0.002 | 0.012 | 0.005 | True |
| tree_trunk/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/P2@0.8/recall/pooled | 0.116 | 0.002 | 0.003 | 0.006 | 0.007 | True |
| tree_trunk/P2@0.8/recall/mean | 0.152 | 0.004 | 0.003 | 0.008 | 0.008 | True |
| tree_trunk/P2@0.8/precision/pooled | -0.023 | 0.003 | 0.003 | 0.007 | 0.007 | True |
| tree_trunk/P2@0.8/precision/mean | 0.032 | 0.012 | 0.018 | 0.028 | 0.048 | False |
| tree_trunk/P2@0.8/iou/pooled | 0.100 | 0.002 | 0.003 | 0.005 | 0.007 | True |
| tree_trunk/P2@0.8/iou/mean | 0.106 | 0.003 | 0.003 | 0.007 | 0.007 | True |
| tree_trunk/P1/recall/pooled | 0.034 | 0.003 | 0.002 | 0.008 | 0.005 | True |
| tree_trunk/P1/recall/mean | 0.033 | 0.003 | 0.002 | 0.009 | 0.007 | True |
| tree_trunk/P1/precision/pooled | 0.024 | 0.004 | 0.006 | 0.008 | 0.015 | True |
| tree_trunk/P1/precision/mean | 0.026 | 0.013 | 0.007 | 0.033 | 0.017 | True |
| tree_trunk/P1/recall_H/pooled | 0.003 | 0.002 | 0.001 | 0.005 | 0.002 | True |
| tree_trunk/P1/recall_H/mean | 0.005 | 0.011 | 0.006 | 0.027 | 0.015 | False |

#### rock, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| rock/P2@0.5/recall_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.5/recall_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.5/precision_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.5/precision_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.5/iou_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.5/iou_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/P1/recall_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P1/recall_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P2@0.5/recall_H/pooled | 0.010 | 0.018 | 0.000 | 0.041 | 0.000 | False |
| rock/P2@0.5/recall_H/mean | 0.019 | 0.036 | 0.000 | 0.083 | 0.000 | False |
| rock/P2@0.5/precision_H/pooled | n/a | 0.236 | n/a | 0.500 | n/a | None |
| rock/P2@0.5/precision_H/mean | n/a | 0.183 | n/a | 0.333 | n/a | None |
| rock/P2@0.5/iou_H/pooled | 0.010 | 0.017 | 0.000 | 0.040 | 0.000 | False |
| rock/P2@0.5/iou_H/mean | 0.013 | 0.024 | 0.000 | 0.056 | 0.000 | False |
| rock/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P2@0.5/recall/pooled | 0.079 | 0.010 | 0.021 | 0.028 | 0.045 | True |
| rock/P2@0.5/recall/mean | 0.088 | 0.022 | 0.019 | 0.057 | 0.038 | True |
| rock/P2@0.5/precision/pooled | 0.023 | 0.020 | 0.034 | 0.050 | 0.088 | False |
| rock/P2@0.5/precision/mean | -0.083 | 0.056 | 0.056 | 0.131 | 0.138 | False |
| rock/P2@0.5/iou/pooled | 0.048 | 0.008 | 0.017 | 0.019 | 0.037 | True |
| rock/P2@0.5/iou/mean | 0.038 | 0.017 | 0.011 | 0.042 | 0.023 | True |
| rock/P1/recall/pooled | 0.008 | 0.016 | 0.005 | 0.039 | 0.011 | False |
| rock/P1/recall/mean | 0.007 | 0.010 | 0.010 | 0.022 | 0.024 | False |
| rock/P1/precision/pooled | 0.009 | 0.016 | 0.014 | 0.040 | 0.030 | False |
| rock/P1/precision/mean | -0.014 | 0.026 | 0.031 | 0.073 | 0.074 | False |
| rock/P1/recall_H/pooled | -0.002 | 0.000 | 0.005 | 0.000 | 0.010 | False |
| rock/P1/recall_H/mean | -0.008 | 0.000 | 0.019 | 0.000 | 0.042 | False |

#### rock, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| rock/P2@0.8/recall_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.8/recall_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.8/precision_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.8/precision_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.8/iou_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P2@0.8/iou_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/P1/recall_Hnear/pooled | n/a | n/a | n/a | n/a | n/a | None |
| rock/P1/recall_Hnear/mean | n/a | n/a | n/a | n/a | n/a | None |
| rock/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P2@0.8/recall_H/pooled | 0.004 | 0.009 | 0.000 | 0.021 | 0.000 | False |
| rock/P2@0.8/recall_H/mean | 0.008 | 0.019 | 0.000 | 0.042 | 0.000 | False |
| rock/P2@0.8/precision_H/pooled | n/a | 0.577 | n/a | 1.000 | n/a | None |
| rock/P2@0.8/precision_H/mean | n/a | 0.577 | n/a | 1.000 | n/a | None |
| rock/P2@0.8/iou_H/pooled | 0.004 | 0.009 | 0.000 | 0.021 | 0.000 | False |
| rock/P2@0.8/iou_H/mean | 0.008 | 0.019 | 0.000 | 0.042 | 0.000 | False |
| rock/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P2@0.8/recall/pooled | 0.102 | 0.010 | 0.011 | 0.022 | 0.022 | True |
| rock/P2@0.8/recall/mean | 0.109 | 0.010 | 0.011 | 0.026 | 0.027 | True |
| rock/P2@0.8/precision/pooled | 0.049 | 0.011 | 0.025 | 0.028 | 0.065 | True |
| rock/P2@0.8/precision/mean | -0.017 | 0.047 | 0.061 | 0.121 | 0.141 | False |
| rock/P2@0.8/iou/pooled | 0.070 | 0.007 | 0.010 | 0.016 | 0.022 | True |
| rock/P2@0.8/iou/mean | 0.051 | 0.009 | 0.009 | 0.023 | 0.025 | True |
| rock/P1/recall/pooled | 0.008 | 0.016 | 0.005 | 0.039 | 0.011 | False |
| rock/P1/recall/mean | 0.007 | 0.010 | 0.010 | 0.022 | 0.024 | False |
| rock/P1/precision/pooled | 0.009 | 0.016 | 0.014 | 0.040 | 0.030 | False |
| rock/P1/precision/mean | -0.014 | 0.026 | 0.031 | 0.073 | 0.074 | False |
| rock/P1/recall_H/pooled | -0.002 | 0.000 | 0.005 | 0.000 | 0.010 | False |
| rock/P1/recall_H/mean | -0.008 | 0.000 | 0.019 | 0.000 | 0.042 | False |

#### bush_or_small_tree, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| bush_or_small_tree/P2@0.5/recall_Hnear/pooled | 0.096 | 0.076 | 0.003 | 0.183 | 0.008 | True |
| bush_or_small_tree/P2@0.5/recall_Hnear/mean | 0.384 | 0.098 | 0.006 | 0.226 | 0.013 | True |
| bush_or_small_tree/P2@0.5/precision_Hnear/pooled | -0.131 | 0.130 | 0.076 | 0.313 | 0.187 | False |
| bush_or_small_tree/P2@0.5/precision_Hnear/mean | -0.155 | 0.064 | 0.144 | 0.149 | 0.350 | False |
| bush_or_small_tree/P2@0.5/iou_Hnear/pooled | 0.074 | 0.063 | 0.003 | 0.153 | 0.007 | True |
| bush_or_small_tree/P2@0.5/iou_Hnear/mean | 0.065 | 0.030 | 0.006 | 0.073 | 0.015 | True |
| bush_or_small_tree/P1/recall_Hnear/pooled | 0.006 | 0.014 | 0.000 | 0.031 | 0.000 | False |
| bush_or_small_tree/P1/recall_Hnear/mean | 0.001 | 0.003 | 0.000 | 0.007 | 0.000 | False |
| bush_or_small_tree/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/P2@0.5/recall_H/pooled | 0.096 | 0.001 | 0.001 | 0.003 | 0.003 | True |
| bush_or_small_tree/P2@0.5/recall_H/mean | 0.250 | 0.028 | 0.002 | 0.067 | 0.006 | True |
| bush_or_small_tree/P2@0.5/precision_H/pooled | 0.135 | 0.012 | 0.123 | 0.030 | 0.248 | True |
| bush_or_small_tree/P2@0.5/precision_H/mean | 0.039 | 0.033 | 0.119 | 0.082 | 0.282 | False |
| bush_or_small_tree/P2@0.5/iou_H/pooled | 0.087 | 0.001 | 0.001 | 0.002 | 0.003 | True |
| bush_or_small_tree/P2@0.5/iou_H/mean | 0.075 | 0.011 | 0.002 | 0.027 | 0.006 | True |
| bush_or_small_tree/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/P2@0.5/recall/pooled | 0.181 | 0.005 | 0.006 | 0.010 | 0.015 | True |
| bush_or_small_tree/P2@0.5/recall/mean | 0.319 | 0.006 | 0.008 | 0.013 | 0.020 | True |
| bush_or_small_tree/P2@0.5/precision/pooled | -0.100 | 0.007 | 0.020 | 0.018 | 0.052 | True |
| bush_or_small_tree/P2@0.5/precision/mean | 0.009 | 0.015 | 0.043 | 0.037 | 0.118 | False |
| bush_or_small_tree/P2@0.5/iou/pooled | 0.150 | 0.004 | 0.006 | 0.010 | 0.015 | True |
| bush_or_small_tree/P2@0.5/iou/mean | 0.126 | 0.003 | 0.007 | 0.006 | 0.017 | True |
| bush_or_small_tree/P1/recall/pooled | 0.010 | 0.001 | 0.003 | 0.003 | 0.007 | True |
| bush_or_small_tree/P1/recall/mean | 0.010 | 0.006 | 0.002 | 0.016 | 0.005 | True |
| bush_or_small_tree/P1/precision/pooled | -0.018 | 0.007 | 0.011 | 0.016 | 0.028 | False |
| bush_or_small_tree/P1/precision/mean | -0.038 | 0.021 | 0.020 | 0.050 | 0.053 | False |
| bush_or_small_tree/P1/recall_H/pooled | -0.000 | 0.003 | 0.003 | 0.008 | 0.008 | False |
| bush_or_small_tree/P1/recall_H/mean | -0.001 | 0.003 | 0.005 | 0.007 | 0.011 | False |

#### bush_or_small_tree, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| bush_or_small_tree/P2@0.8/recall_Hnear/pooled | 0.034 | 0.041 | 0.006 | 0.099 | 0.015 | False |
| bush_or_small_tree/P2@0.8/recall_Hnear/mean | 0.311 | 0.007 | 0.008 | 0.017 | 0.015 | True |
| bush_or_small_tree/P2@0.8/precision_Hnear/pooled | -0.176 | 0.138 | 0.094 | 0.339 | 0.261 | False |
| bush_or_small_tree/P2@0.8/precision_Hnear/mean | -0.122 | 0.057 | 0.170 | 0.150 | 0.417 | False |
| bush_or_small_tree/P2@0.8/iou_Hnear/pooled | 0.027 | 0.037 | 0.006 | 0.089 | 0.015 | False |
| bush_or_small_tree/P2@0.8/iou_Hnear/mean | 0.040 | 0.007 | 0.008 | 0.019 | 0.017 | True |
| bush_or_small_tree/P1/recall_Hnear/pooled | 0.006 | 0.014 | 0.000 | 0.031 | 0.000 | False |
| bush_or_small_tree/P1/recall_Hnear/mean | 0.001 | 0.003 | 0.000 | 0.007 | 0.000 | False |
| bush_or_small_tree/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/P2@0.8/recall_H/pooled | 0.084 | 0.001 | 0.001 | 0.002 | 0.003 | True |
| bush_or_small_tree/P2@0.8/recall_H/mean | 0.209 | 0.010 | 0.002 | 0.023 | 0.006 | True |
| bush_or_small_tree/P2@0.8/precision_H/pooled | 0.108 | 0.006 | 0.121 | 0.012 | 0.329 | False |
| bush_or_small_tree/P2@0.8/precision_H/mean | 0.110 | 0.037 | 0.132 | 0.101 | 0.310 | False |
| bush_or_small_tree/P2@0.8/iou_H/pooled | 0.078 | 0.001 | 0.001 | 0.002 | 0.003 | True |
| bush_or_small_tree/P2@0.8/iou_H/mean | 0.062 | 0.002 | 0.002 | 0.006 | 0.006 | True |
| bush_or_small_tree/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/P2@0.8/recall/pooled | 0.167 | 0.003 | 0.003 | 0.006 | 0.008 | True |
| bush_or_small_tree/P2@0.8/recall/mean | 0.315 | 0.004 | 0.007 | 0.007 | 0.018 | True |
| bush_or_small_tree/P2@0.8/precision/pooled | -0.102 | 0.004 | 0.020 | 0.010 | 0.045 | True |
| bush_or_small_tree/P2@0.8/precision/mean | 0.044 | 0.021 | 0.019 | 0.057 | 0.054 | False |
| bush_or_small_tree/P2@0.8/iou/pooled | 0.144 | 0.002 | 0.003 | 0.006 | 0.008 | True |
| bush_or_small_tree/P2@0.8/iou/mean | 0.134 | 0.002 | 0.007 | 0.006 | 0.017 | True |
| bush_or_small_tree/P1/recall/pooled | 0.010 | 0.001 | 0.003 | 0.003 | 0.007 | True |
| bush_or_small_tree/P1/recall/mean | 0.010 | 0.006 | 0.002 | 0.016 | 0.005 | True |
| bush_or_small_tree/P1/precision/pooled | -0.018 | 0.007 | 0.011 | 0.016 | 0.028 | False |
| bush_or_small_tree/P1/precision/mean | -0.038 | 0.021 | 0.020 | 0.050 | 0.053 | False |
| bush_or_small_tree/P1/recall_H/pooled | -0.000 | 0.003 | 0.003 | 0.008 | 0.008 | False |
| bush_or_small_tree/P1/recall_H/mean | -0.001 | 0.003 | 0.005 | 0.007 | 0.011 | False |

GT cells occupied in every repeat of one arm and none of the other: {"bush_or_small_tree": {"first_only": {"blocked": 590, "no_return": 70, "passed_through": 0, "seen": 79, "unobserved": 9}, "second_only": {"blocked": 5, "no_return": 0, "passed_through": 0, "seen": 0, "unobserved": 0}}, "obstacles": {"first_only": {"blocked": 995, "no_return": 233, "passed_through": 0, "seen": 321, "unobserved": 9}, "second_only": {"blocked": 50, "no_return": 0, "passed_through": 0, "seen": 21, "unobserved": 0}}, "rock": {"first_only": {"blocked": 5, "no_return": 0, "passed_through": 0, "seen": 1, "unobserved": 0}, "second_only": {"blocked": 0, "no_return": 0, "passed_through": 0, "seen": 0, "unobserved": 0}}, "tree_trunk": {"first_only": {"blocked": 400, "no_return": 163, "passed_through": 0, "seen": 241, "unobserved": 0}, "second_only": {"blocked": 45, "no_return": 0, "passed_through": 0, "seen": 21, "unobserved": 0}}}

## Guard: image quality (test + guard frames)

| metric | may_scene_a | may_scene_a_radar_init | may_scene_a_radar_init_loss |
|---|---|---|---|
| image/psnr | 18.215 ± 0.050 (n=5) | 17.849 ± 0.061 (n=5) | 18.118 ± 0.042 (n=5) |
| image/ssim | 0.535 ± 0.000 (n=5) | 0.531 ± 0.001 (n=5) | 0.523 ± 0.000 (n=5) |
| image/lpips | 0.637 ± 0.001 (n=5) | 0.653 ± 0.002 (n=5) | 0.602 ± 0.001 (n=5) |

## Report only: lidar render

| metric | may_scene_a | may_scene_a_radar_init | may_scene_a_radar_init_loss |
|---|---|---|---|
| lidar_render/chamfer_distance | 1.039 ± 0.039 (n=5) | 1.808 ± 0.113 (n=5) | 0.978 ± 0.010 (n=5) |
| lidar_render/median_chamfer_distance | 1.012 ± 0.051 (n=5) | 1.724 ± 0.132 (n=5) | 1.054 ± 0.312 (n=5) |
| lidar_render/depth_median_l2 | 0.004 ± 0.000 (n=5) | 0.005 ± 0.000 (n=5) | 0.006 ± 0.000 (n=5) |
| lidar_render/depth_mean_rel_l2 | 0.277 ± 0.122 (n=5) | 0.125 ± 0.012 (n=5) | 0.137 ± 0.010 (n=5) |

