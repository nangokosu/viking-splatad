# P1/P2 over repeats: june_tall_grass

Descriptive only: a variant gap counts only when it exceeds the spread between repeats without losing precision_H. Repeats share one seed, so they are not paired. Cells are mean ± sample std (n = repeats).

- **june_scene_a_cam** (`splatad`): june_tall_grass-june_scene_a_cam-20261116-134222-17bb7a1 [17bb7a1; split identical to the reference record], june_tall_grass-june_scene_a_cam-20261116-135429-17bb7a1 [17bb7a1; split identical to the reference record], june_tall_grass-june_scene_a_cam-20261116-135432-17bb7a1 [17bb7a1; split identical to the reference record], june_tall_grass-june_scene_a_cam-20261116-135435-17bb7a1 [17bb7a1; split identical to the reference record], june_tall_grass-june_scene_a_cam-20261118-032048-c28a956 [c28a956; split identical to the reference record]
- **june_scene_a_radar_init_cam** (`splatad-radar-init`): june_tall_grass-june_scene_a_radar_init_cam-20261116-134824-17bb7a1 [17bb7a1; split identical to the reference record], june_tall_grass-june_scene_a_radar_init_cam-20261116-135427-17bb7a1 [17bb7a1; split identical to the reference record], june_tall_grass-june_scene_a_radar_init_cam-20261116-135430-17bb7a1 [17bb7a1; split identical to the reference record], june_tall_grass-june_scene_a_radar_init_cam-20261116-135433-17bb7a1 [17bb7a1; split identical to the reference record], june_tall_grass-june_scene_a_radar_init_cam-20261116-135436-17bb7a1 [17bb7a1; split identical to the reference record]
- **june_scene_a_radar_init_loss_cam** (`splatad-radar-init-lidar`): june_tall_grass-june_scene_a_radar_init_loss_cam-20261116-135425-17bb7a1 [17bb7a1; split identical to the reference record], june_tall_grass-june_scene_a_radar_init_loss_cam-20261116-135428-17bb7a1 [17bb7a1; split identical to the reference record], june_tall_grass-june_scene_a_radar_init_loss_cam-20261116-135431-17bb7a1 [17bb7a1; split identical to the reference record], june_tall_grass-june_scene_a_radar_init_loss_cam-20261116-135437-17bb7a1 [17bb7a1; split identical to the reference record], june_tall_grass-june_scene_a_radar_init_loss_cam-20261118-211540-c28a956 [c28a956; split identical to the reference record]

## obstacles, P2 occupied at >= 0.5

| metric | june_scene_a_cam | june_scene_a_radar_init_cam | june_scene_a_radar_init_loss_cam |
|---|---|---|---|
| P2@0.5/recall_Hnear/pooled | 0.107 ± 0.042 (n=5) | 0.052 ± 0.015 (n=5) | 0.146 ± 0.021 (n=5) |
| P2@0.5/recall_Hnear/mean | 0.115 ± 0.033 (n=5) | 0.078 ± 0.059 (n=5) | 0.215 ± 0.016 (n=5) |
| P2@0.5/precision_Hnear/pooled | 0.343 ± 0.107 (n=5) | 0.452 ± 0.172 (n=5) | 0.494 ± 0.053 (n=5) |
| P2@0.5/precision_Hnear/mean | 0.516 ± 0.092 (n=5) | 0.443 ± 0.146 (n=5) | 0.480 ± 0.043 (n=5) |
| P2@0.5/iou_Hnear/pooled | 0.088 ± 0.032 (n=5) | 0.049 ± 0.015 (n=5) | 0.127 ± 0.017 (n=5) |
| P2@0.5/iou_Hnear/mean | 0.089 ± 0.029 (n=5) | 0.050 ± 0.031 (n=5) | 0.113 ± 0.011 (n=5) |
| P1/recall_Hnear/pooled | 0.031 ± 0.002 (n=5) | 0.012 ± 0.005 (n=5) | 0.022 ± 0.004 (n=5) |
| P1/recall_Hnear/mean | 0.045 ± 0.010 (n=5) | 0.004 ± 0.002 (n=5) | 0.008 ± 0.006 (n=5) |
| hidden_near/gt_cells | 276.000 ± 0.000 (n=5) | 276.000 ± 0.000 (n=5) | 276.000 ± 0.000 (n=5) |
| hidden_near/objects_with_hidden_gt | 14.000 ± 0.000 (n=5) | 14.000 ± 0.000 (n=5) | 14.000 ± 0.000 (n=5) |
| P2@0.5/recall_H/pooled | 0.035 ± 0.007 (n=5) | 0.025 ± 0.003 (n=5) | 0.048 ± 0.004 (n=5) |
| P2@0.5/recall_H/mean | 0.056 ± 0.007 (n=5) | 0.047 ± 0.009 (n=5) | 0.083 ± 0.007 (n=5) |
| P2@0.5/precision_H/pooled | 0.401 ± 0.057 (n=5) | 0.472 ± 0.028 (n=5) | 0.451 ± 0.051 (n=5) |
| P2@0.5/precision_H/mean | 0.365 ± 0.023 (n=5) | 0.406 ± 0.026 (n=5) | 0.445 ± 0.029 (n=5) |
| P2@0.5/iou_H/pooled | 0.033 ± 0.007 (n=5) | 0.025 ± 0.003 (n=5) | 0.046 ± 0.004 (n=5) |
| P2@0.5/iou_H/mean | 0.040 ± 0.006 (n=5) | 0.034 ± 0.004 (n=5) | 0.055 ± 0.006 (n=5) |
| hidden/gt_cells | 2795.000 ± 0.000 (n=5) | 2795.000 ± 0.000 (n=5) | 2795.000 ± 0.000 (n=5) |
| hidden/objects_with_hidden_gt | 79.000 ± 0.000 (n=5) | 79.000 ± 0.000 (n=5) | 79.000 ± 0.000 (n=5) |
| P2@0.5/recall/pooled | 0.342 ± 0.006 (n=5) | 0.312 ± 0.002 (n=5) | 0.380 ± 0.003 (n=5) |
| P2@0.5/recall/mean | 0.395 ± 0.004 (n=5) | 0.357 ± 0.001 (n=5) | 0.435 ± 0.002 (n=5) |
| P2@0.5/precision/pooled | 0.789 ± 0.005 (n=5) | 0.784 ± 0.004 (n=5) | 0.796 ± 0.004 (n=5) |
| P2@0.5/precision/mean | 0.646 ± 0.004 (n=5) | 0.624 ± 0.015 (n=5) | 0.677 ± 0.007 (n=5) |
| P2@0.5/iou/pooled | 0.313 ± 0.005 (n=5) | 0.287 ± 0.002 (n=5) | 0.346 ± 0.003 (n=5) |
| P2@0.5/iou/mean | 0.305 ± 0.003 (n=5) | 0.279 ± 0.002 (n=5) | 0.337 ± 0.002 (n=5) |
| P1/recall/pooled | 0.322 ± 0.003 (n=5) | 0.311 ± 0.003 (n=5) | 0.329 ± 0.003 (n=5) |
| P1/recall/mean | 0.363 ± 0.004 (n=5) | 0.352 ± 0.002 (n=5) | 0.369 ± 0.002 (n=5) |
| P1/precision/pooled | 0.644 ± 0.003 (n=5) | 0.640 ± 0.003 (n=5) | 0.644 ± 0.002 (n=5) |
| P1/precision/mean | 0.561 ± 0.010 (n=5) | 0.575 ± 0.009 (n=5) | 0.575 ± 0.009 (n=5) |
| P1/recall_H/pooled | 0.007 ± 0.002 (n=5) | 0.003 ± 0.001 (n=5) | 0.008 ± 0.001 (n=5) |
| P1/recall_H/mean | 0.011 ± 0.003 (n=5) | 0.002 ± 0.001 (n=5) | 0.013 ± 0.002 (n=5) |


## obstacles, P2 occupied at >= 0.8

| metric | june_scene_a_cam | june_scene_a_radar_init_cam | june_scene_a_radar_init_loss_cam |
|---|---|---|---|
| P2@0.8/recall_Hnear/pooled | 0.064 ± 0.017 (n=5) | 0.035 ± 0.014 (n=5) | 0.041 ± 0.012 (n=5) |
| P2@0.8/recall_Hnear/mean | 0.068 ± 0.029 (n=5) | 0.046 ± 0.041 (n=5) | 0.123 ± 0.032 (n=5) |
| P2@0.8/precision_Hnear/pooled | 0.372 ± 0.152 (n=5) | 0.446 ± 0.189 (n=5) | 0.435 ± 0.095 (n=5) |
| P2@0.8/precision_Hnear/mean | 0.480 ± 0.083 (n=5) | 0.413 ± 0.131 (n=5) | 0.427 ± 0.088 (n=5) |
| P2@0.8/iou_Hnear/pooled | 0.058 ± 0.017 (n=5) | 0.033 ± 0.014 (n=5) | 0.039 ± 0.012 (n=5) |
| P2@0.8/iou_Hnear/mean | 0.054 ± 0.021 (n=5) | 0.031 ± 0.019 (n=5) | 0.072 ± 0.022 (n=5) |
| P1/recall_Hnear/pooled | 0.031 ± 0.002 (n=5) | 0.012 ± 0.005 (n=5) | 0.022 ± 0.004 (n=5) |
| P1/recall_Hnear/mean | 0.045 ± 0.010 (n=5) | 0.004 ± 0.002 (n=5) | 0.008 ± 0.006 (n=5) |
| hidden_near/gt_cells | 276.000 ± 0.000 (n=5) | 276.000 ± 0.000 (n=5) | 276.000 ± 0.000 (n=5) |
| hidden_near/objects_with_hidden_gt | 14.000 ± 0.000 (n=5) | 14.000 ± 0.000 (n=5) | 14.000 ± 0.000 (n=5) |
| P2@0.8/recall_H/pooled | 0.022 ± 0.004 (n=5) | 0.019 ± 0.003 (n=5) | 0.019 ± 0.001 (n=5) |
| P2@0.8/recall_H/mean | 0.036 ± 0.009 (n=5) | 0.034 ± 0.009 (n=5) | 0.038 ± 0.009 (n=5) |
| P2@0.8/precision_H/pooled | 0.413 ± 0.057 (n=5) | 0.493 ± 0.028 (n=5) | 0.445 ± 0.050 (n=5) |
| P2@0.8/precision_H/mean | 0.362 ± 0.048 (n=5) | 0.431 ± 0.005 (n=5) | 0.433 ± 0.038 (n=5) |
| P2@0.8/iou_H/pooled | 0.021 ± 0.004 (n=5) | 0.019 ± 0.003 (n=5) | 0.019 ± 0.001 (n=5) |
| P2@0.8/iou_H/mean | 0.027 ± 0.008 (n=5) | 0.028 ± 0.005 (n=5) | 0.025 ± 0.006 (n=5) |
| hidden/gt_cells | 2795.000 ± 0.000 (n=5) | 2795.000 ± 0.000 (n=5) | 2795.000 ± 0.000 (n=5) |
| hidden/objects_with_hidden_gt | 79.000 ± 0.000 (n=5) | 79.000 ± 0.000 (n=5) | 79.000 ± 0.000 (n=5) |
| P2@0.8/recall/pooled | 0.304 ± 0.002 (n=5) | 0.276 ± 0.002 (n=5) | 0.338 ± 0.002 (n=5) |
| P2@0.8/recall/mean | 0.358 ± 0.001 (n=5) | 0.320 ± 0.002 (n=5) | 0.398 ± 0.002 (n=5) |
| P2@0.8/precision/pooled | 0.813 ± 0.005 (n=5) | 0.803 ± 0.005 (n=5) | 0.823 ± 0.003 (n=5) |
| P2@0.8/precision/mean | 0.663 ± 0.011 (n=5) | 0.632 ± 0.016 (n=5) | 0.702 ± 0.010 (n=5) |
| P2@0.8/iou/pooled | 0.284 ± 0.002 (n=5) | 0.259 ± 0.002 (n=5) | 0.315 ± 0.002 (n=5) |
| P2@0.8/iou/mean | 0.284 ± 0.001 (n=5) | 0.258 ± 0.002 (n=5) | 0.314 ± 0.003 (n=5) |
| P1/recall/pooled | 0.322 ± 0.003 (n=5) | 0.311 ± 0.003 (n=5) | 0.329 ± 0.003 (n=5) |
| P1/recall/mean | 0.363 ± 0.004 (n=5) | 0.352 ± 0.002 (n=5) | 0.369 ± 0.002 (n=5) |
| P1/precision/pooled | 0.644 ± 0.003 (n=5) | 0.640 ± 0.003 (n=5) | 0.644 ± 0.002 (n=5) |
| P1/precision/mean | 0.561 ± 0.010 (n=5) | 0.575 ± 0.009 (n=5) | 0.575 ± 0.009 (n=5) |
| P1/recall_H/pooled | 0.007 ± 0.002 (n=5) | 0.003 ± 0.001 (n=5) | 0.008 ± 0.001 (n=5) |
| P1/recall_H/mean | 0.011 ± 0.003 (n=5) | 0.002 ± 0.001 (n=5) | 0.013 ± 0.002 (n=5) |


## tree_trunk, P2 occupied at >= 0.5

| metric | june_scene_a_cam | june_scene_a_radar_init_cam | june_scene_a_radar_init_loss_cam |
|---|---|---|---|
| P2@0.5/recall_Hnear/pooled | 0.048 ± 0.013 (n=5) | 0.044 ± 0.013 (n=5) | 0.054 ± 0.015 (n=5) |
| P2@0.5/recall_Hnear/mean | 0.077 ± 0.057 (n=5) | 0.058 ± 0.033 (n=5) | 0.069 ± 0.023 (n=5) |
| P2@0.5/precision_Hnear/pooled | 0.803 ± 0.093 (n=5) | 0.606 ± 0.133 (n=5) | 0.608 ± 0.115 (n=5) |
| P2@0.5/precision_Hnear/mean | 0.636 ± 0.172 (n=5) | 0.446 ± 0.090 (n=5) | 0.474 ± 0.076 (n=5) |
| P2@0.5/iou_Hnear/pooled | 0.047 ± 0.013 (n=5) | 0.043 ± 0.013 (n=5) | 0.052 ± 0.015 (n=5) |
| P2@0.5/iou_Hnear/mean | 0.067 ± 0.049 (n=5) | 0.044 ± 0.026 (n=5) | 0.049 ± 0.016 (n=5) |
| P1/recall_Hnear/pooled | 0.010 ± 0.007 (n=5) | 0.003 ± 0.004 (n=5) | 0.005 ± 0.007 (n=5) |
| P1/recall_Hnear/mean | 0.015 ± 0.015 (n=5) | 0.002 ± 0.003 (n=5) | 0.002 ± 0.003 (n=5) |
| hidden_near/gt_cells | 126.000 ± 0.000 (n=5) | 126.000 ± 0.000 (n=5) | 126.000 ± 0.000 (n=5) |
| hidden_near/objects_with_hidden_gt | 7.000 ± 0.000 (n=5) | 7.000 ± 0.000 (n=5) | 7.000 ± 0.000 (n=5) |
| P2@0.5/recall_H/pooled | 0.037 ± 0.004 (n=5) | 0.048 ± 0.006 (n=5) | 0.062 ± 0.005 (n=5) |
| P2@0.5/recall_H/mean | 0.046 ± 0.010 (n=5) | 0.041 ± 0.001 (n=5) | 0.070 ± 0.011 (n=5) |
| P2@0.5/precision_H/pooled | 0.487 ± 0.057 (n=5) | 0.613 ± 0.052 (n=5) | 0.599 ± 0.081 (n=5) |
| P2@0.5/precision_H/mean | 0.491 ± 0.051 (n=5) | 0.531 ± 0.021 (n=5) | 0.595 ± 0.052 (n=5) |
| P2@0.5/iou_H/pooled | 0.036 ± 0.004 (n=5) | 0.046 ± 0.006 (n=5) | 0.059 ± 0.005 (n=5) |
| P2@0.5/iou_H/mean | 0.041 ± 0.008 (n=5) | 0.036 ± 0.002 (n=5) | 0.062 ± 0.013 (n=5) |
| hidden/gt_cells | 981.000 ± 0.000 (n=5) | 981.000 ± 0.000 (n=5) | 981.000 ± 0.000 (n=5) |
| hidden/objects_with_hidden_gt | 49.000 ± 0.000 (n=5) | 49.000 ± 0.000 (n=5) | 49.000 ± 0.000 (n=5) |
| P2@0.5/recall/pooled | 0.435 ± 0.004 (n=5) | 0.417 ± 0.002 (n=5) | 0.467 ± 0.004 (n=5) |
| P2@0.5/recall/mean | 0.461 ± 0.007 (n=5) | 0.432 ± 0.003 (n=5) | 0.491 ± 0.004 (n=5) |
| P2@0.5/precision/pooled | 0.828 ± 0.003 (n=5) | 0.823 ± 0.004 (n=5) | 0.834 ± 0.004 (n=5) |
| P2@0.5/precision/mean | 0.714 ± 0.009 (n=5) | 0.703 ± 0.013 (n=5) | 0.749 ± 0.004 (n=5) |
| P2@0.5/iou/pooled | 0.399 ± 0.003 (n=5) | 0.382 ± 0.002 (n=5) | 0.427 ± 0.004 (n=5) |
| P2@0.5/iou/mean | 0.369 ± 0.005 (n=5) | 0.346 ± 0.004 (n=5) | 0.396 ± 0.004 (n=5) |
| P1/recall/pooled | 0.408 ± 0.004 (n=5) | 0.399 ± 0.003 (n=5) | 0.416 ± 0.002 (n=5) |
| P1/recall/mean | 0.424 ± 0.003 (n=5) | 0.414 ± 0.003 (n=5) | 0.431 ± 0.002 (n=5) |
| P1/precision/pooled | 0.680 ± 0.003 (n=5) | 0.678 ± 0.004 (n=5) | 0.680 ± 0.002 (n=5) |
| P1/precision/mean | 0.620 ± 0.012 (n=5) | 0.634 ± 0.011 (n=5) | 0.642 ± 0.007 (n=5) |
| P1/recall_H/pooled | 0.009 ± 0.005 (n=5) | 0.003 ± 0.001 (n=5) | 0.009 ± 0.004 (n=5) |
| P1/recall_H/mean | 0.007 ± 0.003 (n=5) | 0.002 ± 0.001 (n=5) | 0.007 ± 0.003 (n=5) |


## tree_trunk, P2 occupied at >= 0.8

| metric | june_scene_a_cam | june_scene_a_radar_init_cam | june_scene_a_radar_init_loss_cam |
|---|---|---|---|
| P2@0.8/recall_Hnear/pooled | 0.030 ± 0.013 (n=5) | 0.038 ± 0.015 (n=5) | 0.024 ± 0.013 (n=5) |
| P2@0.8/recall_Hnear/mean | 0.027 ± 0.035 (n=5) | 0.046 ± 0.022 (n=5) | 0.021 ± 0.024 (n=5) |
| P2@0.8/precision_Hnear/pooled | 0.750 ± 0.059 (n=5) | 0.630 ± 0.177 (n=5) | 0.454 ± 0.217 (n=5) |
| P2@0.8/precision_Hnear/mean | 0.550 ± 0.148 (n=5) | 0.458 ± 0.084 (n=5) | 0.407 ± 0.158 (n=5) |
| P2@0.8/iou_Hnear/pooled | 0.030 ± 0.013 (n=5) | 0.037 ± 0.015 (n=5) | 0.023 ± 0.012 (n=5) |
| P2@0.8/iou_Hnear/mean | 0.024 ± 0.031 (n=5) | 0.036 ± 0.018 (n=5) | 0.018 ± 0.021 (n=5) |
| P1/recall_Hnear/pooled | 0.010 ± 0.007 (n=5) | 0.003 ± 0.004 (n=5) | 0.005 ± 0.007 (n=5) |
| P1/recall_Hnear/mean | 0.015 ± 0.015 (n=5) | 0.002 ± 0.003 (n=5) | 0.002 ± 0.003 (n=5) |
| hidden_near/gt_cells | 126.000 ± 0.000 (n=5) | 126.000 ± 0.000 (n=5) | 126.000 ± 0.000 (n=5) |
| hidden_near/objects_with_hidden_gt | 7.000 ± 0.000 (n=5) | 7.000 ± 0.000 (n=5) | 7.000 ± 0.000 (n=5) |
| P2@0.8/recall_H/pooled | 0.026 ± 0.005 (n=5) | 0.038 ± 0.005 (n=5) | 0.028 ± 0.004 (n=5) |
| P2@0.8/recall_H/mean | 0.027 ± 0.009 (n=5) | 0.034 ± 0.003 (n=5) | 0.021 ± 0.006 (n=5) |
| P2@0.8/precision_H/pooled | 0.515 ± 0.043 (n=5) | 0.641 ± 0.042 (n=5) | 0.575 ± 0.078 (n=5) |
| P2@0.8/precision_H/mean | 0.474 ± 0.052 (n=5) | 0.580 ± 0.023 (n=5) | 0.537 ± 0.079 (n=5) |
| P2@0.8/iou_H/pooled | 0.026 ± 0.004 (n=5) | 0.037 ± 0.005 (n=5) | 0.027 ± 0.004 (n=5) |
| P2@0.8/iou_H/mean | 0.025 ± 0.009 (n=5) | 0.031 ± 0.003 (n=5) | 0.020 ± 0.006 (n=5) |
| hidden/gt_cells | 981.000 ± 0.000 (n=5) | 981.000 ± 0.000 (n=5) | 981.000 ± 0.000 (n=5) |
| hidden/objects_with_hidden_gt | 49.000 ± 0.000 (n=5) | 49.000 ± 0.000 (n=5) | 49.000 ± 0.000 (n=5) |
| P2@0.8/recall/pooled | 0.397 ± 0.003 (n=5) | 0.376 ± 0.002 (n=5) | 0.429 ± 0.002 (n=5) |
| P2@0.8/recall/mean | 0.423 ± 0.004 (n=5) | 0.394 ± 0.003 (n=5) | 0.456 ± 0.003 (n=5) |
| P2@0.8/precision/pooled | 0.845 ± 0.003 (n=5) | 0.838 ± 0.004 (n=5) | 0.847 ± 0.002 (n=5) |
| P2@0.8/precision/mean | 0.722 ± 0.009 (n=5) | 0.706 ± 0.012 (n=5) | 0.766 ± 0.007 (n=5) |
| P2@0.8/iou/pooled | 0.370 ± 0.002 (n=5) | 0.350 ± 0.003 (n=5) | 0.398 ± 0.002 (n=5) |
| P2@0.8/iou/mean | 0.346 ± 0.002 (n=5) | 0.323 ± 0.004 (n=5) | 0.372 ± 0.003 (n=5) |
| P1/recall/pooled | 0.408 ± 0.004 (n=5) | 0.399 ± 0.003 (n=5) | 0.416 ± 0.002 (n=5) |
| P1/recall/mean | 0.424 ± 0.003 (n=5) | 0.414 ± 0.003 (n=5) | 0.431 ± 0.002 (n=5) |
| P1/precision/pooled | 0.680 ± 0.003 (n=5) | 0.678 ± 0.004 (n=5) | 0.680 ± 0.002 (n=5) |
| P1/precision/mean | 0.620 ± 0.012 (n=5) | 0.634 ± 0.011 (n=5) | 0.642 ± 0.007 (n=5) |
| P1/recall_H/pooled | 0.009 ± 0.005 (n=5) | 0.003 ± 0.001 (n=5) | 0.009 ± 0.004 (n=5) |
| P1/recall_H/mean | 0.007 ± 0.003 (n=5) | 0.002 ± 0.001 (n=5) | 0.007 ± 0.003 (n=5) |


## rock, P2 occupied at >= 0.5

| metric | june_scene_a_cam | june_scene_a_radar_init_cam | june_scene_a_radar_init_loss_cam |
|---|---|---|---|
| P2@0.5/recall_Hnear/pooled | 0.050 ± 0.112 (n=5) | 0.050 ± 0.112 (n=5) | 0.300 ± 0.112 (n=5) |
| P2@0.5/recall_Hnear/mean | 0.050 ± 0.112 (n=5) | 0.050 ± 0.112 (n=5) | 0.300 ± 0.112 (n=5) |
| P2@0.5/precision_Hnear/pooled | 0.200 ± 0.447 (n=5) | 0.333 ± 0.577 (n=3) | 0.833 ± 0.236 (n=5) |
| P2@0.5/precision_Hnear/mean | 0.200 ± 0.447 (n=5) | 0.333 ± 0.577 (n=3) | 0.833 ± 0.236 (n=5) |
| P2@0.5/iou_Hnear/pooled | 0.050 ± 0.112 (n=5) | 0.050 ± 0.112 (n=5) | 0.270 ± 0.076 (n=5) |
| P2@0.5/iou_Hnear/mean | 0.050 ± 0.112 (n=5) | 0.050 ± 0.112 (n=5) | 0.270 ± 0.076 (n=5) |
| P1/recall_Hnear/pooled | 0.000 ± 0.000 (n=5) | 0.000 ± 0.000 (n=5) | 0.000 ± 0.000 (n=5) |
| P1/recall_Hnear/mean | 0.000 ± 0.000 (n=5) | 0.000 ± 0.000 (n=5) | 0.000 ± 0.000 (n=5) |
| hidden_near/gt_cells | 4.000 ± 0.000 (n=5) | 4.000 ± 0.000 (n=5) | 4.000 ± 0.000 (n=5) |
| hidden_near/objects_with_hidden_gt | 1.000 ± 0.000 (n=5) | 1.000 ± 0.000 (n=5) | 1.000 ± 0.000 (n=5) |
| P2@0.5/recall_H/pooled | 0.004 ± 0.008 (n=5) | 0.004 ± 0.008 (n=5) | 0.022 ± 0.008 (n=5) |
| P2@0.5/recall_H/mean | 0.007 ± 0.016 (n=5) | 0.007 ± 0.016 (n=5) | 0.043 ± 0.016 (n=5) |
| P2@0.5/precision_H/pooled | 0.200 ± 0.447 (n=5) | 0.167 ± 0.289 (n=3) | 0.433 ± 0.091 (n=5) |
| P2@0.5/precision_H/mean | 0.200 ± 0.447 (n=5) | 0.167 ± 0.289 (n=3) | 0.383 ± 0.112 (n=5) |
| P2@0.5/iou_H/pooled | 0.004 ± 0.008 (n=5) | 0.004 ± 0.008 (n=5) | 0.021 ± 0.008 (n=5) |
| P2@0.5/iou_H/mean | 0.007 ± 0.016 (n=5) | 0.007 ± 0.016 (n=5) | 0.039 ± 0.011 (n=5) |
| hidden/gt_cells | 55.000 ± 0.000 (n=5) | 55.000 ± 0.000 (n=5) | 55.000 ± 0.000 (n=5) |
| hidden/objects_with_hidden_gt | 7.000 ± 0.000 (n=5) | 7.000 ± 0.000 (n=5) | 7.000 ± 0.000 (n=5) |
| P2@0.5/recall/pooled | 0.198 ± 0.015 (n=5) | 0.167 ± 0.017 (n=5) | 0.266 ± 0.006 (n=5) |
| P2@0.5/recall/mean | 0.207 ± 0.013 (n=5) | 0.187 ± 0.017 (n=5) | 0.282 ± 0.007 (n=5) |
| P2@0.5/precision/pooled | 0.210 ± 0.024 (n=5) | 0.209 ± 0.017 (n=5) | 0.219 ± 0.005 (n=5) |
| P2@0.5/precision/mean | 0.243 ± 0.082 (n=5) | 0.224 ± 0.076 (n=5) | 0.236 ± 0.018 (n=5) |
| P2@0.5/iou/pooled | 0.113 ± 0.012 (n=5) | 0.102 ± 0.011 (n=5) | 0.136 ± 0.003 (n=5) |
| P2@0.5/iou/mean | 0.073 ± 0.011 (n=5) | 0.072 ± 0.016 (n=5) | 0.100 ± 0.003 (n=5) |
| P1/recall/pooled | 0.242 ± 0.018 (n=5) | 0.221 ± 0.017 (n=5) | 0.256 ± 0.019 (n=5) |
| P1/recall/mean | 0.236 ± 0.019 (n=5) | 0.212 ± 0.014 (n=5) | 0.246 ± 0.021 (n=5) |
| P1/precision/pooled | 0.206 ± 0.013 (n=5) | 0.186 ± 0.013 (n=5) | 0.201 ± 0.015 (n=5) |
| P1/precision/mean | 0.214 ± 0.015 (n=5) | 0.195 ± 0.012 (n=5) | 0.210 ± 0.017 (n=5) |
| P1/recall_H/pooled | 0.000 ± 0.000 (n=5) | 0.000 ± 0.000 (n=5) | 0.000 ± 0.000 (n=5) |
| P1/recall_H/mean | 0.000 ± 0.000 (n=5) | 0.000 ± 0.000 (n=5) | 0.000 ± 0.000 (n=5) |


## rock, P2 occupied at >= 0.8

| metric | june_scene_a_cam | june_scene_a_radar_init_cam | june_scene_a_radar_init_loss_cam |
|---|---|---|---|
| P2@0.8/recall_Hnear/pooled | 0.050 ± 0.112 (n=5) | 0.000 ± 0.000 (n=5) | 0.300 ± 0.112 (n=5) |
| P2@0.8/recall_Hnear/mean | 0.050 ± 0.112 (n=5) | 0.000 ± 0.000 (n=5) | 0.300 ± 0.112 (n=5) |
| P2@0.8/precision_Hnear/pooled | 0.333 ± 0.577 (n=3) | 0.000 (n=1) | 0.900 ± 0.224 (n=5) |
| P2@0.8/precision_Hnear/mean | 0.333 ± 0.577 (n=3) | 0.000 (n=1) | 0.900 ± 0.224 (n=5) |
| P2@0.8/iou_Hnear/pooled | 0.050 ± 0.112 (n=5) | 0.000 ± 0.000 (n=5) | 0.290 ± 0.119 (n=5) |
| P2@0.8/iou_Hnear/mean | 0.050 ± 0.112 (n=5) | 0.000 ± 0.000 (n=5) | 0.290 ± 0.119 (n=5) |
| P1/recall_Hnear/pooled | 0.000 ± 0.000 (n=5) | 0.000 ± 0.000 (n=5) | 0.000 ± 0.000 (n=5) |
| P1/recall_Hnear/mean | 0.000 ± 0.000 (n=5) | 0.000 ± 0.000 (n=5) | 0.000 ± 0.000 (n=5) |
| hidden_near/gt_cells | 4.000 ± 0.000 (n=5) | 4.000 ± 0.000 (n=5) | 4.000 ± 0.000 (n=5) |
| hidden_near/objects_with_hidden_gt | 1.000 ± 0.000 (n=5) | 1.000 ± 0.000 (n=5) | 1.000 ± 0.000 (n=5) |
| P2@0.8/recall_H/pooled | 0.004 ± 0.008 (n=5) | 0.000 ± 0.000 (n=5) | 0.022 ± 0.008 (n=5) |
| P2@0.8/recall_H/mean | 0.007 ± 0.016 (n=5) | 0.000 ± 0.000 (n=5) | 0.043 ± 0.016 (n=5) |
| P2@0.8/precision_H/pooled | 0.333 ± 0.577 (n=3) | 0.000 (n=1) | 0.733 ± 0.365 (n=5) |
| P2@0.8/precision_H/mean | 0.333 ± 0.577 (n=3) | 0.000 (n=1) | 0.717 ± 0.389 (n=5) |
| P2@0.8/iou_H/pooled | 0.004 ± 0.008 (n=5) | 0.000 ± 0.000 (n=5) | 0.022 ± 0.008 (n=5) |
| P2@0.8/iou_H/mean | 0.007 ± 0.016 (n=5) | 0.000 ± 0.000 (n=5) | 0.041 ± 0.017 (n=5) |
| hidden/gt_cells | 55.000 ± 0.000 (n=5) | 55.000 ± 0.000 (n=5) | 55.000 ± 0.000 (n=5) |
| hidden/objects_with_hidden_gt | 7.000 ± 0.000 (n=5) | 7.000 ± 0.000 (n=5) | 7.000 ± 0.000 (n=5) |
| P2@0.8/recall/pooled | 0.186 ± 0.013 (n=5) | 0.153 ± 0.019 (n=5) | 0.240 ± 0.011 (n=5) |
| P2@0.8/recall/mean | 0.197 ± 0.012 (n=5) | 0.168 ± 0.019 (n=5) | 0.257 ± 0.011 (n=5) |
| P2@0.8/precision/pooled | 0.224 ± 0.017 (n=5) | 0.227 ± 0.023 (n=5) | 0.216 ± 0.006 (n=5) |
| P2@0.8/precision/mean | 0.286 ± 0.106 (n=5) | 0.322 ± 0.137 (n=5) | 0.275 ± 0.053 (n=5) |
| P2@0.8/iou/pooled | 0.113 ± 0.009 (n=5) | 0.101 ± 0.012 (n=5) | 0.128 ± 0.005 (n=5) |
| P2@0.8/iou/mean | 0.078 ± 0.014 (n=5) | 0.074 ± 0.014 (n=5) | 0.098 ± 0.006 (n=5) |
| P1/recall/pooled | 0.242 ± 0.018 (n=5) | 0.221 ± 0.017 (n=5) | 0.256 ± 0.019 (n=5) |
| P1/recall/mean | 0.236 ± 0.019 (n=5) | 0.212 ± 0.014 (n=5) | 0.246 ± 0.021 (n=5) |
| P1/precision/pooled | 0.206 ± 0.013 (n=5) | 0.186 ± 0.013 (n=5) | 0.201 ± 0.015 (n=5) |
| P1/precision/mean | 0.214 ± 0.015 (n=5) | 0.195 ± 0.012 (n=5) | 0.210 ± 0.017 (n=5) |
| P1/recall_H/pooled | 0.000 ± 0.000 (n=5) | 0.000 ± 0.000 (n=5) | 0.000 ± 0.000 (n=5) |
| P1/recall_H/mean | 0.000 ± 0.000 (n=5) | 0.000 ± 0.000 (n=5) | 0.000 ± 0.000 (n=5) |


## bush_or_small_tree, P2 occupied at >= 0.5

| metric | june_scene_a_cam | june_scene_a_radar_init_cam | june_scene_a_radar_init_loss_cam |
|---|---|---|---|
| P2@0.5/recall_Hnear/pooled | 0.160 ± 0.068 (n=5) | 0.059 ± 0.021 (n=5) | 0.221 ± 0.031 (n=5) |
| P2@0.5/recall_Hnear/mean | 0.170 ± 0.050 (n=5) | 0.105 ± 0.098 (n=5) | 0.372 ± 0.052 (n=5) |
| P2@0.5/precision_Hnear/pooled | 0.306 ± 0.112 (n=5) | 0.406 ± 0.220 (n=5) | 0.470 ± 0.057 (n=5) |
| P2@0.5/precision_Hnear/mean | 0.467 ± 0.062 (n=5) | 0.444 ± 0.179 (n=5) | 0.420 ± 0.035 (n=5) |
| P2@0.5/iou_Hnear/pooled | 0.116 ± 0.044 (n=5) | 0.054 ± 0.021 (n=5) | 0.176 ± 0.023 (n=5) |
| P2@0.5/iou_Hnear/mean | 0.124 ± 0.038 (n=5) | 0.057 ± 0.041 (n=5) | 0.185 ± 0.041 (n=5) |
| P1/recall_Hnear/pooled | 0.051 ± 0.004 (n=5) | 0.021 ± 0.008 (n=5) | 0.037 ± 0.010 (n=5) |
| P1/recall_Hnear/mean | 0.087 ± 0.012 (n=5) | 0.006 ± 0.002 (n=5) | 0.016 ± 0.014 (n=5) |
| hidden_near/gt_cells | 146.000 ± 0.000 (n=5) | 146.000 ± 0.000 (n=5) | 146.000 ± 0.000 (n=5) |
| hidden_near/objects_with_hidden_gt | 6.000 ± 0.000 (n=5) | 6.000 ± 0.000 (n=5) | 6.000 ± 0.000 (n=5) |
| P2@0.5/recall_H/pooled | 0.034 ± 0.011 (n=5) | 0.014 ± 0.005 (n=5) | 0.042 ± 0.009 (n=5) |
| P2@0.5/recall_H/mean | 0.092 ± 0.012 (n=5) | 0.073 ± 0.032 (n=5) | 0.123 ± 0.019 (n=5) |
| P2@0.5/precision_H/pooled | 0.363 ± 0.071 (n=5) | 0.331 ± 0.062 (n=5) | 0.376 ± 0.073 (n=5) |
| P2@0.5/precision_H/mean | 0.216 ± 0.027 (n=5) | 0.262 ± 0.042 (n=5) | 0.241 ± 0.038 (n=5) |
| P2@0.5/iou_H/pooled | 0.032 ± 0.010 (n=5) | 0.013 ± 0.004 (n=5) | 0.039 ± 0.008 (n=5) |
| P2@0.5/iou_H/mean | 0.048 ± 0.004 (n=5) | 0.036 ± 0.009 (n=5) | 0.046 ± 0.008 (n=5) |
| hidden/gt_cells | 1759.000 ± 0.000 (n=5) | 1759.000 ± 0.000 (n=5) | 1759.000 ± 0.000 (n=5) |
| hidden/objects_with_hidden_gt | 23.000 ± 0.000 (n=5) | 23.000 ± 0.000 (n=5) | 23.000 ± 0.000 (n=5) |
| P2@0.5/recall/pooled | 0.144 ± 0.020 (n=5) | 0.090 ± 0.007 (n=5) | 0.196 ± 0.008 (n=5) |
| P2@0.5/recall/mean | 0.217 ± 0.010 (n=5) | 0.141 ± 0.013 (n=5) | 0.282 ± 0.009 (n=5) |
| P2@0.5/precision/pooled | 0.640 ± 0.015 (n=5) | 0.575 ± 0.020 (n=5) | 0.688 ± 0.022 (n=5) |
| P2@0.5/precision/mean | 0.436 ± 0.020 (n=5) | 0.391 ± 0.032 (n=5) | 0.491 ± 0.033 (n=5) |
| P2@0.5/iou/pooled | 0.133 ± 0.017 (n=5) | 0.084 ± 0.006 (n=5) | 0.180 ± 0.008 (n=5) |
| P2@0.5/iou/mean | 0.146 ± 0.005 (n=5) | 0.100 ± 0.006 (n=5) | 0.195 ± 0.006 (n=5) |
| P1/recall/pooled | 0.136 ± 0.006 (n=5) | 0.122 ± 0.004 (n=5) | 0.143 ± 0.004 (n=5) |
| P1/recall/mean | 0.184 ± 0.009 (n=5) | 0.170 ± 0.005 (n=5) | 0.180 ± 0.007 (n=5) |
| P1/precision/pooled | 0.508 ± 0.010 (n=5) | 0.491 ± 0.012 (n=5) | 0.516 ± 0.013 (n=5) |
| P1/precision/mean | 0.331 ± 0.011 (n=5) | 0.338 ± 0.019 (n=5) | 0.331 ± 0.014 (n=5) |
| P1/recall_H/pooled | 0.007 ± 0.002 (n=5) | 0.002 ± 0.001 (n=5) | 0.008 ± 0.002 (n=5) |
| P1/recall_H/mean | 0.024 ± 0.003 (n=5) | 0.002 ± 0.001 (n=5) | 0.030 ± 0.005 (n=5) |


## bush_or_small_tree, P2 occupied at >= 0.8

| metric | june_scene_a_cam | june_scene_a_radar_init_cam | june_scene_a_radar_init_loss_cam |
|---|---|---|---|
| P2@0.8/recall_Hnear/pooled | 0.095 ± 0.025 (n=5) | 0.033 ± 0.018 (n=5) | 0.048 ± 0.016 (n=5) |
| P2@0.8/recall_Hnear/mean | 0.117 ± 0.062 (n=5) | 0.053 ± 0.082 (n=5) | 0.213 ± 0.075 (n=5) |
| P2@0.8/precision_Hnear/pooled | 0.335 ± 0.153 (n=5) | 0.359 ± 0.226 (n=5) | 0.392 ± 0.067 (n=5) |
| P2@0.8/precision_Hnear/mean | 0.420 ± 0.111 (n=5) | 0.363 ± 0.188 (n=5) | 0.340 ± 0.102 (n=5) |
| P2@0.8/iou_Hnear/pooled | 0.079 ± 0.025 (n=5) | 0.031 ± 0.017 (n=5) | 0.044 ± 0.014 (n=5) |
| P2@0.8/iou_Hnear/mean | 0.094 ± 0.043 (n=5) | 0.028 ± 0.029 (n=5) | 0.112 ± 0.038 (n=5) |
| P1/recall_Hnear/pooled | 0.051 ± 0.004 (n=5) | 0.021 ± 0.008 (n=5) | 0.037 ± 0.010 (n=5) |
| P1/recall_Hnear/mean | 0.087 ± 0.012 (n=5) | 0.006 ± 0.002 (n=5) | 0.016 ± 0.014 (n=5) |
| hidden_near/gt_cells | 146.000 ± 0.000 (n=5) | 146.000 ± 0.000 (n=5) | 146.000 ± 0.000 (n=5) |
| hidden_near/objects_with_hidden_gt | 6.000 ± 0.000 (n=5) | 6.000 ± 0.000 (n=5) | 6.000 ± 0.000 (n=5) |
| P2@0.8/recall_H/pooled | 0.020 ± 0.005 (n=5) | 0.009 ± 0.004 (n=5) | 0.014 ± 0.002 (n=5) |
| P2@0.8/recall_H/mean | 0.063 ± 0.012 (n=5) | 0.044 ± 0.030 (n=5) | 0.071 ± 0.020 (n=5) |
| P2@0.8/precision_H/pooled | 0.365 ± 0.070 (n=5) | 0.320 ± 0.068 (n=5) | 0.356 ± 0.058 (n=5) |
| P2@0.8/precision_H/mean | 0.237 ± 0.064 (n=5) | 0.250 ± 0.018 (n=5) | 0.287 ± 0.049 (n=5) |
| P2@0.8/iou_H/pooled | 0.020 ± 0.005 (n=5) | 0.009 ± 0.004 (n=5) | 0.014 ± 0.002 (n=5) |
| P2@0.8/iou_H/mean | 0.036 ± 0.006 (n=5) | 0.028 ± 0.014 (n=5) | 0.030 ± 0.006 (n=5) |
| hidden/gt_cells | 1759.000 ± 0.000 (n=5) | 1759.000 ± 0.000 (n=5) | 1759.000 ± 0.000 (n=5) |
| hidden/objects_with_hidden_gt | 23.000 ± 0.000 (n=5) | 23.000 ± 0.000 (n=5) | 23.000 ± 0.000 (n=5) |
| P2@0.8/recall/pooled | 0.106 ± 0.008 (n=5) | 0.065 ± 0.003 (n=5) | 0.145 ± 0.005 (n=5) |
| P2@0.8/recall/mean | 0.170 ± 0.007 (n=5) | 0.101 ± 0.011 (n=5) | 0.233 ± 0.005 (n=5) |
| P2@0.8/precision/pooled | 0.671 ± 0.016 (n=5) | 0.579 ± 0.011 (n=5) | 0.760 ± 0.015 (n=5) |
| P2@0.8/precision/mean | 0.474 ± 0.038 (n=5) | 0.396 ± 0.042 (n=5) | 0.536 ± 0.026 (n=5) |
| P2@0.8/iou/pooled | 0.101 ± 0.007 (n=5) | 0.062 ± 0.003 (n=5) | 0.139 ± 0.005 (n=5) |
| P2@0.8/iou/mean | 0.125 ± 0.005 (n=5) | 0.079 ± 0.007 (n=5) | 0.171 ± 0.005 (n=5) |
| P1/recall/pooled | 0.136 ± 0.006 (n=5) | 0.122 ± 0.004 (n=5) | 0.143 ± 0.004 (n=5) |
| P1/recall/mean | 0.184 ± 0.009 (n=5) | 0.170 ± 0.005 (n=5) | 0.180 ± 0.007 (n=5) |
| P1/precision/pooled | 0.508 ± 0.010 (n=5) | 0.491 ± 0.012 (n=5) | 0.516 ± 0.013 (n=5) |
| P1/precision/mean | 0.331 ± 0.011 (n=5) | 0.338 ± 0.019 (n=5) | 0.331 ± 0.014 (n=5) |
| P1/recall_H/pooled | 0.007 ± 0.002 (n=5) | 0.002 ± 0.001 (n=5) | 0.008 ± 0.002 (n=5) |
| P1/recall_H/mean | 0.024 ± 0.003 (n=5) | 0.002 ± 0.001 (n=5) | 0.030 ± 0.005 (n=5) |


## Arm pairs (gap = first - second; separated = the two arms' [min, max] do not overlap)

### june_scene_a_radar_init_cam - june_scene_a_cam

#### obstacles, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| obstacles/P2@0.5/recall_Hnear/pooled | -0.055 | 0.015 | 0.042 | 0.033 | 0.116 | False |
| obstacles/P2@0.5/recall_Hnear/mean | -0.037 | 0.059 | 0.033 | 0.135 | 0.068 | False |
| obstacles/P2@0.5/precision_Hnear/pooled | 0.109 | 0.172 | 0.107 | 0.411 | 0.302 | False |
| obstacles/P2@0.5/precision_Hnear/mean | -0.073 | 0.146 | 0.092 | 0.366 | 0.250 | False |
| obstacles/P2@0.5/iou_Hnear/pooled | -0.039 | 0.015 | 0.032 | 0.033 | 0.085 | False |
| obstacles/P2@0.5/iou_Hnear/mean | -0.039 | 0.031 | 0.029 | 0.077 | 0.063 | False |
| obstacles/P1/recall_Hnear/pooled | -0.019 | 0.005 | 0.002 | 0.014 | 0.004 | True |
| obstacles/P1/recall_Hnear/mean | -0.041 | 0.002 | 0.010 | 0.005 | 0.025 | True |
| obstacles/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/P2@0.5/recall_H/pooled | -0.009 | 0.003 | 0.007 | 0.008 | 0.019 | False |
| obstacles/P2@0.5/recall_H/mean | -0.008 | 0.009 | 0.007 | 0.021 | 0.016 | False |
| obstacles/P2@0.5/precision_H/pooled | 0.071 | 0.028 | 0.057 | 0.075 | 0.155 | False |
| obstacles/P2@0.5/precision_H/mean | 0.042 | 0.026 | 0.023 | 0.068 | 0.051 | False |
| obstacles/P2@0.5/iou_H/pooled | -0.008 | 0.003 | 0.007 | 0.007 | 0.018 | False |
| obstacles/P2@0.5/iou_H/mean | -0.007 | 0.004 | 0.006 | 0.009 | 0.014 | False |
| obstacles/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/P2@0.5/recall/pooled | -0.030 | 0.002 | 0.006 | 0.005 | 0.015 | True |
| obstacles/P2@0.5/recall/mean | -0.038 | 0.001 | 0.004 | 0.003 | 0.010 | True |
| obstacles/P2@0.5/precision/pooled | -0.005 | 0.004 | 0.005 | 0.010 | 0.013 | False |
| obstacles/P2@0.5/precision/mean | -0.022 | 0.015 | 0.004 | 0.039 | 0.010 | False |
| obstacles/P2@0.5/iou/pooled | -0.026 | 0.002 | 0.005 | 0.005 | 0.011 | True |
| obstacles/P2@0.5/iou/mean | -0.026 | 0.002 | 0.003 | 0.006 | 0.007 | True |
| obstacles/P1/recall/pooled | -0.011 | 0.003 | 0.003 | 0.006 | 0.008 | True |
| obstacles/P1/recall/mean | -0.011 | 0.002 | 0.004 | 0.006 | 0.011 | True |
| obstacles/P1/precision/pooled | -0.003 | 0.003 | 0.003 | 0.008 | 0.008 | False |
| obstacles/P1/precision/mean | 0.014 | 0.009 | 0.010 | 0.019 | 0.028 | False |
| obstacles/P1/recall_H/pooled | -0.005 | 0.001 | 0.002 | 0.002 | 0.006 | True |
| obstacles/P1/recall_H/mean | -0.009 | 0.001 | 0.003 | 0.001 | 0.006 | True |

#### obstacles, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| obstacles/P2@0.8/recall_Hnear/pooled | -0.030 | 0.014 | 0.017 | 0.036 | 0.043 | False |
| obstacles/P2@0.8/recall_Hnear/mean | -0.022 | 0.041 | 0.029 | 0.110 | 0.064 | False |
| obstacles/P2@0.8/precision_Hnear/pooled | 0.073 | 0.189 | 0.152 | 0.506 | 0.370 | False |
| obstacles/P2@0.8/precision_Hnear/mean | -0.067 | 0.131 | 0.083 | 0.348 | 0.194 | False |
| obstacles/P2@0.8/iou_Hnear/pooled | -0.025 | 0.014 | 0.017 | 0.035 | 0.044 | False |
| obstacles/P2@0.8/iou_Hnear/mean | -0.023 | 0.019 | 0.021 | 0.052 | 0.054 | False |
| obstacles/P1/recall_Hnear/pooled | -0.019 | 0.005 | 0.002 | 0.014 | 0.004 | True |
| obstacles/P1/recall_Hnear/mean | -0.041 | 0.002 | 0.010 | 0.005 | 0.025 | True |
| obstacles/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/P2@0.8/recall_H/pooled | -0.003 | 0.003 | 0.004 | 0.006 | 0.011 | False |
| obstacles/P2@0.8/recall_H/mean | -0.002 | 0.009 | 0.009 | 0.025 | 0.023 | False |
| obstacles/P2@0.8/precision_H/pooled | 0.080 | 0.028 | 0.057 | 0.065 | 0.147 | False |
| obstacles/P2@0.8/precision_H/mean | 0.069 | 0.005 | 0.048 | 0.014 | 0.131 | True |
| obstacles/P2@0.8/iou_H/pooled | -0.003 | 0.003 | 0.004 | 0.006 | 0.011 | False |
| obstacles/P2@0.8/iou_H/mean | 0.001 | 0.005 | 0.008 | 0.013 | 0.022 | False |
| obstacles/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/P2@0.8/recall/pooled | -0.028 | 0.002 | 0.002 | 0.005 | 0.005 | True |
| obstacles/P2@0.8/recall/mean | -0.037 | 0.002 | 0.001 | 0.005 | 0.003 | True |
| obstacles/P2@0.8/precision/pooled | -0.010 | 0.005 | 0.005 | 0.009 | 0.015 | False |
| obstacles/P2@0.8/precision/mean | -0.031 | 0.016 | 0.011 | 0.042 | 0.030 | False |
| obstacles/P2@0.8/iou/pooled | -0.025 | 0.002 | 0.002 | 0.005 | 0.004 | True |
| obstacles/P2@0.8/iou/mean | -0.026 | 0.002 | 0.001 | 0.006 | 0.003 | True |
| obstacles/P1/recall/pooled | -0.011 | 0.003 | 0.003 | 0.006 | 0.008 | True |
| obstacles/P1/recall/mean | -0.011 | 0.002 | 0.004 | 0.006 | 0.011 | True |
| obstacles/P1/precision/pooled | -0.003 | 0.003 | 0.003 | 0.008 | 0.008 | False |
| obstacles/P1/precision/mean | 0.014 | 0.009 | 0.010 | 0.019 | 0.028 | False |
| obstacles/P1/recall_H/pooled | -0.005 | 0.001 | 0.002 | 0.002 | 0.006 | True |
| obstacles/P1/recall_H/mean | -0.009 | 0.001 | 0.003 | 0.001 | 0.006 | True |

#### tree_trunk, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| tree_trunk/P2@0.5/recall_Hnear/pooled | -0.003 | 0.013 | 0.013 | 0.032 | 0.032 | False |
| tree_trunk/P2@0.5/recall_Hnear/mean | -0.019 | 0.033 | 0.057 | 0.093 | 0.136 | False |
| tree_trunk/P2@0.5/precision_Hnear/pooled | -0.197 | 0.133 | 0.093 | 0.339 | 0.222 | False |
| tree_trunk/P2@0.5/precision_Hnear/mean | -0.190 | 0.090 | 0.172 | 0.236 | 0.456 | False |
| tree_trunk/P2@0.5/iou_Hnear/pooled | -0.004 | 0.013 | 0.013 | 0.031 | 0.032 | False |
| tree_trunk/P2@0.5/iou_Hnear/mean | -0.023 | 0.026 | 0.049 | 0.072 | 0.118 | False |
| tree_trunk/P1/recall_Hnear/pooled | -0.006 | 0.004 | 0.007 | 0.008 | 0.016 | False |
| tree_trunk/P1/recall_Hnear/mean | -0.013 | 0.003 | 0.015 | 0.005 | 0.036 | False |
| tree_trunk/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/P2@0.5/recall_H/pooled | 0.010 | 0.006 | 0.004 | 0.015 | 0.009 | False |
| tree_trunk/P2@0.5/recall_H/mean | -0.005 | 0.001 | 0.010 | 0.004 | 0.024 | False |
| tree_trunk/P2@0.5/precision_H/pooled | 0.125 | 0.052 | 0.057 | 0.128 | 0.140 | False |
| tree_trunk/P2@0.5/precision_H/mean | 0.039 | 0.021 | 0.051 | 0.056 | 0.128 | False |
| tree_trunk/P2@0.5/iou_H/pooled | 0.010 | 0.006 | 0.004 | 0.014 | 0.009 | False |
| tree_trunk/P2@0.5/iou_H/mean | -0.005 | 0.002 | 0.008 | 0.006 | 0.022 | False |
| tree_trunk/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/P2@0.5/recall/pooled | -0.019 | 0.002 | 0.004 | 0.005 | 0.009 | True |
| tree_trunk/P2@0.5/recall/mean | -0.029 | 0.003 | 0.007 | 0.007 | 0.019 | True |
| tree_trunk/P2@0.5/precision/pooled | -0.006 | 0.004 | 0.003 | 0.011 | 0.008 | False |
| tree_trunk/P2@0.5/precision/mean | -0.010 | 0.013 | 0.009 | 0.036 | 0.023 | False |
| tree_trunk/P2@0.5/iou/pooled | -0.017 | 0.002 | 0.003 | 0.004 | 0.007 | True |
| tree_trunk/P2@0.5/iou/mean | -0.023 | 0.004 | 0.005 | 0.011 | 0.012 | True |
| tree_trunk/P1/recall/pooled | -0.009 | 0.003 | 0.004 | 0.007 | 0.010 | False |
| tree_trunk/P1/recall/mean | -0.009 | 0.003 | 0.003 | 0.006 | 0.009 | True |
| tree_trunk/P1/precision/pooled | -0.002 | 0.004 | 0.003 | 0.011 | 0.008 | False |
| tree_trunk/P1/precision/mean | 0.014 | 0.011 | 0.012 | 0.028 | 0.032 | False |
| tree_trunk/P1/recall_H/pooled | -0.006 | 0.001 | 0.005 | 0.002 | 0.010 | False |
| tree_trunk/P1/recall_H/mean | -0.005 | 0.001 | 0.003 | 0.002 | 0.008 | False |

#### tree_trunk, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| tree_trunk/P2@0.8/recall_Hnear/pooled | 0.008 | 0.015 | 0.013 | 0.040 | 0.032 | False |
| tree_trunk/P2@0.8/recall_Hnear/mean | 0.018 | 0.022 | 0.035 | 0.053 | 0.083 | False |
| tree_trunk/P2@0.8/precision_Hnear/pooled | -0.120 | 0.177 | 0.059 | 0.467 | 0.167 | False |
| tree_trunk/P2@0.8/precision_Hnear/mean | -0.092 | 0.084 | 0.148 | 0.229 | 0.406 | False |
| tree_trunk/P2@0.8/iou_Hnear/pooled | 0.007 | 0.015 | 0.013 | 0.038 | 0.031 | False |
| tree_trunk/P2@0.8/iou_Hnear/mean | 0.012 | 0.018 | 0.031 | 0.046 | 0.073 | False |
| tree_trunk/P1/recall_Hnear/pooled | -0.006 | 0.004 | 0.007 | 0.008 | 0.016 | False |
| tree_trunk/P1/recall_Hnear/mean | -0.013 | 0.003 | 0.015 | 0.005 | 0.036 | False |
| tree_trunk/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/P2@0.8/recall_H/pooled | 0.012 | 0.005 | 0.005 | 0.011 | 0.009 | False |
| tree_trunk/P2@0.8/recall_H/mean | 0.007 | 0.003 | 0.009 | 0.008 | 0.024 | False |
| tree_trunk/P2@0.8/precision_H/pooled | 0.126 | 0.042 | 0.043 | 0.102 | 0.103 | True |
| tree_trunk/P2@0.8/precision_H/mean | 0.106 | 0.023 | 0.052 | 0.056 | 0.126 | False |
| tree_trunk/P2@0.8/iou_H/pooled | 0.012 | 0.005 | 0.004 | 0.011 | 0.009 | True |
| tree_trunk/P2@0.8/iou_H/mean | 0.006 | 0.003 | 0.009 | 0.006 | 0.023 | False |
| tree_trunk/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/P2@0.8/recall/pooled | -0.022 | 0.002 | 0.003 | 0.006 | 0.007 | True |
| tree_trunk/P2@0.8/recall/mean | -0.029 | 0.003 | 0.004 | 0.006 | 0.010 | True |
| tree_trunk/P2@0.8/precision/pooled | -0.008 | 0.004 | 0.003 | 0.010 | 0.007 | True |
| tree_trunk/P2@0.8/precision/mean | -0.016 | 0.012 | 0.009 | 0.029 | 0.023 | False |
| tree_trunk/P2@0.8/iou/pooled | -0.020 | 0.003 | 0.002 | 0.006 | 0.006 | True |
| tree_trunk/P2@0.8/iou/mean | -0.023 | 0.004 | 0.002 | 0.009 | 0.006 | True |
| tree_trunk/P1/recall/pooled | -0.009 | 0.003 | 0.004 | 0.007 | 0.010 | False |
| tree_trunk/P1/recall/mean | -0.009 | 0.003 | 0.003 | 0.006 | 0.009 | True |
| tree_trunk/P1/precision/pooled | -0.002 | 0.004 | 0.003 | 0.011 | 0.008 | False |
| tree_trunk/P1/precision/mean | 0.014 | 0.011 | 0.012 | 0.028 | 0.032 | False |
| tree_trunk/P1/recall_H/pooled | -0.006 | 0.001 | 0.005 | 0.002 | 0.010 | False |
| tree_trunk/P1/recall_H/mean | -0.005 | 0.001 | 0.003 | 0.002 | 0.008 | False |

#### rock, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| rock/P2@0.5/recall_Hnear/pooled | 0.000 | 0.112 | 0.112 | 0.250 | 0.250 | False |
| rock/P2@0.5/recall_Hnear/mean | 0.000 | 0.112 | 0.112 | 0.250 | 0.250 | False |
| rock/P2@0.5/precision_Hnear/pooled | 0.133 | 0.577 | 0.447 | 1.000 | 1.000 | False |
| rock/P2@0.5/precision_Hnear/mean | 0.133 | 0.577 | 0.447 | 1.000 | 1.000 | False |
| rock/P2@0.5/iou_Hnear/pooled | 0.000 | 0.112 | 0.112 | 0.250 | 0.250 | False |
| rock/P2@0.5/iou_Hnear/mean | 0.000 | 0.112 | 0.112 | 0.250 | 0.250 | False |
| rock/P1/recall_Hnear/pooled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P1/recall_Hnear/mean | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P2@0.5/recall_H/pooled | 0.000 | 0.008 | 0.008 | 0.018 | 0.018 | False |
| rock/P2@0.5/recall_H/mean | 0.000 | 0.016 | 0.016 | 0.036 | 0.036 | False |
| rock/P2@0.5/precision_H/pooled | -0.033 | 0.289 | 0.447 | 0.500 | 1.000 | False |
| rock/P2@0.5/precision_H/mean | -0.033 | 0.289 | 0.447 | 0.500 | 1.000 | False |
| rock/P2@0.5/iou_H/pooled | -0.000 | 0.008 | 0.008 | 0.018 | 0.018 | False |
| rock/P2@0.5/iou_H/mean | 0.000 | 0.016 | 0.016 | 0.036 | 0.036 | False |
| rock/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P2@0.5/recall/pooled | -0.031 | 0.017 | 0.015 | 0.047 | 0.035 | False |
| rock/P2@0.5/recall/mean | -0.021 | 0.017 | 0.013 | 0.042 | 0.030 | False |
| rock/P2@0.5/precision/pooled | -0.001 | 0.017 | 0.024 | 0.047 | 0.055 | False |
| rock/P2@0.5/precision/mean | -0.018 | 0.076 | 0.082 | 0.198 | 0.173 | False |
| rock/P2@0.5/iou/pooled | -0.011 | 0.011 | 0.012 | 0.029 | 0.028 | False |
| rock/P2@0.5/iou/mean | -0.002 | 0.016 | 0.011 | 0.035 | 0.028 | False |
| rock/P1/recall/pooled | -0.021 | 0.017 | 0.018 | 0.047 | 0.047 | False |
| rock/P1/recall/mean | -0.023 | 0.014 | 0.019 | 0.038 | 0.050 | False |
| rock/P1/precision/pooled | -0.019 | 0.013 | 0.013 | 0.029 | 0.027 | False |
| rock/P1/precision/mean | -0.018 | 0.012 | 0.015 | 0.029 | 0.032 | False |
| rock/P1/recall_H/pooled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P1/recall_H/mean | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |

#### rock, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| rock/P2@0.8/recall_Hnear/pooled | -0.050 | 0.000 | 0.112 | 0.000 | 0.250 | False |
| rock/P2@0.8/recall_Hnear/mean | -0.050 | 0.000 | 0.112 | 0.000 | 0.250 | False |
| rock/P2@0.8/precision_Hnear/pooled | -0.333 | n/a | 0.577 | 0.000 | 1.000 | None |
| rock/P2@0.8/precision_Hnear/mean | -0.333 | n/a | 0.577 | 0.000 | 1.000 | None |
| rock/P2@0.8/iou_Hnear/pooled | -0.050 | 0.000 | 0.112 | 0.000 | 0.250 | False |
| rock/P2@0.8/iou_Hnear/mean | -0.050 | 0.000 | 0.112 | 0.000 | 0.250 | False |
| rock/P1/recall_Hnear/pooled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P1/recall_Hnear/mean | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P2@0.8/recall_H/pooled | -0.004 | 0.000 | 0.008 | 0.000 | 0.018 | False |
| rock/P2@0.8/recall_H/mean | -0.007 | 0.000 | 0.016 | 0.000 | 0.036 | False |
| rock/P2@0.8/precision_H/pooled | -0.333 | n/a | 0.577 | 0.000 | 1.000 | None |
| rock/P2@0.8/precision_H/mean | -0.333 | n/a | 0.577 | 0.000 | 1.000 | None |
| rock/P2@0.8/iou_H/pooled | -0.004 | 0.000 | 0.008 | 0.000 | 0.018 | False |
| rock/P2@0.8/iou_H/mean | -0.007 | 0.000 | 0.016 | 0.000 | 0.036 | False |
| rock/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P2@0.8/recall/pooled | -0.033 | 0.019 | 0.013 | 0.047 | 0.024 | False |
| rock/P2@0.8/recall/mean | -0.029 | 0.019 | 0.012 | 0.042 | 0.033 | False |
| rock/P2@0.8/precision/pooled | 0.003 | 0.023 | 0.017 | 0.063 | 0.037 | False |
| rock/P2@0.8/precision/mean | 0.036 | 0.137 | 0.106 | 0.317 | 0.261 | False |
| rock/P2@0.8/iou/pooled | -0.012 | 0.012 | 0.009 | 0.033 | 0.018 | False |
| rock/P2@0.8/iou/mean | -0.004 | 0.014 | 0.014 | 0.033 | 0.035 | False |
| rock/P1/recall/pooled | -0.021 | 0.017 | 0.018 | 0.047 | 0.047 | False |
| rock/P1/recall/mean | -0.023 | 0.014 | 0.019 | 0.038 | 0.050 | False |
| rock/P1/precision/pooled | -0.019 | 0.013 | 0.013 | 0.029 | 0.027 | False |
| rock/P1/precision/mean | -0.018 | 0.012 | 0.015 | 0.029 | 0.032 | False |
| rock/P1/recall_H/pooled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P1/recall_H/mean | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |

#### bush_or_small_tree, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| bush_or_small_tree/P2@0.5/recall_Hnear/pooled | -0.101 | 0.021 | 0.068 | 0.048 | 0.192 | False |
| bush_or_small_tree/P2@0.5/recall_Hnear/mean | -0.064 | 0.098 | 0.050 | 0.206 | 0.125 | False |
| bush_or_small_tree/P2@0.5/precision_Hnear/pooled | 0.099 | 0.220 | 0.112 | 0.541 | 0.314 | False |
| bush_or_small_tree/P2@0.5/precision_Hnear/mean | -0.023 | 0.179 | 0.062 | 0.483 | 0.136 | False |
| bush_or_small_tree/P2@0.5/iou_Hnear/pooled | -0.062 | 0.021 | 0.044 | 0.046 | 0.114 | False |
| bush_or_small_tree/P2@0.5/iou_Hnear/mean | -0.067 | 0.041 | 0.038 | 0.095 | 0.104 | False |
| bush_or_small_tree/P1/recall_Hnear/pooled | -0.030 | 0.008 | 0.004 | 0.021 | 0.007 | True |
| bush_or_small_tree/P1/recall_Hnear/mean | -0.081 | 0.002 | 0.012 | 0.006 | 0.028 | True |
| bush_or_small_tree/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/P2@0.5/recall_H/pooled | -0.020 | 0.005 | 0.011 | 0.013 | 0.029 | True |
| bush_or_small_tree/P2@0.5/recall_H/mean | -0.018 | 0.032 | 0.012 | 0.080 | 0.032 | False |
| bush_or_small_tree/P2@0.5/precision_H/pooled | -0.032 | 0.062 | 0.071 | 0.159 | 0.192 | False |
| bush_or_small_tree/P2@0.5/precision_H/mean | 0.046 | 0.042 | 0.027 | 0.112 | 0.063 | False |
| bush_or_small_tree/P2@0.5/iou_H/pooled | -0.019 | 0.004 | 0.010 | 0.012 | 0.026 | True |
| bush_or_small_tree/P2@0.5/iou_H/mean | -0.011 | 0.009 | 0.004 | 0.026 | 0.012 | False |
| bush_or_small_tree/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/P2@0.5/recall/pooled | -0.054 | 0.007 | 0.020 | 0.015 | 0.048 | True |
| bush_or_small_tree/P2@0.5/recall/mean | -0.075 | 0.013 | 0.010 | 0.029 | 0.024 | True |
| bush_or_small_tree/P2@0.5/precision/pooled | -0.064 | 0.020 | 0.015 | 0.046 | 0.032 | True |
| bush_or_small_tree/P2@0.5/precision/mean | -0.045 | 0.032 | 0.020 | 0.073 | 0.052 | False |
| bush_or_small_tree/P2@0.5/iou/pooled | -0.048 | 0.006 | 0.017 | 0.014 | 0.041 | True |
| bush_or_small_tree/P2@0.5/iou/mean | -0.046 | 0.006 | 0.005 | 0.014 | 0.014 | True |
| bush_or_small_tree/P1/recall/pooled | -0.015 | 0.004 | 0.006 | 0.011 | 0.015 | True |
| bush_or_small_tree/P1/recall/mean | -0.014 | 0.005 | 0.009 | 0.014 | 0.025 | False |
| bush_or_small_tree/P1/precision/pooled | -0.016 | 0.012 | 0.010 | 0.029 | 0.023 | False |
| bush_or_small_tree/P1/precision/mean | 0.007 | 0.019 | 0.011 | 0.043 | 0.028 | False |
| bush_or_small_tree/P1/recall_H/pooled | -0.005 | 0.001 | 0.002 | 0.003 | 0.005 | True |
| bush_or_small_tree/P1/recall_H/mean | -0.022 | 0.001 | 0.003 | 0.002 | 0.009 | True |

#### bush_or_small_tree, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| bush_or_small_tree/P2@0.8/recall_Hnear/pooled | -0.062 | 0.018 | 0.025 | 0.041 | 0.068 | False |
| bush_or_small_tree/P2@0.8/recall_Hnear/mean | -0.064 | 0.082 | 0.062 | 0.194 | 0.149 | False |
| bush_or_small_tree/P2@0.8/precision_Hnear/pooled | 0.024 | 0.226 | 0.153 | 0.557 | 0.389 | False |
| bush_or_small_tree/P2@0.8/precision_Hnear/mean | -0.057 | 0.188 | 0.111 | 0.467 | 0.295 | False |
| bush_or_small_tree/P2@0.8/iou_Hnear/pooled | -0.048 | 0.017 | 0.025 | 0.039 | 0.069 | False |
| bush_or_small_tree/P2@0.8/iou_Hnear/mean | -0.066 | 0.029 | 0.043 | 0.069 | 0.097 | False |
| bush_or_small_tree/P1/recall_Hnear/pooled | -0.030 | 0.008 | 0.004 | 0.021 | 0.007 | True |
| bush_or_small_tree/P1/recall_Hnear/mean | -0.081 | 0.002 | 0.012 | 0.006 | 0.028 | True |
| bush_or_small_tree/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/P2@0.8/recall_H/pooled | -0.011 | 0.004 | 0.005 | 0.010 | 0.013 | True |
| bush_or_small_tree/P2@0.8/recall_H/mean | -0.019 | 0.030 | 0.012 | 0.072 | 0.032 | False |
| bush_or_small_tree/P2@0.8/precision_H/pooled | -0.045 | 0.068 | 0.070 | 0.160 | 0.177 | False |
| bush_or_small_tree/P2@0.8/precision_H/mean | 0.014 | 0.018 | 0.064 | 0.037 | 0.141 | False |
| bush_or_small_tree/P2@0.8/iou_H/pooled | -0.010 | 0.004 | 0.005 | 0.009 | 0.012 | True |
| bush_or_small_tree/P2@0.8/iou_H/mean | -0.008 | 0.014 | 0.006 | 0.030 | 0.016 | False |
| bush_or_small_tree/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/P2@0.8/recall/pooled | -0.041 | 0.003 | 0.008 | 0.008 | 0.020 | True |
| bush_or_small_tree/P2@0.8/recall/mean | -0.069 | 0.011 | 0.007 | 0.027 | 0.018 | True |
| bush_or_small_tree/P2@0.8/precision/pooled | -0.092 | 0.011 | 0.016 | 0.030 | 0.042 | True |
| bush_or_small_tree/P2@0.8/precision/mean | -0.077 | 0.042 | 0.038 | 0.101 | 0.089 | False |
| bush_or_small_tree/P2@0.8/iou/pooled | -0.039 | 0.003 | 0.007 | 0.008 | 0.018 | True |
| bush_or_small_tree/P2@0.8/iou/mean | -0.045 | 0.007 | 0.005 | 0.016 | 0.013 | True |
| bush_or_small_tree/P1/recall/pooled | -0.015 | 0.004 | 0.006 | 0.011 | 0.015 | True |
| bush_or_small_tree/P1/recall/mean | -0.014 | 0.005 | 0.009 | 0.014 | 0.025 | False |
| bush_or_small_tree/P1/precision/pooled | -0.016 | 0.012 | 0.010 | 0.029 | 0.023 | False |
| bush_or_small_tree/P1/precision/mean | 0.007 | 0.019 | 0.011 | 0.043 | 0.028 | False |
| bush_or_small_tree/P1/recall_H/pooled | -0.005 | 0.001 | 0.002 | 0.003 | 0.005 | True |
| bush_or_small_tree/P1/recall_H/mean | -0.022 | 0.001 | 0.003 | 0.002 | 0.009 | True |

GT cells occupied in every repeat of one arm and none of the other: {"bush_or_small_tree": {"first_only": {"blocked": 5, "no_return": 0, "passed_through": 0, "seen": 0, "unobserved": 0}, "second_only": {"blocked": 9, "no_return": 0, "passed_through": 0, "seen": 9, "unobserved": 0}}, "obstacles": {"first_only": {"blocked": 17, "no_return": 0, "passed_through": 0, "seen": 0, "unobserved": 0}, "second_only": {"blocked": 26, "no_return": 1, "passed_through": 0, "seen": 19, "unobserved": 0}}, "rock": {"first_only": {"blocked": 0, "no_return": 0, "passed_through": 0, "seen": 0, "unobserved": 0}, "second_only": {"blocked": 0, "no_return": 0, "passed_through": 0, "seen": 0, "unobserved": 0}}, "tree_trunk": {"first_only": {"blocked": 12, "no_return": 0, "passed_through": 0, "seen": 0, "unobserved": 0}, "second_only": {"blocked": 17, "no_return": 1, "passed_through": 0, "seen": 10, "unobserved": 0}}}

### june_scene_a_radar_init_loss_cam - june_scene_a_cam

#### obstacles, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| obstacles/P2@0.5/recall_Hnear/pooled | 0.038 | 0.021 | 0.042 | 0.051 | 0.116 | False |
| obstacles/P2@0.5/recall_Hnear/mean | 0.101 | 0.016 | 0.033 | 0.036 | 0.068 | True |
| obstacles/P2@0.5/precision_Hnear/pooled | 0.151 | 0.053 | 0.107 | 0.134 | 0.302 | False |
| obstacles/P2@0.5/precision_Hnear/mean | -0.036 | 0.043 | 0.092 | 0.101 | 0.250 | False |
| obstacles/P2@0.5/iou_Hnear/pooled | 0.038 | 0.017 | 0.032 | 0.039 | 0.085 | False |
| obstacles/P2@0.5/iou_Hnear/mean | 0.024 | 0.011 | 0.029 | 0.029 | 0.063 | False |
| obstacles/P1/recall_Hnear/pooled | -0.009 | 0.004 | 0.002 | 0.011 | 0.004 | False |
| obstacles/P1/recall_Hnear/mean | -0.037 | 0.006 | 0.010 | 0.014 | 0.025 | True |
| obstacles/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/P2@0.5/recall_H/pooled | 0.014 | 0.004 | 0.007 | 0.012 | 0.019 | False |
| obstacles/P2@0.5/recall_H/mean | 0.027 | 0.007 | 0.007 | 0.016 | 0.016 | True |
| obstacles/P2@0.5/precision_H/pooled | 0.050 | 0.051 | 0.057 | 0.137 | 0.155 | False |
| obstacles/P2@0.5/precision_H/mean | 0.081 | 0.029 | 0.023 | 0.072 | 0.051 | True |
| obstacles/P2@0.5/iou_H/pooled | 0.013 | 0.004 | 0.007 | 0.011 | 0.018 | False |
| obstacles/P2@0.5/iou_H/mean | 0.015 | 0.006 | 0.006 | 0.015 | 0.014 | False |
| obstacles/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/P2@0.5/recall/pooled | 0.038 | 0.003 | 0.006 | 0.007 | 0.015 | True |
| obstacles/P2@0.5/recall/mean | 0.040 | 0.002 | 0.004 | 0.005 | 0.010 | True |
| obstacles/P2@0.5/precision/pooled | 0.007 | 0.004 | 0.005 | 0.011 | 0.013 | False |
| obstacles/P2@0.5/precision/mean | 0.031 | 0.007 | 0.004 | 0.020 | 0.010 | True |
| obstacles/P2@0.5/iou/pooled | 0.033 | 0.003 | 0.005 | 0.008 | 0.011 | True |
| obstacles/P2@0.5/iou/mean | 0.031 | 0.002 | 0.003 | 0.006 | 0.007 | True |
| obstacles/P1/recall/pooled | 0.007 | 0.003 | 0.003 | 0.006 | 0.008 | False |
| obstacles/P1/recall/mean | 0.006 | 0.002 | 0.004 | 0.006 | 0.011 | False |
| obstacles/P1/precision/pooled | 0.000 | 0.002 | 0.003 | 0.005 | 0.008 | False |
| obstacles/P1/precision/mean | 0.014 | 0.009 | 0.010 | 0.021 | 0.028 | False |
| obstacles/P1/recall_H/pooled | 0.001 | 0.001 | 0.002 | 0.004 | 0.006 | False |
| obstacles/P1/recall_H/mean | 0.002 | 0.002 | 0.003 | 0.005 | 0.006 | False |

#### obstacles, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| obstacles/P2@0.8/recall_Hnear/pooled | -0.024 | 0.012 | 0.017 | 0.033 | 0.043 | False |
| obstacles/P2@0.8/recall_Hnear/mean | 0.056 | 0.032 | 0.029 | 0.074 | 0.064 | False |
| obstacles/P2@0.8/precision_Hnear/pooled | 0.063 | 0.095 | 0.152 | 0.248 | 0.370 | False |
| obstacles/P2@0.8/precision_Hnear/mean | -0.053 | 0.088 | 0.083 | 0.197 | 0.194 | False |
| obstacles/P2@0.8/iou_Hnear/pooled | -0.020 | 0.012 | 0.017 | 0.031 | 0.044 | False |
| obstacles/P2@0.8/iou_Hnear/mean | 0.018 | 0.022 | 0.021 | 0.050 | 0.054 | False |
| obstacles/P1/recall_Hnear/pooled | -0.009 | 0.004 | 0.002 | 0.011 | 0.004 | False |
| obstacles/P1/recall_Hnear/mean | -0.037 | 0.006 | 0.010 | 0.014 | 0.025 | True |
| obstacles/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/P2@0.8/recall_H/pooled | -0.003 | 0.001 | 0.004 | 0.001 | 0.011 | False |
| obstacles/P2@0.8/recall_H/mean | 0.002 | 0.009 | 0.009 | 0.022 | 0.023 | False |
| obstacles/P2@0.8/precision_H/pooled | 0.031 | 0.050 | 0.057 | 0.135 | 0.147 | False |
| obstacles/P2@0.8/precision_H/mean | 0.071 | 0.038 | 0.048 | 0.102 | 0.131 | False |
| obstacles/P2@0.8/iou_H/pooled | -0.003 | 0.001 | 0.004 | 0.002 | 0.011 | False |
| obstacles/P2@0.8/iou_H/mean | -0.002 | 0.006 | 0.008 | 0.013 | 0.022 | False |
| obstacles/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/P2@0.8/recall/pooled | 0.034 | 0.002 | 0.002 | 0.006 | 0.005 | True |
| obstacles/P2@0.8/recall/mean | 0.041 | 0.002 | 0.001 | 0.006 | 0.003 | True |
| obstacles/P2@0.8/precision/pooled | 0.010 | 0.003 | 0.005 | 0.006 | 0.015 | False |
| obstacles/P2@0.8/precision/mean | 0.039 | 0.010 | 0.011 | 0.027 | 0.030 | True |
| obstacles/P2@0.8/iou/pooled | 0.031 | 0.002 | 0.002 | 0.006 | 0.004 | True |
| obstacles/P2@0.8/iou/mean | 0.029 | 0.003 | 0.001 | 0.007 | 0.003 | True |
| obstacles/P1/recall/pooled | 0.007 | 0.003 | 0.003 | 0.006 | 0.008 | False |
| obstacles/P1/recall/mean | 0.006 | 0.002 | 0.004 | 0.006 | 0.011 | False |
| obstacles/P1/precision/pooled | 0.000 | 0.002 | 0.003 | 0.005 | 0.008 | False |
| obstacles/P1/precision/mean | 0.014 | 0.009 | 0.010 | 0.021 | 0.028 | False |
| obstacles/P1/recall_H/pooled | 0.001 | 0.001 | 0.002 | 0.004 | 0.006 | False |
| obstacles/P1/recall_H/mean | 0.002 | 0.002 | 0.003 | 0.005 | 0.006 | False |

#### tree_trunk, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| tree_trunk/P2@0.5/recall_Hnear/pooled | 0.006 | 0.015 | 0.013 | 0.040 | 0.032 | False |
| tree_trunk/P2@0.5/recall_Hnear/mean | -0.007 | 0.023 | 0.057 | 0.055 | 0.136 | False |
| tree_trunk/P2@0.5/precision_Hnear/pooled | -0.195 | 0.115 | 0.093 | 0.308 | 0.222 | False |
| tree_trunk/P2@0.5/precision_Hnear/mean | -0.161 | 0.076 | 0.172 | 0.181 | 0.456 | False |
| tree_trunk/P2@0.5/iou_Hnear/pooled | 0.005 | 0.015 | 0.013 | 0.039 | 0.032 | False |
| tree_trunk/P2@0.5/iou_Hnear/mean | -0.018 | 0.016 | 0.049 | 0.041 | 0.118 | False |
| tree_trunk/P1/recall_Hnear/pooled | -0.005 | 0.007 | 0.007 | 0.016 | 0.016 | False |
| tree_trunk/P1/recall_Hnear/mean | -0.012 | 0.003 | 0.015 | 0.007 | 0.036 | False |
| tree_trunk/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/P2@0.5/recall_H/pooled | 0.025 | 0.005 | 0.004 | 0.014 | 0.009 | True |
| tree_trunk/P2@0.5/recall_H/mean | 0.024 | 0.011 | 0.010 | 0.027 | 0.024 | False |
| tree_trunk/P2@0.5/precision_H/pooled | 0.112 | 0.081 | 0.057 | 0.217 | 0.140 | False |
| tree_trunk/P2@0.5/precision_H/mean | 0.104 | 0.052 | 0.051 | 0.115 | 0.128 | False |
| tree_trunk/P2@0.5/iou_H/pooled | 0.024 | 0.005 | 0.004 | 0.014 | 0.009 | True |
| tree_trunk/P2@0.5/iou_H/mean | 0.021 | 0.013 | 0.008 | 0.031 | 0.022 | False |
| tree_trunk/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/P2@0.5/recall/pooled | 0.031 | 0.004 | 0.004 | 0.010 | 0.009 | True |
| tree_trunk/P2@0.5/recall/mean | 0.030 | 0.004 | 0.007 | 0.009 | 0.019 | True |
| tree_trunk/P2@0.5/precision/pooled | 0.005 | 0.004 | 0.003 | 0.010 | 0.008 | False |
| tree_trunk/P2@0.5/precision/mean | 0.036 | 0.004 | 0.009 | 0.009 | 0.023 | True |
| tree_trunk/P2@0.5/iou/pooled | 0.028 | 0.004 | 0.003 | 0.009 | 0.007 | True |
| tree_trunk/P2@0.5/iou/mean | 0.027 | 0.004 | 0.005 | 0.011 | 0.012 | True |
| tree_trunk/P1/recall/pooled | 0.007 | 0.002 | 0.004 | 0.007 | 0.010 | True |
| tree_trunk/P1/recall/mean | 0.008 | 0.002 | 0.003 | 0.004 | 0.009 | True |
| tree_trunk/P1/precision/pooled | -0.001 | 0.002 | 0.003 | 0.004 | 0.008 | False |
| tree_trunk/P1/precision/mean | 0.022 | 0.007 | 0.012 | 0.018 | 0.032 | False |
| tree_trunk/P1/recall_H/pooled | -0.000 | 0.004 | 0.005 | 0.008 | 0.010 | False |
| tree_trunk/P1/recall_H/mean | -0.000 | 0.003 | 0.003 | 0.006 | 0.008 | False |

#### tree_trunk, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| tree_trunk/P2@0.8/recall_Hnear/pooled | -0.006 | 0.013 | 0.013 | 0.032 | 0.032 | False |
| tree_trunk/P2@0.8/recall_Hnear/mean | -0.006 | 0.024 | 0.035 | 0.060 | 0.083 | False |
| tree_trunk/P2@0.8/precision_Hnear/pooled | -0.296 | 0.217 | 0.059 | 0.524 | 0.167 | False |
| tree_trunk/P2@0.8/precision_Hnear/mean | -0.143 | 0.158 | 0.148 | 0.433 | 0.406 | False |
| tree_trunk/P2@0.8/iou_Hnear/pooled | -0.007 | 0.012 | 0.013 | 0.031 | 0.031 | False |
| tree_trunk/P2@0.8/iou_Hnear/mean | -0.006 | 0.021 | 0.031 | 0.052 | 0.073 | False |
| tree_trunk/P1/recall_Hnear/pooled | -0.005 | 0.007 | 0.007 | 0.016 | 0.016 | False |
| tree_trunk/P1/recall_Hnear/mean | -0.012 | 0.003 | 0.015 | 0.007 | 0.036 | False |
| tree_trunk/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/P2@0.8/recall_H/pooled | 0.001 | 0.004 | 0.005 | 0.010 | 0.009 | False |
| tree_trunk/P2@0.8/recall_H/mean | -0.006 | 0.006 | 0.009 | 0.016 | 0.024 | False |
| tree_trunk/P2@0.8/precision_H/pooled | 0.060 | 0.078 | 0.043 | 0.190 | 0.103 | False |
| tree_trunk/P2@0.8/precision_H/mean | 0.064 | 0.079 | 0.052 | 0.157 | 0.126 | False |
| tree_trunk/P2@0.8/iou_H/pooled | 0.002 | 0.004 | 0.004 | 0.010 | 0.009 | False |
| tree_trunk/P2@0.8/iou_H/mean | -0.005 | 0.006 | 0.009 | 0.016 | 0.023 | False |
| tree_trunk/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/P2@0.8/recall/pooled | 0.032 | 0.002 | 0.003 | 0.005 | 0.007 | True |
| tree_trunk/P2@0.8/recall/mean | 0.033 | 0.003 | 0.004 | 0.008 | 0.010 | True |
| tree_trunk/P2@0.8/precision/pooled | 0.002 | 0.002 | 0.003 | 0.006 | 0.007 | False |
| tree_trunk/P2@0.8/precision/mean | 0.044 | 0.007 | 0.009 | 0.014 | 0.023 | True |
| tree_trunk/P2@0.8/iou/pooled | 0.028 | 0.002 | 0.002 | 0.006 | 0.006 | True |
| tree_trunk/P2@0.8/iou/mean | 0.026 | 0.003 | 0.002 | 0.009 | 0.006 | True |
| tree_trunk/P1/recall/pooled | 0.007 | 0.002 | 0.004 | 0.007 | 0.010 | True |
| tree_trunk/P1/recall/mean | 0.008 | 0.002 | 0.003 | 0.004 | 0.009 | True |
| tree_trunk/P1/precision/pooled | -0.001 | 0.002 | 0.003 | 0.004 | 0.008 | False |
| tree_trunk/P1/precision/mean | 0.022 | 0.007 | 0.012 | 0.018 | 0.032 | False |
| tree_trunk/P1/recall_H/pooled | -0.000 | 0.004 | 0.005 | 0.008 | 0.010 | False |
| tree_trunk/P1/recall_H/mean | -0.000 | 0.003 | 0.003 | 0.006 | 0.008 | False |

#### rock, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| rock/P2@0.5/recall_Hnear/pooled | 0.250 | 0.112 | 0.112 | 0.250 | 0.250 | False |
| rock/P2@0.5/recall_Hnear/mean | 0.250 | 0.112 | 0.112 | 0.250 | 0.250 | False |
| rock/P2@0.5/precision_Hnear/pooled | 0.633 | 0.236 | 0.447 | 0.500 | 1.000 | False |
| rock/P2@0.5/precision_Hnear/mean | 0.633 | 0.236 | 0.447 | 0.500 | 1.000 | False |
| rock/P2@0.5/iou_Hnear/pooled | 0.220 | 0.076 | 0.112 | 0.200 | 0.250 | False |
| rock/P2@0.5/iou_Hnear/mean | 0.220 | 0.076 | 0.112 | 0.200 | 0.250 | False |
| rock/P1/recall_Hnear/pooled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P1/recall_Hnear/mean | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P2@0.5/recall_H/pooled | 0.018 | 0.008 | 0.008 | 0.018 | 0.018 | False |
| rock/P2@0.5/recall_H/mean | 0.036 | 0.016 | 0.016 | 0.036 | 0.036 | False |
| rock/P2@0.5/precision_H/pooled | 0.233 | 0.091 | 0.447 | 0.167 | 1.000 | False |
| rock/P2@0.5/precision_H/mean | 0.183 | 0.112 | 0.447 | 0.250 | 1.000 | False |
| rock/P2@0.5/iou_H/pooled | 0.018 | 0.008 | 0.008 | 0.018 | 0.018 | False |
| rock/P2@0.5/iou_H/mean | 0.031 | 0.011 | 0.016 | 0.029 | 0.036 | False |
| rock/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P2@0.5/recall/pooled | 0.068 | 0.006 | 0.015 | 0.012 | 0.035 | True |
| rock/P2@0.5/recall/mean | 0.074 | 0.007 | 0.013 | 0.017 | 0.030 | True |
| rock/P2@0.5/precision/pooled | 0.009 | 0.005 | 0.024 | 0.014 | 0.055 | False |
| rock/P2@0.5/precision/mean | -0.006 | 0.018 | 0.082 | 0.045 | 0.173 | False |
| rock/P2@0.5/iou/pooled | 0.023 | 0.003 | 0.012 | 0.009 | 0.028 | True |
| rock/P2@0.5/iou/mean | 0.027 | 0.003 | 0.011 | 0.009 | 0.028 | True |
| rock/P1/recall/pooled | 0.014 | 0.019 | 0.018 | 0.047 | 0.047 | False |
| rock/P1/recall/mean | 0.010 | 0.021 | 0.019 | 0.054 | 0.050 | False |
| rock/P1/precision/pooled | -0.004 | 0.015 | 0.013 | 0.036 | 0.027 | False |
| rock/P1/precision/mean | -0.004 | 0.017 | 0.015 | 0.041 | 0.032 | False |
| rock/P1/recall_H/pooled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P1/recall_H/mean | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |

#### rock, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| rock/P2@0.8/recall_Hnear/pooled | 0.250 | 0.112 | 0.112 | 0.250 | 0.250 | False |
| rock/P2@0.8/recall_Hnear/mean | 0.250 | 0.112 | 0.112 | 0.250 | 0.250 | False |
| rock/P2@0.8/precision_Hnear/pooled | 0.567 | 0.224 | 0.577 | 0.500 | 1.000 | False |
| rock/P2@0.8/precision_Hnear/mean | 0.567 | 0.224 | 0.577 | 0.500 | 1.000 | False |
| rock/P2@0.8/iou_Hnear/pooled | 0.240 | 0.119 | 0.112 | 0.300 | 0.250 | False |
| rock/P2@0.8/iou_Hnear/mean | 0.240 | 0.119 | 0.112 | 0.300 | 0.250 | False |
| rock/P1/recall_Hnear/pooled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P1/recall_Hnear/mean | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P2@0.8/recall_H/pooled | 0.018 | 0.008 | 0.008 | 0.018 | 0.018 | False |
| rock/P2@0.8/recall_H/mean | 0.036 | 0.016 | 0.016 | 0.036 | 0.036 | False |
| rock/P2@0.8/precision_H/pooled | 0.400 | 0.365 | 0.577 | 0.667 | 1.000 | False |
| rock/P2@0.8/precision_H/mean | 0.383 | 0.389 | 0.577 | 0.750 | 1.000 | False |
| rock/P2@0.8/iou_H/pooled | 0.018 | 0.008 | 0.008 | 0.019 | 0.018 | False |
| rock/P2@0.8/iou_H/mean | 0.034 | 0.017 | 0.016 | 0.043 | 0.036 | False |
| rock/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P2@0.8/recall/pooled | 0.054 | 0.011 | 0.013 | 0.024 | 0.024 | True |
| rock/P2@0.8/recall/mean | 0.059 | 0.011 | 0.012 | 0.025 | 0.033 | True |
| rock/P2@0.8/precision/pooled | -0.008 | 0.006 | 0.017 | 0.015 | 0.037 | False |
| rock/P2@0.8/precision/mean | -0.011 | 0.053 | 0.106 | 0.108 | 0.261 | False |
| rock/P2@0.8/iou/pooled | 0.015 | 0.005 | 0.009 | 0.012 | 0.018 | False |
| rock/P2@0.8/iou/mean | 0.020 | 0.006 | 0.014 | 0.015 | 0.035 | False |
| rock/P1/recall/pooled | 0.014 | 0.019 | 0.018 | 0.047 | 0.047 | False |
| rock/P1/recall/mean | 0.010 | 0.021 | 0.019 | 0.054 | 0.050 | False |
| rock/P1/precision/pooled | -0.004 | 0.015 | 0.013 | 0.036 | 0.027 | False |
| rock/P1/precision/mean | -0.004 | 0.017 | 0.015 | 0.041 | 0.032 | False |
| rock/P1/recall_H/pooled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P1/recall_H/mean | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |

#### bush_or_small_tree, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| bush_or_small_tree/P2@0.5/recall_Hnear/pooled | 0.060 | 0.031 | 0.068 | 0.068 | 0.192 | False |
| bush_or_small_tree/P2@0.5/recall_Hnear/mean | 0.202 | 0.052 | 0.050 | 0.116 | 0.125 | True |
| bush_or_small_tree/P2@0.5/precision_Hnear/pooled | 0.164 | 0.057 | 0.112 | 0.129 | 0.314 | False |
| bush_or_small_tree/P2@0.5/precision_Hnear/mean | -0.047 | 0.035 | 0.062 | 0.076 | 0.136 | False |
| bush_or_small_tree/P2@0.5/iou_Hnear/pooled | 0.060 | 0.023 | 0.044 | 0.053 | 0.114 | False |
| bush_or_small_tree/P2@0.5/iou_Hnear/mean | 0.061 | 0.041 | 0.038 | 0.098 | 0.104 | False |
| bush_or_small_tree/P1/recall_Hnear/pooled | -0.014 | 0.010 | 0.004 | 0.027 | 0.007 | False |
| bush_or_small_tree/P1/recall_Hnear/mean | -0.071 | 0.014 | 0.012 | 0.034 | 0.028 | True |
| bush_or_small_tree/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/P2@0.5/recall_H/pooled | 0.008 | 0.009 | 0.011 | 0.021 | 0.029 | False |
| bush_or_small_tree/P2@0.5/recall_H/mean | 0.031 | 0.019 | 0.012 | 0.051 | 0.032 | False |
| bush_or_small_tree/P2@0.5/precision_H/pooled | 0.013 | 0.073 | 0.071 | 0.202 | 0.192 | False |
| bush_or_small_tree/P2@0.5/precision_H/mean | 0.025 | 0.038 | 0.027 | 0.095 | 0.063 | False |
| bush_or_small_tree/P2@0.5/iou_H/pooled | 0.007 | 0.008 | 0.010 | 0.020 | 0.026 | False |
| bush_or_small_tree/P2@0.5/iou_H/mean | -0.001 | 0.008 | 0.004 | 0.020 | 0.012 | False |
| bush_or_small_tree/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/P2@0.5/recall/pooled | 0.052 | 0.008 | 0.020 | 0.021 | 0.048 | True |
| bush_or_small_tree/P2@0.5/recall/mean | 0.066 | 0.009 | 0.010 | 0.021 | 0.024 | True |
| bush_or_small_tree/P2@0.5/precision/pooled | 0.048 | 0.022 | 0.015 | 0.059 | 0.032 | True |
| bush_or_small_tree/P2@0.5/precision/mean | 0.055 | 0.033 | 0.020 | 0.073 | 0.052 | False |
| bush_or_small_tree/P2@0.5/iou/pooled | 0.047 | 0.008 | 0.017 | 0.021 | 0.041 | True |
| bush_or_small_tree/P2@0.5/iou/mean | 0.048 | 0.006 | 0.005 | 0.013 | 0.014 | True |
| bush_or_small_tree/P1/recall/pooled | 0.007 | 0.004 | 0.006 | 0.011 | 0.015 | False |
| bush_or_small_tree/P1/recall/mean | -0.004 | 0.007 | 0.009 | 0.015 | 0.025 | False |
| bush_or_small_tree/P1/precision/pooled | 0.008 | 0.013 | 0.010 | 0.031 | 0.023 | False |
| bush_or_small_tree/P1/precision/mean | 0.000 | 0.014 | 0.011 | 0.036 | 0.028 | False |
| bush_or_small_tree/P1/recall_H/pooled | 0.002 | 0.002 | 0.002 | 0.006 | 0.005 | False |
| bush_or_small_tree/P1/recall_H/mean | 0.007 | 0.005 | 0.003 | 0.014 | 0.009 | False |

#### bush_or_small_tree, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| bush_or_small_tree/P2@0.8/recall_Hnear/pooled | -0.047 | 0.016 | 0.025 | 0.041 | 0.068 | False |
| bush_or_small_tree/P2@0.8/recall_Hnear/mean | 0.096 | 0.075 | 0.062 | 0.171 | 0.149 | False |
| bush_or_small_tree/P2@0.8/precision_Hnear/pooled | 0.057 | 0.067 | 0.153 | 0.181 | 0.389 | False |
| bush_or_small_tree/P2@0.8/precision_Hnear/mean | -0.080 | 0.102 | 0.111 | 0.244 | 0.295 | False |
| bush_or_small_tree/P2@0.8/iou_Hnear/pooled | -0.035 | 0.014 | 0.025 | 0.036 | 0.069 | False |
| bush_or_small_tree/P2@0.8/iou_Hnear/mean | 0.018 | 0.038 | 0.043 | 0.088 | 0.097 | False |
| bush_or_small_tree/P1/recall_Hnear/pooled | -0.014 | 0.010 | 0.004 | 0.027 | 0.007 | False |
| bush_or_small_tree/P1/recall_Hnear/mean | -0.071 | 0.014 | 0.012 | 0.034 | 0.028 | True |
| bush_or_small_tree/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/P2@0.8/recall_H/pooled | -0.006 | 0.002 | 0.005 | 0.005 | 0.013 | False |
| bush_or_small_tree/P2@0.8/recall_H/mean | 0.008 | 0.020 | 0.012 | 0.047 | 0.032 | False |
| bush_or_small_tree/P2@0.8/precision_H/pooled | -0.009 | 0.058 | 0.070 | 0.141 | 0.177 | False |
| bush_or_small_tree/P2@0.8/precision_H/mean | 0.050 | 0.049 | 0.064 | 0.115 | 0.141 | False |
| bush_or_small_tree/P2@0.8/iou_H/pooled | -0.006 | 0.002 | 0.005 | 0.005 | 0.012 | False |
| bush_or_small_tree/P2@0.8/iou_H/mean | -0.005 | 0.006 | 0.006 | 0.015 | 0.016 | False |
| bush_or_small_tree/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/P2@0.8/recall/pooled | 0.039 | 0.005 | 0.008 | 0.012 | 0.020 | True |
| bush_or_small_tree/P2@0.8/recall/mean | 0.063 | 0.005 | 0.007 | 0.012 | 0.018 | True |
| bush_or_small_tree/P2@0.8/precision/pooled | 0.088 | 0.015 | 0.016 | 0.042 | 0.042 | True |
| bush_or_small_tree/P2@0.8/precision/mean | 0.062 | 0.026 | 0.038 | 0.059 | 0.089 | False |
| bush_or_small_tree/P2@0.8/iou/pooled | 0.038 | 0.005 | 0.007 | 0.012 | 0.018 | True |
| bush_or_small_tree/P2@0.8/iou/mean | 0.046 | 0.005 | 0.005 | 0.013 | 0.013 | True |
| bush_or_small_tree/P1/recall/pooled | 0.007 | 0.004 | 0.006 | 0.011 | 0.015 | False |
| bush_or_small_tree/P1/recall/mean | -0.004 | 0.007 | 0.009 | 0.015 | 0.025 | False |
| bush_or_small_tree/P1/precision/pooled | 0.008 | 0.013 | 0.010 | 0.031 | 0.023 | False |
| bush_or_small_tree/P1/precision/mean | 0.000 | 0.014 | 0.011 | 0.036 | 0.028 | False |
| bush_or_small_tree/P1/recall_H/pooled | 0.002 | 0.002 | 0.002 | 0.006 | 0.005 | False |
| bush_or_small_tree/P1/recall_H/mean | 0.007 | 0.005 | 0.003 | 0.014 | 0.009 | False |

GT cells occupied in every repeat of one arm and none of the other: {"bush_or_small_tree": {"first_only": {"blocked": 52, "no_return": 0, "passed_through": 0, "seen": 9, "unobserved": 0}, "second_only": {"blocked": 6, "no_return": 0, "passed_through": 0, "seen": 0, "unobserved": 0}}, "obstacles": {"first_only": {"blocked": 172, "no_return": 0, "passed_through": 0, "seen": 55, "unobserved": 0}, "second_only": {"blocked": 46, "no_return": 1, "passed_through": 0, "seen": 5, "unobserved": 0}}, "rock": {"first_only": {"blocked": 2, "no_return": 0, "passed_through": 0, "seen": 0, "unobserved": 0}, "second_only": {"blocked": 0, "no_return": 0, "passed_through": 0, "seen": 0, "unobserved": 0}}, "tree_trunk": {"first_only": {"blocked": 118, "no_return": 0, "passed_through": 0, "seen": 46, "unobserved": 0}, "second_only": {"blocked": 40, "no_return": 1, "passed_through": 0, "seen": 5, "unobserved": 0}}}

### june_scene_a_radar_init_loss_cam - june_scene_a_radar_init_cam

#### obstacles, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| obstacles/P2@0.5/recall_Hnear/pooled | 0.093 | 0.021 | 0.015 | 0.051 | 0.033 | True |
| obstacles/P2@0.5/recall_Hnear/mean | 0.138 | 0.016 | 0.059 | 0.036 | 0.135 | True |
| obstacles/P2@0.5/precision_Hnear/pooled | 0.042 | 0.053 | 0.172 | 0.134 | 0.411 | False |
| obstacles/P2@0.5/precision_Hnear/mean | 0.036 | 0.043 | 0.146 | 0.101 | 0.366 | False |
| obstacles/P2@0.5/iou_Hnear/pooled | 0.077 | 0.017 | 0.015 | 0.039 | 0.033 | True |
| obstacles/P2@0.5/iou_Hnear/mean | 0.063 | 0.011 | 0.031 | 0.029 | 0.077 | True |
| obstacles/P1/recall_Hnear/pooled | 0.009 | 0.004 | 0.005 | 0.011 | 0.014 | False |
| obstacles/P1/recall_Hnear/mean | 0.004 | 0.006 | 0.002 | 0.014 | 0.005 | False |
| obstacles/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/P2@0.5/recall_H/pooled | 0.023 | 0.004 | 0.003 | 0.012 | 0.008 | True |
| obstacles/P2@0.5/recall_H/mean | 0.036 | 0.007 | 0.009 | 0.016 | 0.021 | True |
| obstacles/P2@0.5/precision_H/pooled | -0.021 | 0.051 | 0.028 | 0.137 | 0.075 | False |
| obstacles/P2@0.5/precision_H/mean | 0.039 | 0.029 | 0.026 | 0.072 | 0.068 | False |
| obstacles/P2@0.5/iou_H/pooled | 0.021 | 0.004 | 0.003 | 0.011 | 0.007 | True |
| obstacles/P2@0.5/iou_H/mean | 0.022 | 0.006 | 0.004 | 0.015 | 0.009 | True |
| obstacles/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/P2@0.5/recall/pooled | 0.068 | 0.003 | 0.002 | 0.007 | 0.005 | True |
| obstacles/P2@0.5/recall/mean | 0.078 | 0.002 | 0.001 | 0.005 | 0.003 | True |
| obstacles/P2@0.5/precision/pooled | 0.012 | 0.004 | 0.004 | 0.011 | 0.010 | True |
| obstacles/P2@0.5/precision/mean | 0.054 | 0.007 | 0.015 | 0.020 | 0.039 | True |
| obstacles/P2@0.5/iou/pooled | 0.059 | 0.003 | 0.002 | 0.008 | 0.005 | True |
| obstacles/P2@0.5/iou/mean | 0.058 | 0.002 | 0.002 | 0.006 | 0.006 | True |
| obstacles/P1/recall/pooled | 0.018 | 0.003 | 0.003 | 0.006 | 0.006 | True |
| obstacles/P1/recall/mean | 0.017 | 0.002 | 0.002 | 0.006 | 0.006 | True |
| obstacles/P1/precision/pooled | 0.004 | 0.002 | 0.003 | 0.005 | 0.008 | False |
| obstacles/P1/precision/mean | 0.000 | 0.009 | 0.009 | 0.021 | 0.019 | False |
| obstacles/P1/recall_H/pooled | 0.006 | 0.001 | 0.001 | 0.004 | 0.002 | True |
| obstacles/P1/recall_H/mean | 0.011 | 0.002 | 0.001 | 0.005 | 0.001 | True |

#### obstacles, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| obstacles/P2@0.8/recall_Hnear/pooled | 0.006 | 0.012 | 0.014 | 0.033 | 0.036 | False |
| obstacles/P2@0.8/recall_Hnear/mean | 0.078 | 0.032 | 0.041 | 0.074 | 0.110 | False |
| obstacles/P2@0.8/precision_Hnear/pooled | -0.011 | 0.095 | 0.189 | 0.248 | 0.506 | False |
| obstacles/P2@0.8/precision_Hnear/mean | 0.015 | 0.088 | 0.131 | 0.197 | 0.348 | False |
| obstacles/P2@0.8/iou_Hnear/pooled | 0.005 | 0.012 | 0.014 | 0.031 | 0.035 | False |
| obstacles/P2@0.8/iou_Hnear/mean | 0.041 | 0.022 | 0.019 | 0.050 | 0.052 | False |
| obstacles/P1/recall_Hnear/pooled | 0.009 | 0.004 | 0.005 | 0.011 | 0.014 | False |
| obstacles/P1/recall_Hnear/mean | 0.004 | 0.006 | 0.002 | 0.014 | 0.005 | False |
| obstacles/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/P2@0.8/recall_H/pooled | -0.000 | 0.001 | 0.003 | 0.001 | 0.006 | False |
| obstacles/P2@0.8/recall_H/mean | 0.004 | 0.009 | 0.009 | 0.022 | 0.025 | False |
| obstacles/P2@0.8/precision_H/pooled | -0.048 | 0.050 | 0.028 | 0.135 | 0.065 | False |
| obstacles/P2@0.8/precision_H/mean | 0.002 | 0.038 | 0.005 | 0.102 | 0.014 | False |
| obstacles/P2@0.8/iou_H/pooled | -0.000 | 0.001 | 0.003 | 0.002 | 0.006 | False |
| obstacles/P2@0.8/iou_H/mean | -0.003 | 0.006 | 0.005 | 0.013 | 0.013 | False |
| obstacles/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| obstacles/P2@0.8/recall/pooled | 0.062 | 0.002 | 0.002 | 0.006 | 0.005 | True |
| obstacles/P2@0.8/recall/mean | 0.078 | 0.002 | 0.002 | 0.006 | 0.005 | True |
| obstacles/P2@0.8/precision/pooled | 0.020 | 0.003 | 0.005 | 0.006 | 0.009 | True |
| obstacles/P2@0.8/precision/mean | 0.071 | 0.010 | 0.016 | 0.027 | 0.042 | True |
| obstacles/P2@0.8/iou/pooled | 0.057 | 0.002 | 0.002 | 0.006 | 0.005 | True |
| obstacles/P2@0.8/iou/mean | 0.056 | 0.003 | 0.002 | 0.007 | 0.006 | True |
| obstacles/P1/recall/pooled | 0.018 | 0.003 | 0.003 | 0.006 | 0.006 | True |
| obstacles/P1/recall/mean | 0.017 | 0.002 | 0.002 | 0.006 | 0.006 | True |
| obstacles/P1/precision/pooled | 0.004 | 0.002 | 0.003 | 0.005 | 0.008 | False |
| obstacles/P1/precision/mean | 0.000 | 0.009 | 0.009 | 0.021 | 0.019 | False |
| obstacles/P1/recall_H/pooled | 0.006 | 0.001 | 0.001 | 0.004 | 0.002 | True |
| obstacles/P1/recall_H/mean | 0.011 | 0.002 | 0.001 | 0.005 | 0.001 | True |

#### tree_trunk, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| tree_trunk/P2@0.5/recall_Hnear/pooled | 0.010 | 0.015 | 0.013 | 0.040 | 0.032 | False |
| tree_trunk/P2@0.5/recall_Hnear/mean | 0.012 | 0.023 | 0.033 | 0.055 | 0.093 | False |
| tree_trunk/P2@0.5/precision_Hnear/pooled | 0.002 | 0.115 | 0.133 | 0.308 | 0.339 | False |
| tree_trunk/P2@0.5/precision_Hnear/mean | 0.029 | 0.076 | 0.090 | 0.181 | 0.236 | False |
| tree_trunk/P2@0.5/iou_Hnear/pooled | 0.009 | 0.015 | 0.013 | 0.039 | 0.031 | False |
| tree_trunk/P2@0.5/iou_Hnear/mean | 0.005 | 0.016 | 0.026 | 0.041 | 0.072 | False |
| tree_trunk/P1/recall_Hnear/pooled | 0.002 | 0.007 | 0.004 | 0.016 | 0.008 | False |
| tree_trunk/P1/recall_Hnear/mean | 0.000 | 0.003 | 0.003 | 0.007 | 0.005 | False |
| tree_trunk/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/P2@0.5/recall_H/pooled | 0.014 | 0.005 | 0.006 | 0.014 | 0.015 | False |
| tree_trunk/P2@0.5/recall_H/mean | 0.029 | 0.011 | 0.001 | 0.027 | 0.004 | True |
| tree_trunk/P2@0.5/precision_H/pooled | -0.013 | 0.081 | 0.052 | 0.217 | 0.128 | False |
| tree_trunk/P2@0.5/precision_H/mean | 0.065 | 0.052 | 0.021 | 0.115 | 0.056 | False |
| tree_trunk/P2@0.5/iou_H/pooled | 0.013 | 0.005 | 0.006 | 0.014 | 0.014 | False |
| tree_trunk/P2@0.5/iou_H/mean | 0.026 | 0.013 | 0.002 | 0.031 | 0.006 | True |
| tree_trunk/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/P2@0.5/recall/pooled | 0.050 | 0.004 | 0.002 | 0.010 | 0.005 | True |
| tree_trunk/P2@0.5/recall/mean | 0.059 | 0.004 | 0.003 | 0.009 | 0.007 | True |
| tree_trunk/P2@0.5/precision/pooled | 0.011 | 0.004 | 0.004 | 0.010 | 0.011 | True |
| tree_trunk/P2@0.5/precision/mean | 0.046 | 0.004 | 0.013 | 0.009 | 0.036 | True |
| tree_trunk/P2@0.5/iou/pooled | 0.045 | 0.004 | 0.002 | 0.009 | 0.004 | True |
| tree_trunk/P2@0.5/iou/mean | 0.050 | 0.004 | 0.004 | 0.011 | 0.011 | True |
| tree_trunk/P1/recall/pooled | 0.016 | 0.002 | 0.003 | 0.007 | 0.007 | True |
| tree_trunk/P1/recall/mean | 0.017 | 0.002 | 0.003 | 0.004 | 0.006 | True |
| tree_trunk/P1/precision/pooled | 0.002 | 0.002 | 0.004 | 0.004 | 0.011 | False |
| tree_trunk/P1/precision/mean | 0.008 | 0.007 | 0.011 | 0.018 | 0.028 | False |
| tree_trunk/P1/recall_H/pooled | 0.005 | 0.004 | 0.001 | 0.008 | 0.002 | False |
| tree_trunk/P1/recall_H/mean | 0.005 | 0.003 | 0.001 | 0.006 | 0.002 | True |

#### tree_trunk, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| tree_trunk/P2@0.8/recall_Hnear/pooled | -0.014 | 0.013 | 0.015 | 0.032 | 0.040 | False |
| tree_trunk/P2@0.8/recall_Hnear/mean | -0.024 | 0.024 | 0.022 | 0.060 | 0.053 | False |
| tree_trunk/P2@0.8/precision_Hnear/pooled | -0.177 | 0.217 | 0.177 | 0.524 | 0.467 | False |
| tree_trunk/P2@0.8/precision_Hnear/mean | -0.051 | 0.158 | 0.084 | 0.433 | 0.229 | False |
| tree_trunk/P2@0.8/iou_Hnear/pooled | -0.014 | 0.012 | 0.015 | 0.031 | 0.038 | False |
| tree_trunk/P2@0.8/iou_Hnear/mean | -0.019 | 0.021 | 0.018 | 0.052 | 0.046 | False |
| tree_trunk/P1/recall_Hnear/pooled | 0.002 | 0.007 | 0.004 | 0.016 | 0.008 | False |
| tree_trunk/P1/recall_Hnear/mean | 0.000 | 0.003 | 0.003 | 0.007 | 0.005 | False |
| tree_trunk/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/P2@0.8/recall_H/pooled | -0.010 | 0.004 | 0.005 | 0.010 | 0.011 | False |
| tree_trunk/P2@0.8/recall_H/mean | -0.013 | 0.006 | 0.003 | 0.016 | 0.008 | False |
| tree_trunk/P2@0.8/precision_H/pooled | -0.066 | 0.078 | 0.042 | 0.190 | 0.102 | False |
| tree_trunk/P2@0.8/precision_H/mean | -0.042 | 0.079 | 0.023 | 0.157 | 0.056 | False |
| tree_trunk/P2@0.8/iou_H/pooled | -0.010 | 0.004 | 0.005 | 0.010 | 0.011 | False |
| tree_trunk/P2@0.8/iou_H/mean | -0.011 | 0.006 | 0.003 | 0.016 | 0.006 | False |
| tree_trunk/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| tree_trunk/P2@0.8/recall/pooled | 0.053 | 0.002 | 0.002 | 0.005 | 0.006 | True |
| tree_trunk/P2@0.8/recall/mean | 0.062 | 0.003 | 0.003 | 0.008 | 0.006 | True |
| tree_trunk/P2@0.8/precision/pooled | 0.010 | 0.002 | 0.004 | 0.006 | 0.010 | True |
| tree_trunk/P2@0.8/precision/mean | 0.059 | 0.007 | 0.012 | 0.014 | 0.029 | True |
| tree_trunk/P2@0.8/iou/pooled | 0.048 | 0.002 | 0.003 | 0.006 | 0.006 | True |
| tree_trunk/P2@0.8/iou/mean | 0.049 | 0.003 | 0.004 | 0.009 | 0.009 | True |
| tree_trunk/P1/recall/pooled | 0.016 | 0.002 | 0.003 | 0.007 | 0.007 | True |
| tree_trunk/P1/recall/mean | 0.017 | 0.002 | 0.003 | 0.004 | 0.006 | True |
| tree_trunk/P1/precision/pooled | 0.002 | 0.002 | 0.004 | 0.004 | 0.011 | False |
| tree_trunk/P1/precision/mean | 0.008 | 0.007 | 0.011 | 0.018 | 0.028 | False |
| tree_trunk/P1/recall_H/pooled | 0.005 | 0.004 | 0.001 | 0.008 | 0.002 | False |
| tree_trunk/P1/recall_H/mean | 0.005 | 0.003 | 0.001 | 0.006 | 0.002 | True |

#### rock, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| rock/P2@0.5/recall_Hnear/pooled | 0.250 | 0.112 | 0.112 | 0.250 | 0.250 | False |
| rock/P2@0.5/recall_Hnear/mean | 0.250 | 0.112 | 0.112 | 0.250 | 0.250 | False |
| rock/P2@0.5/precision_Hnear/pooled | 0.500 | 0.236 | 0.577 | 0.500 | 1.000 | False |
| rock/P2@0.5/precision_Hnear/mean | 0.500 | 0.236 | 0.577 | 0.500 | 1.000 | False |
| rock/P2@0.5/iou_Hnear/pooled | 0.220 | 0.076 | 0.112 | 0.200 | 0.250 | False |
| rock/P2@0.5/iou_Hnear/mean | 0.220 | 0.076 | 0.112 | 0.200 | 0.250 | False |
| rock/P1/recall_Hnear/pooled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P1/recall_Hnear/mean | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P2@0.5/recall_H/pooled | 0.018 | 0.008 | 0.008 | 0.018 | 0.018 | False |
| rock/P2@0.5/recall_H/mean | 0.036 | 0.016 | 0.016 | 0.036 | 0.036 | False |
| rock/P2@0.5/precision_H/pooled | 0.267 | 0.091 | 0.289 | 0.167 | 0.500 | False |
| rock/P2@0.5/precision_H/mean | 0.217 | 0.112 | 0.289 | 0.250 | 0.500 | False |
| rock/P2@0.5/iou_H/pooled | 0.018 | 0.008 | 0.008 | 0.018 | 0.018 | False |
| rock/P2@0.5/iou_H/mean | 0.031 | 0.011 | 0.016 | 0.029 | 0.036 | False |
| rock/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P2@0.5/recall/pooled | 0.099 | 0.006 | 0.017 | 0.012 | 0.047 | True |
| rock/P2@0.5/recall/mean | 0.095 | 0.007 | 0.017 | 0.017 | 0.042 | True |
| rock/P2@0.5/precision/pooled | 0.010 | 0.005 | 0.017 | 0.014 | 0.047 | False |
| rock/P2@0.5/precision/mean | 0.012 | 0.018 | 0.076 | 0.045 | 0.198 | False |
| rock/P2@0.5/iou/pooled | 0.034 | 0.003 | 0.011 | 0.009 | 0.029 | True |
| rock/P2@0.5/iou/mean | 0.029 | 0.003 | 0.016 | 0.009 | 0.035 | True |
| rock/P1/recall/pooled | 0.035 | 0.019 | 0.017 | 0.047 | 0.047 | False |
| rock/P1/recall/mean | 0.033 | 0.021 | 0.014 | 0.054 | 0.038 | False |
| rock/P1/precision/pooled | 0.015 | 0.015 | 0.013 | 0.036 | 0.029 | False |
| rock/P1/precision/mean | 0.014 | 0.017 | 0.012 | 0.041 | 0.029 | False |
| rock/P1/recall_H/pooled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P1/recall_H/mean | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |

#### rock, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| rock/P2@0.8/recall_Hnear/pooled | 0.300 | 0.112 | 0.000 | 0.250 | 0.000 | True |
| rock/P2@0.8/recall_Hnear/mean | 0.300 | 0.112 | 0.000 | 0.250 | 0.000 | True |
| rock/P2@0.8/precision_Hnear/pooled | 0.900 | 0.224 | n/a | 0.500 | 0.000 | None |
| rock/P2@0.8/precision_Hnear/mean | 0.900 | 0.224 | n/a | 0.500 | 0.000 | None |
| rock/P2@0.8/iou_Hnear/pooled | 0.290 | 0.119 | 0.000 | 0.300 | 0.000 | True |
| rock/P2@0.8/iou_Hnear/mean | 0.290 | 0.119 | 0.000 | 0.300 | 0.000 | True |
| rock/P1/recall_Hnear/pooled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P1/recall_Hnear/mean | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P2@0.8/recall_H/pooled | 0.022 | 0.008 | 0.000 | 0.018 | 0.000 | True |
| rock/P2@0.8/recall_H/mean | 0.043 | 0.016 | 0.000 | 0.036 | 0.000 | True |
| rock/P2@0.8/precision_H/pooled | 0.733 | 0.365 | n/a | 0.667 | 0.000 | None |
| rock/P2@0.8/precision_H/mean | 0.717 | 0.389 | n/a | 0.750 | 0.000 | None |
| rock/P2@0.8/iou_H/pooled | 0.022 | 0.008 | 0.000 | 0.019 | 0.000 | True |
| rock/P2@0.8/iou_H/mean | 0.041 | 0.017 | 0.000 | 0.043 | 0.000 | True |
| rock/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P2@0.8/recall/pooled | 0.087 | 0.011 | 0.019 | 0.024 | 0.047 | True |
| rock/P2@0.8/recall/mean | 0.088 | 0.011 | 0.019 | 0.025 | 0.042 | True |
| rock/P2@0.8/precision/pooled | -0.011 | 0.006 | 0.023 | 0.015 | 0.063 | False |
| rock/P2@0.8/precision/mean | -0.047 | 0.053 | 0.137 | 0.108 | 0.317 | False |
| rock/P2@0.8/iou/pooled | 0.028 | 0.005 | 0.012 | 0.012 | 0.033 | True |
| rock/P2@0.8/iou/mean | 0.024 | 0.006 | 0.014 | 0.015 | 0.033 | False |
| rock/P1/recall/pooled | 0.035 | 0.019 | 0.017 | 0.047 | 0.047 | False |
| rock/P1/recall/mean | 0.033 | 0.021 | 0.014 | 0.054 | 0.038 | False |
| rock/P1/precision/pooled | 0.015 | 0.015 | 0.013 | 0.036 | 0.029 | False |
| rock/P1/precision/mean | 0.014 | 0.017 | 0.012 | 0.041 | 0.029 | False |
| rock/P1/recall_H/pooled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| rock/P1/recall_H/mean | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |

#### bush_or_small_tree, P2 >= 0.5

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| bush_or_small_tree/P2@0.5/recall_Hnear/pooled | 0.162 | 0.031 | 0.021 | 0.068 | 0.048 | True |
| bush_or_small_tree/P2@0.5/recall_Hnear/mean | 0.266 | 0.052 | 0.098 | 0.116 | 0.206 | True |
| bush_or_small_tree/P2@0.5/precision_Hnear/pooled | 0.065 | 0.057 | 0.220 | 0.129 | 0.541 | False |
| bush_or_small_tree/P2@0.5/precision_Hnear/mean | -0.024 | 0.035 | 0.179 | 0.076 | 0.483 | False |
| bush_or_small_tree/P2@0.5/iou_Hnear/pooled | 0.122 | 0.023 | 0.021 | 0.053 | 0.046 | True |
| bush_or_small_tree/P2@0.5/iou_Hnear/mean | 0.128 | 0.041 | 0.041 | 0.098 | 0.095 | True |
| bush_or_small_tree/P1/recall_Hnear/pooled | 0.016 | 0.010 | 0.008 | 0.027 | 0.021 | False |
| bush_or_small_tree/P1/recall_Hnear/mean | 0.010 | 0.014 | 0.002 | 0.034 | 0.006 | False |
| bush_or_small_tree/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/P2@0.5/recall_H/pooled | 0.028 | 0.009 | 0.005 | 0.021 | 0.013 | True |
| bush_or_small_tree/P2@0.5/recall_H/mean | 0.050 | 0.019 | 0.032 | 0.051 | 0.080 | False |
| bush_or_small_tree/P2@0.5/precision_H/pooled | 0.045 | 0.073 | 0.062 | 0.202 | 0.159 | False |
| bush_or_small_tree/P2@0.5/precision_H/mean | -0.021 | 0.038 | 0.042 | 0.095 | 0.112 | False |
| bush_or_small_tree/P2@0.5/iou_H/pooled | 0.026 | 0.008 | 0.004 | 0.020 | 0.012 | True |
| bush_or_small_tree/P2@0.5/iou_H/mean | 0.010 | 0.008 | 0.009 | 0.020 | 0.026 | False |
| bush_or_small_tree/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/P2@0.5/recall/pooled | 0.106 | 0.008 | 0.007 | 0.021 | 0.015 | True |
| bush_or_small_tree/P2@0.5/recall/mean | 0.141 | 0.009 | 0.013 | 0.021 | 0.029 | True |
| bush_or_small_tree/P2@0.5/precision/pooled | 0.113 | 0.022 | 0.020 | 0.059 | 0.046 | True |
| bush_or_small_tree/P2@0.5/precision/mean | 0.100 | 0.033 | 0.032 | 0.073 | 0.073 | True |
| bush_or_small_tree/P2@0.5/iou/pooled | 0.095 | 0.008 | 0.006 | 0.021 | 0.014 | True |
| bush_or_small_tree/P2@0.5/iou/mean | 0.094 | 0.006 | 0.006 | 0.013 | 0.014 | True |
| bush_or_small_tree/P1/recall/pooled | 0.021 | 0.004 | 0.004 | 0.011 | 0.011 | True |
| bush_or_small_tree/P1/recall/mean | 0.010 | 0.007 | 0.005 | 0.015 | 0.014 | False |
| bush_or_small_tree/P1/precision/pooled | 0.024 | 0.013 | 0.012 | 0.031 | 0.029 | False |
| bush_or_small_tree/P1/precision/mean | -0.007 | 0.014 | 0.019 | 0.036 | 0.043 | False |
| bush_or_small_tree/P1/recall_H/pooled | 0.006 | 0.002 | 0.001 | 0.006 | 0.003 | True |
| bush_or_small_tree/P1/recall_H/mean | 0.029 | 0.005 | 0.001 | 0.014 | 0.002 | True |

#### bush_or_small_tree, P2 >= 0.8

| metric | gap | first std | second std | first range | second range | separated |
|---|---|---|---|---|---|---|
| bush_or_small_tree/P2@0.8/recall_Hnear/pooled | 0.015 | 0.016 | 0.018 | 0.041 | 0.041 | False |
| bush_or_small_tree/P2@0.8/recall_Hnear/mean | 0.160 | 0.075 | 0.082 | 0.171 | 0.194 | False |
| bush_or_small_tree/P2@0.8/precision_Hnear/pooled | 0.033 | 0.067 | 0.226 | 0.181 | 0.557 | False |
| bush_or_small_tree/P2@0.8/precision_Hnear/mean | -0.023 | 0.102 | 0.188 | 0.244 | 0.467 | False |
| bush_or_small_tree/P2@0.8/iou_Hnear/pooled | 0.013 | 0.014 | 0.017 | 0.036 | 0.039 | False |
| bush_or_small_tree/P2@0.8/iou_Hnear/mean | 0.084 | 0.038 | 0.029 | 0.088 | 0.069 | True |
| bush_or_small_tree/P1/recall_Hnear/pooled | 0.016 | 0.010 | 0.008 | 0.027 | 0.021 | False |
| bush_or_small_tree/P1/recall_Hnear/mean | 0.010 | 0.014 | 0.002 | 0.034 | 0.006 | False |
| bush_or_small_tree/hidden_near/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/hidden_near/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/P2@0.8/recall_H/pooled | 0.005 | 0.002 | 0.004 | 0.005 | 0.010 | False |
| bush_or_small_tree/P2@0.8/recall_H/mean | 0.027 | 0.020 | 0.030 | 0.047 | 0.072 | False |
| bush_or_small_tree/P2@0.8/precision_H/pooled | 0.036 | 0.058 | 0.068 | 0.141 | 0.160 | False |
| bush_or_small_tree/P2@0.8/precision_H/mean | 0.036 | 0.049 | 0.018 | 0.115 | 0.037 | False |
| bush_or_small_tree/P2@0.8/iou_H/pooled | 0.004 | 0.002 | 0.004 | 0.005 | 0.009 | False |
| bush_or_small_tree/P2@0.8/iou_H/mean | 0.002 | 0.006 | 0.014 | 0.015 | 0.030 | False |
| bush_or_small_tree/hidden/gt_cells | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/hidden/objects_with_hidden_gt | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | False |
| bush_or_small_tree/P2@0.8/recall/pooled | 0.080 | 0.005 | 0.003 | 0.012 | 0.008 | True |
| bush_or_small_tree/P2@0.8/recall/mean | 0.132 | 0.005 | 0.011 | 0.012 | 0.027 | True |
| bush_or_small_tree/P2@0.8/precision/pooled | 0.181 | 0.015 | 0.011 | 0.042 | 0.030 | True |
| bush_or_small_tree/P2@0.8/precision/mean | 0.139 | 0.026 | 0.042 | 0.059 | 0.101 | True |
| bush_or_small_tree/P2@0.8/iou/pooled | 0.077 | 0.005 | 0.003 | 0.012 | 0.008 | True |
| bush_or_small_tree/P2@0.8/iou/mean | 0.091 | 0.005 | 0.007 | 0.013 | 0.016 | True |
| bush_or_small_tree/P1/recall/pooled | 0.021 | 0.004 | 0.004 | 0.011 | 0.011 | True |
| bush_or_small_tree/P1/recall/mean | 0.010 | 0.007 | 0.005 | 0.015 | 0.014 | False |
| bush_or_small_tree/P1/precision/pooled | 0.024 | 0.013 | 0.012 | 0.031 | 0.029 | False |
| bush_or_small_tree/P1/precision/mean | -0.007 | 0.014 | 0.019 | 0.036 | 0.043 | False |
| bush_or_small_tree/P1/recall_H/pooled | 0.006 | 0.002 | 0.001 | 0.006 | 0.003 | True |
| bush_or_small_tree/P1/recall_H/mean | 0.029 | 0.005 | 0.001 | 0.014 | 0.002 | True |

GT cells occupied in every repeat of one arm and none of the other: {"bush_or_small_tree": {"first_only": {"blocked": 88, "no_return": 0, "passed_through": 0, "seen": 28, "unobserved": 0}, "second_only": {"blocked": 3, "no_return": 0, "passed_through": 0, "seen": 0, "unobserved": 0}}, "obstacles": {"first_only": {"blocked": 196, "no_return": 0, "passed_through": 0, "seen": 109, "unobserved": 0}, "second_only": {"blocked": 53, "no_return": 0, "passed_through": 0, "seen": 1, "unobserved": 0}}, "rock": {"first_only": {"blocked": 3, "no_return": 0, "passed_through": 0, "seen": 0, "unobserved": 0}, "second_only": {"blocked": 0, "no_return": 0, "passed_through": 0, "seen": 0, "unobserved": 0}}, "tree_trunk": {"first_only": {"blocked": 105, "no_return": 0, "passed_through": 0, "seen": 81, "unobserved": 0}, "second_only": {"blocked": 50, "no_return": 0, "passed_through": 0, "seen": 1, "unobserved": 0}}}

## Guard: image quality (test + guard frames)

| metric | june_scene_a_cam | june_scene_a_radar_init_cam | june_scene_a_radar_init_loss_cam |
|---|---|---|---|
| image/psnr | 15.917 ± 0.035 (n=5) | 15.861 ± 0.040 (n=5) | 15.887 ± 0.012 (n=5) |
| image/ssim | 0.413 ± 0.000 (n=5) | 0.410 ± 0.000 (n=5) | 0.408 ± 0.000 (n=5) |
| image/lpips | 0.688 ± 0.002 (n=5) | 0.707 ± 0.003 (n=5) | 0.635 ± 0.001 (n=5) |

## Report only: lidar render

| metric | june_scene_a_cam | june_scene_a_radar_init_cam | june_scene_a_radar_init_loss_cam |
|---|---|---|---|
| lidar_render/chamfer_distance | 0.739 ± 0.021 (n=5) | 0.821 ± 0.038 (n=5) | 0.703 ± 0.019 (n=5) |
| lidar_render/median_chamfer_distance | 0.868 ± 0.074 (n=5) | 0.875 ± 0.034 (n=5) | 1.626 ± 0.315 (n=5) |
| lidar_render/depth_median_l2 | 0.037 ± 0.000 (n=5) | 0.041 ± 0.001 (n=5) | 0.040 ± 0.000 (n=5) |
| lidar_render/depth_mean_rel_l2 | 103.530 ± 2.783 (n=5) | 99.787 ± 7.653 (n=5) | 111.345 ± 5.044 (n=5) |

