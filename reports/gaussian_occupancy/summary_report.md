# Radar arms: recall, precision and IoU over the whole scene

## TL;DR

- Whole scene (pooled), P2 ≥ 0.5, init + loss − stock: june_tall_grass recall +0.038 (sep); june_tall_grass precision +0.007 (ovl); june_tall_grass IoU +0.033 (sep); may_short_grass recall +0.092 (sep); may_short_grass precision −0.079 (sep); may_short_grass IoU +0.067 (sep).
- Whole scene (pooled), P2 ≥ 0.8, init + loss − stock: june_tall_grass recall +0.034 (sep); june_tall_grass precision +0.010 (ovl); june_tall_grass IoU +0.031 (sep); may_short_grass recall +0.082 (sep); may_short_grass precision −0.091 (sep); may_short_grass IoU +0.063 (sep).

Radar init alone is in the tables below. Every gap is read only when it exceeds the spread within the arms (sep = the two arms' [min, max] ranges do not overlap).

## Whole scene: obstacles, pooled over every scored cell

### P2 occupied at ≥ 0.5

| metric | june_tall_grass stock | june_tall_grass radar init | june_tall_grass radar init + loss | may_short_grass stock | may_short_grass radar init | may_short_grass radar init + loss |
|---|---|---|---|---|---|---|
| recall | 0.342 ± 0.006 | 0.312 ± 0.002 | 0.380 ± 0.003 | 0.278 ± 0.003 | 0.221 ± 0.003 | 0.370 ± 0.003 |
| precision | 0.789 ± 0.005 | 0.784 ± 0.004 | 0.796 ± 0.004 | 0.838 ± 0.004 | 0.810 ± 0.005 | 0.758 ± 0.006 |
| IoU | 0.313 ± 0.005 | 0.287 ± 0.002 | 0.346 ± 0.003 | 0.264 ± 0.003 | 0.210 ± 0.003 | 0.331 ± 0.002 |

| metric | june_tall_grass radar init − stock | june_tall_grass radar init + loss − stock | may_short_grass radar init − stock | may_short_grass radar init + loss − stock |
|---|---|---|---|---|
| recall | −0.030 (sep) | +0.038 (sep) | −0.058 (sep) | +0.092 (sep) |
| precision | −0.005 (ovl) | +0.007 (ovl) | −0.027 (sep) | −0.079 (sep) |
| IoU | −0.026 (sep) | +0.033 (sep) | −0.054 (sep) | +0.067 (sep) |

### P2 occupied at ≥ 0.8

| metric | june_tall_grass stock | june_tall_grass radar init | june_tall_grass radar init + loss | may_short_grass stock | may_short_grass radar init | may_short_grass radar init + loss |
|---|---|---|---|---|---|---|
| recall | 0.304 ± 0.002 | 0.276 ± 0.002 | 0.338 ± 0.002 | 0.244 ± 0.002 | 0.190 ± 0.002 | 0.326 ± 0.001 |
| precision | 0.813 ± 0.005 | 0.803 ± 0.005 | 0.823 ± 0.003 | 0.863 ± 0.003 | 0.831 ± 0.002 | 0.772 ± 0.003 |
| IoU | 0.284 ± 0.002 | 0.259 ± 0.002 | 0.315 ± 0.002 | 0.234 ± 0.002 | 0.183 ± 0.002 | 0.297 ± 0.001 |

| metric | june_tall_grass radar init − stock | june_tall_grass radar init + loss − stock | may_short_grass radar init − stock | may_short_grass radar init + loss − stock |
|---|---|---|---|---|
| recall | −0.028 (sep) | +0.034 (sep) | −0.053 (sep) | +0.082 (sep) |
| precision | −0.010 (ovl) | +0.010 (ovl) | −0.032 (sep) | −0.091 (sep) |
| IoU | −0.025 (sep) | +0.031 (sep) | −0.051 (sep) | +0.063 (sep) |

## Render quality: june_tall_grass

Image metrics (test + guard frames):

| metric | stock | radar init | radar init + loss |
|---|---|---|---|
| psnr | 15.917 ± 0.035 | 15.861 ± 0.040 | 15.887 ± 0.012 |
| ssim | 0.413 ± 0.000 | 0.410 ± 0.000 | 0.408 ± 0.000 |
| lpips | 0.688 ± 0.002 | 0.707 ± 0.003 | 0.635 ± 0.001 |

Lidar render metrics:

| metric | stock | radar init | radar init + loss |
|---|---|---|---|
| chamfer_distance | 0.739 ± 0.021 | 0.821 ± 0.038 | 0.703 ± 0.019 |
| median_chamfer_distance | 0.868 ± 0.074 | 0.875 ± 0.034 | 1.626 ± 0.315 |
| depth_median_l2 | 0.037 ± 0.000 | 0.041 ± 0.001 | 0.040 ± 0.000 |
| depth_mean_rel_l2 | 103.530 ± 2.783 | 99.787 ± 7.653 | 111.345 ± 5.044 |

## Render quality: may_short_grass

Image metrics (test + guard frames):

| metric | stock | radar init | radar init + loss |
|---|---|---|---|
| psnr | 18.215 ± 0.050 | 17.849 ± 0.061 | 18.118 ± 0.042 |
| ssim | 0.535 ± 0.000 | 0.531 ± 0.001 | 0.523 ± 0.000 |
| lpips | 0.637 ± 0.001 | 0.653 ± 0.002 | 0.602 ± 0.001 |

Lidar render metrics:

| metric | stock | radar init | radar init + loss |
|---|---|---|---|
| chamfer_distance | 1.039 ± 0.039 | 1.808 ± 0.113 | 0.978 ± 0.010 |
| median_chamfer_distance | 1.012 ± 0.051 | 1.724 ± 0.132 | 1.054 ± 0.312 |
| depth_median_l2 | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.000 |
| depth_mean_rel_l2 | 0.277 ± 0.122 | 0.125 ± 0.012 | 0.137 ± 0.010 |

## Definitions

- **Recall**: share of obstacle cells the model fills. **Precision**: share of filled cells near an obstacle that are obstacle cells. **IoU**: both errors in one number (overlap ÷ union in the scored area).
- **P2 occupied at ≥ t**: a cell counts as filled when the model's occupancy there is at least t. 0.5 is the binding threshold; 0.8 is a stricter check.
- **Whole scene**: all scored cells, pooled.
- **Arms**: stock (no radar); radar init (radar seeds at initialisation); init + loss (radar seeds and a label-free opacity loss near them). Each arm trained 5 times per session.
- Sources: reports/gaussian_occupancy/june_tall_grass/gaussian_occupancy_summary.json, reports/gaussian_occupancy/may_short_grass/gaussian_occupancy_summary.json (scripts/20), built by docs/research/build_summary_report.py.

