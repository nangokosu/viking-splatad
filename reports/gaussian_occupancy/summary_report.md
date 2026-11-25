# Radar arms: recall, precision and IoU, whole scene and hidden near band

## TL;DR

- Whole scene (pooled), P2 ≥ 0.5, init + loss − stock: june_tall_grass recall +0.038 (sep); june_tall_grass precision +0.007 (ovl); june_tall_grass IoU +0.033 (sep); may_short_grass recall +0.092 (sep); may_short_grass precision −0.079 (sep); may_short_grass IoU +0.067 (sep).
- Whole scene (pooled), P2 ≥ 0.8, init + loss − stock: june_tall_grass recall +0.034 (sep); june_tall_grass precision +0.010 (ovl); june_tall_grass IoU +0.031 (sep); may_short_grass recall +0.082 (sep); may_short_grass precision −0.091 (sep); may_short_grass IoU +0.063 (sep).
- Hidden near band (mean over objects), P2 ≥ 0.5, init + loss − stock: june_tall_grass recall +0.104 (sep); june_tall_grass precision −0.025 (ovl); june_tall_grass IoU +0.033 (ovl); may_short_grass recall +0.195 (sep); may_short_grass precision −0.009 (ovl); may_short_grass IoU +0.048 (ovl).
- Hidden near band (mean over objects), P2 ≥ 0.8, init + loss − stock: june_tall_grass recall +0.068 (sep); june_tall_grass precision −0.061 (ovl); june_tall_grass IoU +0.034 (ovl); may_short_grass recall +0.169 (sep); may_short_grass precision −0.057 (ovl); may_short_grass IoU +0.029 (ovl).

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

## Hidden near band (H_near): obstacles, mean over objects

- june_tall_grass: 332 hidden obstacle GT cells in 21 objects (H_near).
- may_short_grass: 193 hidden obstacle GT cells in 13 objects (H_near).

### P2 occupied at ≥ 0.5

| metric | june_tall_grass stock | june_tall_grass radar init | june_tall_grass radar init + loss | may_short_grass stock | may_short_grass radar init | may_short_grass radar init + loss |
|---|---|---|---|---|---|---|
| recall | 0.124 ± 0.038 | 0.085 ± 0.029 | 0.228 ± 0.034 | 0.194 ± 0.059 | 0.190 ± 0.049 | 0.388 ± 0.038 |
| precision | 0.483 ± 0.037 | 0.502 ± 0.123 | 0.458 ± 0.036 | 0.466 ± 0.041 | 0.570 ± 0.084 | 0.457 ± 0.038 |
| IoU | 0.101 ± 0.029 | 0.070 ± 0.026 | 0.134 ± 0.024 | 0.156 ± 0.042 | 0.160 ± 0.033 | 0.204 ± 0.030 |

| metric | june_tall_grass radar init − stock | june_tall_grass radar init + loss − stock | may_short_grass radar init − stock | may_short_grass radar init + loss − stock |
|---|---|---|---|---|
| recall | −0.038 (ovl) | +0.104 (sep) | −0.003 (ovl) | +0.195 (sep) |
| precision | +0.019 (ovl) | −0.025 (ovl) | +0.105 (ovl) | −0.009 (ovl) |
| IoU | −0.031 (ovl) | +0.033 (ovl) | +0.004 (ovl) | +0.048 (ovl) |

### P2 occupied at ≥ 0.8

| metric | june_tall_grass stock | june_tall_grass radar init | june_tall_grass radar init + loss | may_short_grass stock | may_short_grass radar init | may_short_grass radar init + loss |
|---|---|---|---|---|---|---|
| recall | 0.080 ± 0.025 | 0.063 ± 0.035 | 0.148 ± 0.022 | 0.168 ± 0.062 | 0.144 ± 0.021 | 0.337 ± 0.021 |
| precision | 0.484 ± 0.050 | 0.467 ± 0.105 | 0.424 ± 0.033 | 0.536 ± 0.127 | 0.556 ± 0.101 | 0.479 ± 0.043 |
| IoU | 0.067 ± 0.018 | 0.053 ± 0.030 | 0.101 ± 0.018 | 0.145 ± 0.050 | 0.119 ± 0.030 | 0.174 ± 0.018 |

| metric | june_tall_grass radar init − stock | june_tall_grass radar init + loss − stock | may_short_grass radar init − stock | may_short_grass radar init + loss − stock |
|---|---|---|---|---|
| recall | −0.017 (ovl) | +0.068 (sep) | −0.024 (ovl) | +0.169 (sep) |
| precision | −0.017 (ovl) | −0.061 (ovl) | +0.020 (ovl) | −0.057 (ovl) |
| IoU | −0.014 (ovl) | +0.034 (ovl) | −0.026 (ovl) | +0.029 (ovl) |

## Definitions

- **Recall**: share of obstacle cells the model fills. **Precision**: share of filled cells near an obstacle that are obstacle cells. **IoU**: both errors in one number (overlap ÷ union in the scored area).
- **P2 occupied at ≥ t**: a cell counts as filled when the model's occupancy there is at least t. 0.5 is the binding threshold; 0.8 is a stricter check.
- **Whole scene**: all scored cells, pooled. **Hidden near band (H_near)**: cells no lidar beam reached, within 20 m of the driven path.
- **Arms**: stock (no radar); radar init (radar seeds at initialisation); init + loss (radar seeds and a label-free opacity loss near them). Each arm trained 5 times per session.
- Sources: reports/gaussian_occupancy/june_tall_grass/gaussian_occupancy_summary.json, reports/gaussian_occupancy/may_short_grass/gaussian_occupancy_summary.json (scripts/20), built by docs/research/build_summary_report.py.

