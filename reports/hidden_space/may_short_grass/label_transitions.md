# Hidden-space label transitions: may_short_grass

- range margin: 0.4603 m; scored range: 56.18 m
- cells: new 62196 (hidden 18705); old 87068; new only 0; old only 24872
- GT cells by class (new): {'bush_or_small_tree': 7549, 'rock': 178, 'tree_trunk': 11873}
- hidden GT cells by class (new): {'bush_or_small_tree': 3068, 'rock': 97, 'tree_trunk': 2469}
- old archive: runs/may_short_grass-may_scene_a_30k-20261119-131421-d51a4d6/eval/gaussian_occupancy_cells.npz

## obstacles

GT cells in new set 19600, hidden 5634 (share 0.287); old GT cells 29277; old GT cells not GT now: 9677

### Today's label -> H

| old label | hidden | not hidden | total | hidden share |
|---|---|---|---|---|
| seen | 0 | 4253 | 4253 | 0.000 |
| blocked | 4430 | 8989 | 13419 | 0.330 |
| no_return | 1073 | 709 | 1782 | 0.602 |
| passed_through | 0 | 0 | 0 | - |
| unobserved | 131 | 15 | 146 | 0.897 |
| not_in_old_cells | 0 | 0 | 0 | - |

### New evidence of the GT cells (exclusive)

| cause | cells |
|---|---|
| hidden | 5634 |
| seen | 11698 |
| passed_through_not_seen | 2268 |
| never_blocked_nor_seen_nor_passed | 0 |

### Hidden GT cells by layer above the box bottom

| layer | hidden GT cells |
|---|---|
| 0 | 473 |
| 1 | 360 |
| 2 | 318 |
| >=3 | 4483 |
| no_box | 0 |

Hidden GT cells with a seen cell above in the same column: 770

### Horizontal distance to the nearest sweep origin

| distance | hidden GT | all GT |
|---|---|---|
| [0, 5) m | 0 | 1473 |
| [5, 10) m | 15 | 3205 |
| [10, 20) m | 122 | 4551 |
| [20, 30) m | 1232 | 4427 |
| [30, 40) m | 2115 | 3488 |
| [40, inf) m | 2150 | 2456 |

Objects: 206; with GT cells inside: 206; with hidden GT cells inside: 80

## tree_trunk

GT cells in new set 11873, hidden 2469 (share 0.208); old GT cells 15390; old GT cells not GT now: 3517

### Today's label -> H

| old label | hidden | not hidden | total | hidden share |
|---|---|---|---|---|
| seen | 0 | 3650 | 3650 | 0.000 |
| blocked | 1920 | 5237 | 7157 | 0.268 |
| no_return | 531 | 516 | 1047 | 0.507 |
| passed_through | 0 | 0 | 0 | - |
| unobserved | 18 | 1 | 19 | 0.947 |
| not_in_old_cells | 0 | 0 | 0 | - |

### New evidence of the GT cells (exclusive)

| cause | cells |
|---|---|
| hidden | 2469 |
| seen | 9018 |
| passed_through_not_seen | 386 |
| never_blocked_nor_seen_nor_passed | 0 |

### Hidden GT cells by layer above the box bottom

| layer | hidden GT cells |
|---|---|
| 0 | 99 |
| 1 | 91 |
| 2 | 83 |
| >=3 | 2196 |
| no_box | 0 |

Hidden GT cells with a seen cell above in the same column: 680

### Horizontal distance to the nearest sweep origin

| distance | hidden GT | all GT |
|---|---|---|
| [0, 5) m | 0 | 1255 |
| [5, 10) m | 0 | 1911 |
| [10, 20) m | 6 | 2230 |
| [20, 30) m | 251 | 2878 |
| [30, 40) m | 491 | 1579 |
| [40, inf) m | 1721 | 2020 |

Objects: 156; with GT cells inside: 156; with hidden GT cells inside: 52

## bush_or_small_tree

GT cells in new set 7549, hidden 3068 (share 0.406); old GT cells 13677; old GT cells not GT now: 6128

### Today's label -> H

| old label | hidden | not hidden | total | hidden share |
|---|---|---|---|---|
| seen | 0 | 571 | 571 | 0.000 |
| blocked | 2418 | 3703 | 6121 | 0.395 |
| no_return | 537 | 193 | 730 | 0.736 |
| passed_through | 0 | 0 | 0 | - |
| unobserved | 113 | 14 | 127 | 0.890 |
| not_in_old_cells | 0 | 0 | 0 | - |

### New evidence of the GT cells (exclusive)

| cause | cells |
|---|---|
| hidden | 3068 |
| seen | 2609 |
| passed_through_not_seen | 1872 |
| never_blocked_nor_seen_nor_passed | 0 |

### Hidden GT cells by layer above the box bottom

| layer | hidden GT cells |
|---|---|
| 0 | 307 |
| 1 | 237 |
| 2 | 236 |
| >=3 | 2288 |
| no_box | 0 |

Hidden GT cells with a seen cell above in the same column: 90

### Horizontal distance to the nearest sweep origin

| distance | hidden GT | all GT |
|---|---|---|
| [0, 5) m | 0 | 193 |
| [5, 10) m | 15 | 1257 |
| [10, 20) m | 116 | 2313 |
| [20, 30) m | 976 | 1540 |
| [30, 40) m | 1618 | 1903 |
| [40, inf) m | 343 | 343 |

Objects: 31; with GT cells inside: 31; with hidden GT cells inside: 16

## rock

GT cells in new set 178, hidden 97 (share 0.545); old GT cells 210; old GT cells not GT now: 32

### Today's label -> H

| old label | hidden | not hidden | total | hidden share |
|---|---|---|---|---|
| seen | 0 | 32 | 32 | 0.000 |
| blocked | 92 | 49 | 141 | 0.652 |
| no_return | 5 | 0 | 5 | 1.000 |
| passed_through | 0 | 0 | 0 | - |
| unobserved | 0 | 0 | 0 | - |
| not_in_old_cells | 0 | 0 | 0 | - |

### New evidence of the GT cells (exclusive)

| cause | cells |
|---|---|
| hidden | 97 |
| seen | 71 |
| passed_through_not_seen | 10 |
| never_blocked_nor_seen_nor_passed | 0 |

### Hidden GT cells by layer above the box bottom

| layer | hidden GT cells |
|---|---|
| 0 | 65 |
| 1 | 32 |
| 2 | 0 |
| >=3 | 0 |
| no_box | 0 |

Hidden GT cells with a seen cell above in the same column: 0

### Horizontal distance to the nearest sweep origin

| distance | hidden GT | all GT |
|---|---|---|
| [0, 5) m | 0 | 25 |
| [5, 10) m | 0 | 37 |
| [10, 20) m | 0 | 8 |
| [20, 30) m | 5 | 9 |
| [30, 40) m | 6 | 6 |
| [40, inf) m | 86 | 93 |

Objects: 19; with GT cells inside: 19; with hidden GT cells inside: 12
