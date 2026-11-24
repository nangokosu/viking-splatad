# Hidden-space label transitions: june_tall_grass

- range margin: 0.3981 m; scored range: 40.18 m
- cells: new 42537 (hidden 10958); old 73806; new only 0; old only 31269
- GT cells by class (new): {'bush_or_small_tree': 3999, 'rock': 85, 'tree_trunk': 8603}
- hidden GT cells by class (new): {'bush_or_small_tree': 1759, 'rock': 55, 'tree_trunk': 981}
- old archive: runs/june_tall_grass-june_scene_a_cam-20261118-032048-c28a956/eval/gaussian_occupancy_cells.npz

## obstacles

GT cells in new set 12687, hidden 2795 (share 0.220); old GT cells 22890; old GT cells not GT now: 10203

### Today's label -> H

| old label | hidden | not hidden | total | hidden share |
|---|---|---|---|---|
| seen | 0 | 3190 | 3190 | 0.000 |
| blocked | 2680 | 6635 | 9315 | 0.288 |
| no_return | 111 | 67 | 178 | 0.624 |
| passed_through | 0 | 0 | 0 | - |
| unobserved | 4 | 0 | 4 | 1.000 |
| not_in_old_cells | 0 | 0 | 0 | - |

### New evidence of the GT cells (exclusive)

| cause | cells |
|---|---|
| hidden | 2795 |
| seen | 8907 |
| passed_through_not_seen | 985 |
| never_blocked_nor_seen_nor_passed | 0 |

### Hidden GT cells by layer above the box bottom

| layer | hidden GT cells |
|---|---|
| 0 | 398 |
| 1 | 266 |
| 2 | 242 |
| >=3 | 1889 |
| no_box | 0 |

Hidden GT cells with a seen cell above in the same column: 641

### Horizontal distance to the nearest sweep origin

| distance | hidden GT | all GT |
|---|---|---|
| [0, 5) m | 6 | 1304 |
| [5, 10) m | 6 | 827 |
| [10, 20) m | 264 | 4158 |
| [20, 30) m | 764 | 3393 |
| [30, 40) m | 1755 | 3005 |
| [40, inf) m | 0 | 0 |

Objects: 154; with GT cells inside: 154; with hidden GT cells inside: 79

## tree_trunk

GT cells in new set 8603, hidden 981 (share 0.114); old GT cells 14683; old GT cells not GT now: 6080

### Today's label -> H

| old label | hidden | not hidden | total | hidden share |
|---|---|---|---|---|
| seen | 0 | 2899 | 2899 | 0.000 |
| blocked | 955 | 4681 | 5636 | 0.169 |
| no_return | 25 | 42 | 67 | 0.373 |
| passed_through | 0 | 0 | 0 | - |
| unobserved | 1 | 0 | 1 | 1.000 |
| not_in_old_cells | 0 | 0 | 0 | - |

### New evidence of the GT cells (exclusive)

| cause | cells |
|---|---|
| hidden | 981 |
| seen | 7285 |
| passed_through_not_seen | 337 |
| never_blocked_nor_seen_nor_passed | 0 |

### Hidden GT cells by layer above the box bottom

| layer | hidden GT cells |
|---|---|
| 0 | 75 |
| 1 | 34 |
| 2 | 28 |
| >=3 | 844 |
| no_box | 0 |

Hidden GT cells with a seen cell above in the same column: 435

### Horizontal distance to the nearest sweep origin

| distance | hidden GT | all GT |
|---|---|---|
| [0, 5) m | 6 | 1276 |
| [5, 10) m | 0 | 777 |
| [10, 20) m | 120 | 2963 |
| [20, 30) m | 589 | 2632 |
| [30, 40) m | 266 | 955 |
| [40, inf) m | 0 | 0 |

Objects: 113; with GT cells inside: 113; with hidden GT cells inside: 49

## bush_or_small_tree

GT cells in new set 3999, hidden 1759 (share 0.440); old GT cells 8009; old GT cells not GT now: 4010

### Today's label -> H

| old label | hidden | not hidden | total | hidden share |
|---|---|---|---|---|
| seen | 0 | 273 | 273 | 0.000 |
| blocked | 1670 | 1942 | 3612 | 0.462 |
| no_return | 86 | 25 | 111 | 0.775 |
| passed_through | 0 | 0 | 0 | - |
| unobserved | 3 | 0 | 3 | 1.000 |
| not_in_old_cells | 0 | 0 | 0 | - |

### New evidence of the GT cells (exclusive)

| cause | cells |
|---|---|
| hidden | 1759 |
| seen | 1593 |
| passed_through_not_seen | 647 |
| never_blocked_nor_seen_nor_passed | 0 |

### Hidden GT cells by layer above the box bottom

| layer | hidden GT cells |
|---|---|
| 0 | 280 |
| 1 | 220 |
| 2 | 214 |
| >=3 | 1045 |
| no_box | 0 |

Hidden GT cells with a seen cell above in the same column: 206

### Horizontal distance to the nearest sweep origin

| distance | hidden GT | all GT |
|---|---|---|
| [0, 5) m | 0 | 4 |
| [5, 10) m | 6 | 48 |
| [10, 20) m | 140 | 1189 |
| [20, 30) m | 157 | 741 |
| [30, 40) m | 1456 | 2017 |
| [40, inf) m | 0 | 0 |

Objects: 31; with GT cells inside: 31; with hidden GT cells inside: 23

## rock

GT cells in new set 85, hidden 55 (share 0.647); old GT cells 198; old GT cells not GT now: 113

### Today's label -> H

| old label | hidden | not hidden | total | hidden share |
|---|---|---|---|---|
| seen | 0 | 18 | 18 | 0.000 |
| blocked | 55 | 12 | 67 | 0.821 |
| no_return | 0 | 0 | 0 | - |
| passed_through | 0 | 0 | 0 | - |
| unobserved | 0 | 0 | 0 | - |
| not_in_old_cells | 0 | 0 | 0 | - |

### New evidence of the GT cells (exclusive)

| cause | cells |
|---|---|
| hidden | 55 |
| seen | 29 |
| passed_through_not_seen | 1 |
| never_blocked_nor_seen_nor_passed | 0 |

### Hidden GT cells by layer above the box bottom

| layer | hidden GT cells |
|---|---|
| 0 | 43 |
| 1 | 12 |
| 2 | 0 |
| >=3 | 0 |
| no_box | 0 |

Hidden GT cells with a seen cell above in the same column: 0

### Horizontal distance to the nearest sweep origin

| distance | hidden GT | all GT |
|---|---|---|
| [0, 5) m | 0 | 24 |
| [5, 10) m | 0 | 2 |
| [10, 20) m | 4 | 6 |
| [20, 30) m | 18 | 20 |
| [30, 40) m | 33 | 33 |
| [40, inf) m | 0 | 0 |

Objects: 10; with GT cells inside: 10; with hidden GT cells inside: 7
