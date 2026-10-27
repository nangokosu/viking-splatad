# Final fix report
Status: DONE.
1. scripts/11 panel_y_limits: np.isfinite filter, empty -> zero-range path (np.zeros(1)); CSV keeps raw nan/inf (documented in module docstring and SKILL). rolling_median/run_step_range need no change (no crash; steps are ints).
   RED (image): 4 failed, 2 passed (3 plot tests: "Axis limits cannot be NaN or Inf", 1 constants test). GREEN (image): 18 passed (final_fixes, plot_overfitting_curves, eval_split_whole_blocks, overfitting_job_wiring).
2. SKILL.md Overfitting monitor: plot-only failure recovery paragraph; glossary: overfitting_plot_recovered_utc row.
3. ZERO_RANGE_REFERENCE_MAGNITUDE (scripts/11 ~67). 4. FIGURE_DPI/LEGEND_FONT_SIZE/TITLE_FONT_SIZE (~63).
5. scripts/09 ~286 case-insensitive parse. 6. scripts/09 new long lines wrapped (pre-existing long lines untouched).
7. Glossary + SKILL wording ("absent in records written before this change; false / cores when 09 re-run"), glossary val-derivation note. Also reworded "lack" in the schema row.
8. SKILL overview: plot step added as step 2 (renumbered).
9. submit_run.py lazy-import comment; sbatch --eval-whole-test-blocks comment moved below the (d) header.
Mac: uv run pytest -q -> 446 passed, 14 skipped, 3 deselected.
New tests: tests/unit/test_overfitting_final_fixes.py.
Concerns: none; the item-6 line numbers in the brief (~426, ~610) did not exist in 09, so I wrapped every added line >100 chars.
