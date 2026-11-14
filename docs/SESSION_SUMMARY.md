# Session summary

Date: 2026-11-14

## Done

- Reviewed publication of all project work committed through `cc932e4` on
  `Neel-RLF` (38 commits ahead of the last recorded upstream before this handoff).
  Working tree was clean. No source, dependency or test changes today.
- Checked changed tracked files for personal identifiers and credential patterns:
  no hits. Generated datasets, run outputs, local runtime scripts and the visual
  gallery remain Git-ignored; they are not force-added to the project history.
- Last completed native validation: 1,194 tests on summary/review source `394db6a`,
  one allowed traceability skip and eight configured deselections, under the
  standing local-Docker exception. Main evaluator source `f43e055` passed 1,189.

## Current state

- User requested commit/push of all project changes; this handoff accompanies
  the branch publication. Verify upstream equals HEAD after pushing.
- Cluster status could not be reconciled this session; the sandboxed check failed
  to resolve the host. No queue emptiness, completion or failure inferred.
- Last verified November 12: original B/L full and smoke runs complete/fetched;
  first hidden-trunk comparison 244829 complete/fetched (175 trunk views).
  Repeat B/L jobs: 244827/244828. Their comparison: 244830. Final repeated summary:
  244855. Their current outcomes need checking. Old 244546 is held/superseded;
  never release its covariance-based evaluator.
- Training sources: `0097a66` B / `da76f8d` L. Environment unchanged:
  `splatad-3f478d42f203`. No radar initialization/rendering or camera-loss extension.
- Local preview: `data/evaluation-results/tall-grass-loss-only-preview/index.html`
  and `four-panel-comparison.png`. Layout: original, B, L, original with dataset
  trunk GT. No semantic trunk classifier; rightmost labels are not predictions.
- Results and watch records are ignored under `data/evaluation-results/` and
  `data/cluster-validation/`. Raw data, eval GT and factors remain on workdisk.
  No repeat-aware radar benefit concluded. Local occupancy FP is not global
  phantom-tree counting; geometric blockage is not a semantic grass label.

## Next steps

1. Complete/verify the requested push and ensure the branch is synchronized.
2. Restore cluster access, run `submit_run.py runs`, fetch successes automatically
   and inspect comparison 244830/summary 244855. Restart watchers only if needed;
   do not duplicate queued training/evaluations.
3. Review repeated recall/precision, within-arm spread and matched pictures before
   a research conclusion; extend repeats/sensitivity/global FP analysis as needed.
