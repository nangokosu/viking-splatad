# Session summary

Date: 2026-11-11

## Done

- Reread the uploaded DOCX, distinguished its classification, rendered-radar and
  final label-free opacity drafts. User explicitly selected loss-only ablation
  with no radar Gaussian initialization. Saved the implementation/acceptance plan
  in `docs/research/proposal_implementation_plan.md`; no training code changed.
- Prior implementation: `0097a66` passed 1,143 native tests with one allowed skip
  and eight configured deselections. `06e1653` recorded baseline submissions,
  workdisk-only workflow and dated native dependency locks.

## Current state

- June chunks 20–22 were validated and published on the cluster; reference split
  matched. Native env is `splatad-3f478d42f203`. No raw data downloaded to the Mac.
- Baseline smoke 241036 and full B 241037 were submitted on `0097a66` November 9.
  Current status and automatic fetching are unverified: SSH authentication fails
  on November 11. Never infer queue state or run success from that failure.
- Opacity loss and basic train-metadata loader exist. Calibrated train-only radar
  anchors, provenance/cropping contract, exact stock-config preservation, measured
  loss weight, loss-only profiles and complete hidden-trunk evaluation are missing.
- Confirmed new scope is B versus L with identical stock Gaussian initialization.
  Radar anchors supervise opacity only. No Gaussian append, radar rendering,
  semantic training labels, depth loss or photometric extension in this work.
- No existing tests changed or new tests run for this documentation-only plan.

## Next steps

1. Implement calibrated, train-only radar anchor construction and input gates
   following the saved plan, without modifying the canonical split/published B data.
2. Complete and verify loss-only registration/config, exact B/L initialization
   equality, support diagnostics and initial loss-weight calibration.
3. Restore authenticated cluster access, reconcile existing jobs and fetch verified
   success; validate synthetic cases and L smoke before repeated B/L full runs.
4. Compare matched held-out renders and Gaussian occupancy, including hidden-target
   recall, precision/false positives, pictures and repeat spread.
