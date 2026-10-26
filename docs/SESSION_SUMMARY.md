# Last session summary (2026-10-26)

## Done
- **Public-repo prep** (`90e089b`), so collaborators can use their own CS Lab logins from Codex or Claude Code:
  - `AGENTS.md` → `CLAUDE.md` and `.agents/skills` → `../.claude/skills` are symlinks; `tests/test_agent_instructions_sync.py` fails on drift.
  - CLAUDE.md is self-contained: the general code rules are inlined; the phase-1 findings live at `docs/research/phase1_findings.md`; a test workflow for agents without `transform-test-runner`.
  - Both Slurm skills have a per-user one-time setup (own `~/.ssh/config` `User` or `VIKING_SLURM_USER`, ssh key or ControlMaster, never a stored password) and agent-neutral wording.
  - `tests/test_no_personal_info.py` scans tracked files for the running user's home path, logins (local, `VIKING_SLURM_USER`, ssh-config `User`) and private git email. A planted leak fails it.
  - `.gitignore` covers local agent state and `.env` files.
- **History rewritten** with `git filter-repo`: the cluster username → `<user>`, the home path → `~/`, and every author/committer email → the GitHub noreply address. **Every SHA changed.**
  - The repo-local `user.email` is the noreply address.
  - A pre-rewrite mirror is at `../viking-splatad-prerewrite-backup.git`. It is local only: never push it.
  - Old → new SHA map: `.git/filter-repo/commit-map`.
- Full suite before the rewrite: 300 passed, 3 skipped (all pre-existing). Only new tests were added.

## Current state
- **No remote yet.** The repo has not been pushed or made public.
- **Cluster (`/w/20262/<user>/viking-splatad`):**
  - All three datasets (with the no-return sidecar) are staged.
  - The staged code folders carry **pre-rewrite** SHAs, so they no longer match any commit. The next `stage` uploads HEAD.
  - `runs/` is empty, and `envs/` holds only `splatad-3f478d42f203`. The env name is a content hash, so the rewrite does not affect it.
  - Run IDs in the local `runs/submitted.tsv` keep their pre-rewrite SHAs. They are names of deleted runs.
- **Environment: `envs/splatad-3f478d42f203` is built and marked complete** (job 222829). GPU check OK; the locks match the committed ones.
- **Elevation-mapping bias (not fixed):** T16's per-ring median includes near-return parallax. For example, ring 31 is −38.1° in the map but −44.5° in the fitted beam model. Consider using the T51 far-field elevations.
- Home: `~/.cache/yarn` (cluster) is left as is, pending a decision.
- Carried over:
  - The test split is interleaved, so it tests interpolation only.
  - The viser client doesn't build on the nodes.
  - The local Docker image predates `6ad8039`, so bind-mount `src/` for CPU checks.

## Next steps
1. Create the GitHub repo, push, make it public and add the collaborators (the user's call). Delete the backup mirror once the push is verified.
2. Each collaborator does the one-time setup in `running-splatad-render-jobs`, then stages the data and builds the env under their own workdisk directory.
3. Smoke run on `may_chunks0_1_smoke` with `add_missing_points` on (only when the user asks).
   - Run `stage --session may_chunks0_1_smoke` first.
   - Check that `ray_drop_accuracy` falls below 1 and that `job_info.txt` shows `ADD_MISSING_POINTS: True`.
   - Compare times with job 222633 (train 538 s, lidar render 73 s, without no-returns).
4. Decide on the T16 elevation mapping (median vs. far-field fit) before the full runs.
5. Full May Scene A run, then June B and R, all with the same seed and with `add_missing_points` on.
6. Evaluation per PLAN §5:
   - first the §1 June frame check and `05_extract_cuboids.py` for `june_tall_grass`;
   - then P1 (T45–T47) and P2 (T56–T57), with the seen / blocked / no-return breakdown.
