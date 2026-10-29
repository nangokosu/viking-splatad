#!/bin/sh
# Run tests (or any command) inside the pinned reference image, so results do not depend on the
# host's Python or libraries. Reproducibility rule: CLAUDE.md, "Project rules" 9.
#
# Usage:  scripts/test_in_image.sh                        # full suite
#         scripts/test_in_image.sh python scripts/affected_tests.py --run
#         scripts/test_in_image.sh python -m pytest -q tests/unit/test_slurm_submit.py
#
# Environment overrides: VIKING_IMAGE (default viking-splatad:cu118).
set -eu

# Step 1: repo root (this script lives in <root>/scripts/) and the image to use.
repo_root=$(cd "$(dirname "$0")/.." && pwd)
image_name=${VIKING_IMAGE:-viking-splatad:cu118}
# Step 2: no arguments = full suite; the cache provider is off because the repo mount is read-only.
if [ "$#" -eq 0 ]; then
  set -- python -m pytest -q -p no:cacheprovider
fi
# Step 3: the container's own user is root, which tests/test_no_personal_info.py would treat as the
# running user's identifiers. Hand it the host's real ones instead: login (LOGNAME), home path (HOME),
# git config (mounted read-only, found through GIT_CONFIG_GLOBAL) and ssh config (copied inside the
# container at root's passwd home, which is where ssh looks, with the owner ssh requires).
# A missing host file is replaced by /dev/null so the mount never fails.
host_git_config=$HOME/.gitconfig
host_ssh_config=$HOME/.ssh/config
[ -f "$host_git_config" ] || host_git_config=/dev/null
[ -f "$host_ssh_config" ] || host_ssh_config=/dev/null

# Step 4: run. The repo is mounted read-only (tests never write into it) and git is told the mount
# is safe despite the different owner (GIT_CONFIG_COUNT/KEY/VALUE pass one -c safe.directory).
exec docker run --rm --platform linux/amd64 \
  -e "LOGNAME=$(id -un)" -e "HOME=$HOME" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e GIT_CONFIG_GLOBAL=/host/gitconfig \
  -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0=/workspace/viking-splatad \
  -v "$repo_root":/workspace/viking-splatad:ro \
  -v "$host_git_config":/host/gitconfig:ro \
  -v "$host_ssh_config":/host/ssh_config:ro \
  -w /workspace/viking-splatad \
  "$image_name" \
  sh -c 'mkdir -p /root/.ssh && cp /host/ssh_config /root/.ssh/config && chmod 600 /root/.ssh/config && exec "$@"' sh "$@"
