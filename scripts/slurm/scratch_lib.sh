#!/bin/bash
# Shared helpers for the Slurm jobs (sourced by build_env.sbatch and splatad_run.sbatch).
#
# Why scratch: on the CS Lab cluster, compute jobs that write fast to NFS home are SIGKILLed
# within seconds (debug jobs 222528-222535, 2026-10-25: a plain 4 GB `dd` to home killed the
# job, the same write to /scratch/scratch-space or /tmp succeeded). Jobs may READ home freely
# (4.6 GB in 27 s, job 222536) and write to it slowly (500 MB at 20 MB/s survived). So heavy
# writes (environment, training outputs) go to scratch, and results are copied back slowly.
#
# Scratch layout: /scratch/scratch-space/expires-YYYY-Mon-DD/ directories, each lasting 15 days;
# one expiring more than five days ahead always exists (support.cs.toronto.edu slurmresource).

SCRATCH_SPACE_ROOT="/scratch/scratch-space"

# Print "<epoch seconds> <directory>" for the scratch dir that expires last.
# Directory names look like expires-2026-Nov-07; GNU date parses "07 Nov 2026".
latest_expiring_scratch() {
    local scratch_name expiry_text expiry_epoch
    for scratch_name in $(ls -1 "$SCRATCH_SPACE_ROOT" | grep -E '^expires-[0-9]{4}-[A-Za-z]{3}-[0-9]{2}$'); do
        # expires-YYYY-Mon-DD -> "DD Mon YYYY"
        expiry_text="$(echo "$scratch_name" | sed -E 's/^expires-([0-9]{4})-([A-Za-z]{3})-([0-9]{2})$/\3 \2 \1/')"
        expiry_epoch="$(date -u -d "$expiry_text" +%s)"
        echo "$expiry_epoch $SCRATCH_SPACE_ROOT/$scratch_name"
    done | sort -n | tail -1
}

# Fail unless a scratch dir expiring at $1 (epoch s) outlives this job.
# The job's own end = now + its remaining time limit (squeue %L, e.g. "1-02:03:04" or "45:00").
require_scratch_outlives_job() {
    local expiry_epoch="$1" remaining_text remaining_seconds
    remaining_text="$(squeue -h -j "$SLURM_JOB_ID" -o %L)"
    remaining_seconds="$(slurm_duration_seconds "$remaining_text")"
    if (( expiry_epoch <= $(date -u +%s) + remaining_seconds )); then
        echo "ERROR: scratch expires $(date -u -d "@$expiry_epoch") before this job's time limit ends" >&2
        return 1
    fi
}

# Convert a Slurm duration ([D-]HH:MM:SS, MM:SS or SS) to seconds.
slurm_duration_seconds() {
    local duration_text="$1" day_count=0 clock_text
    if [[ "$duration_text" == *-* ]]; then
        day_count="${duration_text%%-*}"
        clock_text="${duration_text#*-}"
    else
        clock_text="$duration_text"
    fi
    # Pad to H:M:S so 45:00 means 45 min, not 45 h.
    local field_count
    field_count="$(awk -F: '{print NF}' <<< "$clock_text")"
    case "$field_count" in
        1) clock_text="0:0:$clock_text" ;;
        2) clock_text="0:$clock_text" ;;
    esac
    # 10# forces base 10 so zero-padded fields like 08 are not read as octal.
    IFS=: read -r hour_count minute_count second_count <<< "$clock_text"
    echo $(( 10#$day_count * 86400 + 10#$hour_count * 3600 + 10#$minute_count * 60 + 10#$second_count ))
}

# Copy a directory tree to NFS home slowly enough not to trigger the bulk-write kill.
# $1 source dir, $2 destination dir, $3 bandwidth limit in KiB/s, remaining args: rsync filters.
copy_back_slowly() {
    local source_dir="$1" destination_dir="$2" bandwidth_kib="$3"
    shift 3
    mkdir -p "$destination_dir"
    rsync -a --bwlimit="$bandwidth_kib" "$@" "$source_dir/" "$destination_dir/"
}
