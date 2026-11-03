"""Regenerate tests/fixtures/test_exclusions_allowlist.json from a collect-only listing.

Run in the reference image (so the same tests are collected as in the real suite):

    scripts/test_in_image.sh python -m pytest --collect-only -q -p no:cacheprovider \
        -m "gpu or real_data" | python scripts/write_exclusions_allowlist.py > /tmp/allowlist.json

The listing on stdin holds one node id per line (pytest -q output); every line containing "::" is
a test that the default ``-m 'not gpu and not real_data'`` deselects. The only skip allowed in the
suite is test_traceability's "0 transforms pending". The result is reviewed like any test edit:
scripts/affected_tests.py flags a change to the allowlist as SUSPICIOUS.
"""

import json
import sys

# The single permitted skip and its reason (CLAUDE.md rule 9: today only "0 transforms pending").
ALLOWED_SKIP = {
    "nodeid": "tests/test_traceability.py::test_all_planned_transforms_implemented",
    "reason_regex": "^0 transforms pending",
}


def main() -> int:
    """Read node ids from stdin and print the allowlist JSON.

    Returns:
        Process exit code (0).
    """
    # Step 1: keep only node-id lines (pytest also prints a summary line without "::").
    deselected_ids = sorted({line.strip() for line in sys.stdin if "::" in line})
    # Step 2: assemble the document; count equals the number of recorded ids.
    allowlist = {
        "skipped": [ALLOWED_SKIP],
        "xfailed": [],
        "deselected_by_marker": {"count": len(deselected_ids), "nodeids": deselected_ids},
    }
    print(json.dumps(allowlist, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
