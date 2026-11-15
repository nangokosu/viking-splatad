"""An external review must bind exact identities; it cannot approve different labels."""

import pytest

from viking_splatad.eval.repeated_hidden import validate_alignment_review


def test_alignment_receipt_requires_completed_review_and_exact_sources():
    """A matching documented review is accepted; changed labels or an unfinished review is refused."""
    identities = {"session": "june", "manifest_sha256": "dataset", "cuboids_sha256": "boxes", "labels_sha256": "labels"}
    receipt = dict(identities, reviewed=True, review_scope="Known synthetic camera/label correspondence")
    digest = validate_alignment_review(receipt, [identities, dict(identities)])
    assert len(digest) == 64
    with pytest.raises(ValueError, match="exact"):
        validate_alignment_review(receipt, [dict(identities, labels_sha256="different")])
    with pytest.raises(ValueError, match="completed"):
        validate_alignment_review(dict(receipt, reviewed=False), [identities])
