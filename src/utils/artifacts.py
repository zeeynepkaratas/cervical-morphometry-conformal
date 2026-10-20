"""Portable hashing for frozen text artifacts."""

import hashlib
from pathlib import Path


def canonical_text_sha256(path: Path) -> str:
    """Normalize only CRLF line endings, preserving all other content bytes."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verify_manifest(path: Path, summary: dict) -> str:
    expected = summary.get("canonical_manifest_sha256")
    observed = canonical_text_sha256(path) if expected else hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if observed != (expected or summary["manifest_sha256"]):
        raise ValueError("Cx22 ShiftEval manifest differs from the frozen pre-inference manifest.")
    return observed
