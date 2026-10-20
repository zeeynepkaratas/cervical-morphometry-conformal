import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.fetch_checkpoint import fetch_checkpoint, sha256
from experiments.train_frozen_split import load_training_split
from src.utils.config import RESULTS_TABLES
from src.utils.artifacts import canonical_text_sha256, verify_manifest


class ArtifactTests(unittest.TestCase):
    def test_manifest_line_endings_are_portable_but_content_is_locked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            path.write_bytes(b"name,value\r\nx,1\r\n")
            summary = {"canonical_manifest_sha256": canonical_text_sha256(path)}
            path.write_bytes(b"name,value\nx,1\n")
            self.assertEqual(verify_manifest(path, summary), summary["canonical_manifest_sha256"])
            path.write_bytes(b"name,value\nx,2\n")
            with self.assertRaises(ValueError):
                verify_manifest(path, summary)

    def test_committed_cx22_manifest_matches_canonical_hash(self):
        summary = json.loads((RESULTS_TABLES / "cx22_shifteval_manifest_summary.json").read_text())
        self.assertEqual(verify_manifest(RESULTS_TABLES / "cx22_shifteval_manifest.csv", summary), summary["canonical_manifest_sha256"])

    def test_corrupt_checkpoint_is_rejected_without_installing(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "bad.pt", Path(directory) / "out.pt"
            source.write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                fetch_checkpoint(target, source)
            self.assertFalse(target.exists())

    def test_existing_checkpoint_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "out.pt"
            target.write_bytes(b"keep")
            with self.assertRaises(ValueError):
                fetch_checkpoint(target)
            self.assertEqual(target.read_bytes(), b"keep")

    def test_verified_local_copy_can_be_installed_and_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source.pt", Path(directory) / "out.pt"
            source.write_bytes(b"fixture")
            with patch("experiments.fetch_checkpoint.CHECKPOINT_SHA256", sha256(source)):
                self.assertEqual(fetch_checkpoint(target, source), target)
                self.assertEqual(fetch_checkpoint(target), target)

    def test_committed_training_split_has_expected_sizes(self):
        split = load_training_split(RESULTS_TABLES / "herlev_group_split.json")
        self.assertEqual([len(split[k]) for k in ("unet_train", "unet_val", "calibration", "test")], [495, 55, 183, 184])

    def test_training_rejects_test_ids_in_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(json.dumps({"unet_train": ["x"], "unet_val": ["y"], "calibration": ["z"], "test": ["x"]}))
            with self.assertRaises(ValueError):
                load_training_split(path)
