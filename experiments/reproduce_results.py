"""Recompute and verify frozen-row results without images or PyTorch.

Run from the repository root: python -m experiments.reproduce_results
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from experiments.exp2_strict_cellwise_coverage import run_strict_cellwise_analysis
from experiments.exp5_cx22_joint_external import run_cx22_joint_external
from src.utils.config import ROOT_DIR, RESULTS_TABLES


def verify_frozen_inputs() -> dict:
    manifest = json.loads((RESULTS_TABLES / "frozen_analysis_inputs.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = ROOT_DIR / item["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen input: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != item["sha256"]:
            raise ValueError(f"Frozen input hash mismatch: {item['path']}")
    return manifest


def _verify_csv(actual: Path, reference: Path, keys: tuple[str, ...]) -> int:
    with actual.open(encoding="utf-8", newline="") as handle:
        observed = list(csv.DictReader(handle))
    with reference.open(encoding="utf-8", newline="") as handle:
        expected = list(csv.DictReader(handle))
    indexed = {tuple(row[key] for key in keys): row for row in observed}
    if len(indexed) != len(observed) or len(observed) != len(expected):
        raise AssertionError(f"Row count or duplicate key mismatch: {actual.name}")
    for row in expected:
        candidate = indexed[tuple(row[key] for key in keys)]
        for name, value in row.items():
            if name.startswith(("joint_mean_", "joint_median_", "joint_full_domain_")) or name == "n_joint_interval_rows":
                continue  # Efficiency is a new diagnostic, not a historical baseline.
            try:
                a, b = float(candidate[name]), float(value)
                equal = (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
            except ValueError:
                equal = candidate[name] == value
            if not equal:
                raise AssertionError(f"Historical result mismatch: {actual.name}: {name}: {candidate[name]} != {value}")
    return len(observed)


def reproduce(output_dir: Path) -> dict:
    output_dir = Path(output_dir).resolve()
    if output_dir == RESULTS_TABLES.resolve():
        raise ValueError("Choose a separate output directory; frozen reference tables must not be overwritten.")
    manifest = verify_frozen_inputs()
    strict = run_strict_cellwise_analysis(output_dir=output_dir)
    external = run_cx22_joint_external(
        strict_repeats_path=output_dir / "exp2_strict_cellwise_repeats.csv", output_dir=output_dir,
    )
    checked_strict = _verify_csv(output_dir / "exp2_strict_cellwise_repeats.csv",
                                RESULTS_TABLES / "exp2_strict_cellwise_repeats.csv", ("seed", "analysis", "measurement"))
    checked_external = _verify_csv(output_dir / "exp5_cx22_joint_external_repeats.csv",
                                  RESULTS_TABLES / "exp5_cx22_joint_external_repeats.csv", ("seed", "partition"))
    report = {
        "input_hashes_verified": len(manifest["files"]),
        "historical_coverage_and_thresholds_match": True,
        "strict_rows_checked": checked_strict,
        "external_rows_checked": checked_external,
        "scope": "Frozen-row recalculation; no retraining, new inference, or untouched validation.",
        "strict_joint": strict["joint_summary"],
        "external_joint": external["summary_by_partition"],
    }
    (output_dir / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    efficiency = {
        "scope": "Post-hoc interval-efficiency diagnostics; thresholds and coverage are unchanged.",
        "width_convention": "Intervals are intersected with measurement support. Invalid predictions are excluded from widths, not assigned zero width.",
        "baseline_circularity_width": 1.0,
        "baseline_circularity_coverage_on_valid_references": 1.0,
        "baseline_note": "The full [0, 1] support is a trivial circularity-only baseline, not a finite-width baseline for unbounded N/C or joint coverage.",
        "herlev": strict["joint_summary"],
        "cx22": external["summary_by_partition"],
    }
    (output_dir / "joint_interval_efficiency_summary.json").write_text(json.dumps(efficiency, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "results" / "reproduced")
    args = parser.parse_args()
    result = reproduce(args.output_dir)
    print(json.dumps({key: result[key] for key in (
        "historical_coverage_and_thresholds_match", "input_hashes_verified", "strict_rows_checked", "external_rows_checked", "scope"
    )}, indent=2))


if __name__ == "__main__":
    main()
