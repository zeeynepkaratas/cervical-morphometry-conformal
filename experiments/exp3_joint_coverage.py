"""Experiment 3: strict cell-wise joint split-conformal coverage.

This experiment is joint split conformal, not joint CQR. It delegates to the
strict one-variant-per-cell protocol and defines success as covering both
morphometric measurements for the same original cell.
"""

from __future__ import annotations

import json

from experiments.exp2_strict_cellwise_coverage import run_strict_cellwise_analysis


def run_experiment_3() -> dict:
    """Run strict joint and marginal cell-level coverage re-analysis."""
    return run_strict_cellwise_analysis()


if __name__ == "__main__":
    print(json.dumps(run_experiment_3(), indent=2))
