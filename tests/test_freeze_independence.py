
# tests/test_freeze_independence.py
#
# Leak test: proves the thresholds fit_rule.py freezes depend only on
# train-run scores, not on anything in the test rows. Reruns
# fit_rule.py twice, once on the original fixture, once on a copy where only
# the test rows' two scores have been scrambled, and asserts the
# threshold values are unchanged. If they weren't, that would mean
# test data is leaking into calibration despite the code being
# structured to prevent that.
#
# requires fit_rule.py to accept --input and --out via argparse.

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent.parent
FIT_SCRIPT = HERE / "fit_rule.py"
ORIGINAL_CSV = HERE / "fake_scores.csv"


def run_fit(input_csv, out_json):
    result = subprocess.run(
        [
            sys.executable,
            str(FIT_SCRIPT),
            "--input",
            str(input_csv),
            "--out",
            str(out_json),
        ],
        cwd=HERE,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"fit_rule.py failed:\n"
        f"{result.stdout}\n"
        f"{result.stderr}"
    )


def test_freeze_independent_of_test_rows(tmp_path):
    out_original = tmp_path / "frozen_original.json"
    run_fit(ORIGINAL_CSV, out_original)

    with open(out_original, encoding="utf-8") as f:
        frozen_original = json.load(f)

    test_run_ids = set(frozen_original["test_run_ids"])

    df = pd.read_csv(ORIGINAL_CSV)
    is_test = df["run_id"].isin(test_run_ids)

    rng = np.random.default_rng(999)

    for col in ["cross_run_mi", "welfare_shift"]:
        test_vals = df.loc[is_test, col].to_numpy().copy()
        rng.shuffle(test_vals)
        df.loc[is_test, col] = test_vals

    scrambled_csv = tmp_path / "fake_scores_scrambled.csv"
    df.to_csv(scrambled_csv, index=False)

    out_scrambled = tmp_path / "frozen_scrambled.json"
    run_fit(scrambled_csv, out_scrambled)

    with open(out_scrambled, encoding="utf-8") as f:
        frozen_scrambled = json.load(f)

    # Confirm that both runs use exactly the same test-run IDs.
    assert frozen_original["test_run_ids"] == frozen_scrambled["test_run_ids"]

    # Confirm that all six thresholds are independent of the
    # scrambled test-row scores.
    for key in [
        "talk_threshold",
        "behaviour_threshold",
        "talk_baseline_a_threshold",
        "behaviour_baseline_a_threshold",
        "talk_baseline_b_threshold",
        "behaviour_baseline_b_threshold",
    ]:
        assert frozen_original[key] == frozen_scrambled[key], (
            f"{key} changed after scrambling test rows only: "
            f"{frozen_original[key]} vs {frozen_scrambled[key]}, "
            f"this means test data is leaking into calibration"
        )
