# evaluate_rule.py
# ABOUTME: Apply a committed frozen rule to hash-matched held-out data.
# Reports primary and single-axis flags with Wilson binomial intervals.
# Calibration is confined to fit_rule.py.

# Python version: 3.14.4
# Required packages: pandas, numpy

import pandas as pd
import numpy as np
import json
import hashlib
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="fake_scores.csv")
parser.add_argument("--rule", default="frozen_rule.json")
parser.add_argument("--outdir", default="results")
args = parser.parse_args()

INPUT_PATH = args.input
FROZEN_RULE_PATH = args.rule
OUTDIR = args.outdir

os.makedirs(OUTDIR, exist_ok=True)

def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

with open(FROZEN_RULE_PATH) as f:
    frozen = json.load(f)

TALK_SCORE_COL = frozen["talk_score_col"]
BEHAVIOUR_SCORE_COL = frozen["behaviour_score_col"]
RUN_ID_COL = frozen["run_id_col"]
ASSIGNED_COL = frozen["assigned_col"]
CASE_TYPE_COL = frozen["case_type_col"]
SEED = frozen["seed"]
POSITIVE_CASE_TYPE = frozen.get("positive_case_type", "harmful")

print(f"Loaded frozen rule from {FROZEN_RULE_PATH}")
print(f"talk_threshold = {frozen['talk_threshold']:.4f}, behaviour_threshold = {frozen['behaviour_threshold']:.4f}")
print("Reminder: check git log for this file's commit timestamp, relative")
print("to when you're running this evaluate script, as evidence the freeze")
print("wasn't adjusted after peeking at test.")

current_hash = file_hash(INPUT_PATH)
assert current_hash == frozen["input_file_hash"], (
    "input file does not match the one fit_rule.py calibrated on, "
    "someone swapped the data between fit and evaluate, stop"
)
print("\u2713 input file hash matches what fit_rule.py calibrated on")

# =============================================================
# PART 1: LOAD, RECONSTRUCT TEST FROM FROZEN RUN IDS ONLY
# =============================================================

df = pd.read_csv(INPUT_PATH)
for col in [RUN_ID_COL, TALK_SCORE_COL, BEHAVIOUR_SCORE_COL, ASSIGNED_COL, CASE_TYPE_COL]:
    assert col in df.columns, f"expected column '{col}' not found"

df["assigned"] = df[ASSIGNED_COL].apply(
    lambda x: (int(x.strip() != "[]") if isinstance(x, str) and x.strip() else np.nan)
)

test_run_ids = set(frozen["test_run_ids"])
test = df[df[RUN_ID_COL].astype(str).isin(test_run_ids)].copy()
assert len(test) > 0, (
    "no rows matched test_run_ids from frozen_rule.json, check INPUT_PATH "
    "is the same data fit_rule.py used, and that run_id types match as strings"
)
print(f"Test rows: {len(test)}, test runs: {test[RUN_ID_COL].nunique()} "
      f"(expected {len(test_run_ids)})")

# =============================================================
# PART 2: ORIENT, CLASSIFY, USING THE FROZEN RULE ONLY
# =============================================================

def oriented(series, higher_is_suspicious):
    return series if higher_is_suspicious else -series

def classify(data, rule):
    talk_s = oriented(data[TALK_SCORE_COL], rule["talk_higher_is_suspicious"])
    behav_s = oriented(data[BEHAVIOUR_SCORE_COL], rule["behaviour_higher_is_suspicious"])
    talk = (talk_s >= rule["talk_threshold"]).astype(int)
    behaviour = (behav_s >= rule["behaviour_threshold"]).astype(int)
    case_label = np.select(
        [
            (talk == 1) & (behaviour == 1),
            (talk == 0) & (behaviour == 1),
            (talk == 1) & (behaviour == 0),
        ],
        ["a_flag", "b_parallel_behaviour", "c_cheap_talk"],
        default="d_competition",
    )
    return talk, behaviour, case_label

test["talk"], test["behaviour"], test["case_label"] = classify(test, frozen)
test["flag"] = (test["case_label"] == "a_flag").astype(int)

print("\nCase label counts on test:")
print(test["case_label"].value_counts())

# =============================================================
# PART 3: SANITY CROSS-CHECK AGAINST ASSIGNED (NOT THE EVALUATION)
# =============================================================
# descriptive only, assigned can't distinguish case one from case
# three, so it's not the evaluation, case_type is, this is a glance

crosscheck = test.groupby("case_label")["assigned"].mean()
print("\nMean assigned=1 rate by case_label (sanity check only, not the evaluation):")
print(crosscheck)

# =============================================================
# PART 4: EVALUATION BY CASE TYPE, PRIMARY RULE VS BOTH BASELINE VARIANTS
# =============================================================

print("\nCrosstab, case_type (rows) vs rule output (columns):")
print(pd.crosstab(test[CASE_TYPE_COL], test["case_label"]))

is_harmful = test[CASE_TYPE_COL] == POSITIVE_CASE_TYPE
detection_rate = test.loc[is_harmful, "flag"].mean() if is_harmful.any() else float("nan")
print(f"\nDetection rate on {POSITIVE_CASE_TYPE} cases: {detection_rate:.3f}")

false_flag_by_type = test.loc[~is_harmful].groupby(CASE_TYPE_COL)["flag"].mean()
print("False-flag rate by non-harmful case_type:")
print(false_flag_by_type)

def score_single_axis(data, score_col, threshold, higher_is_suspicious, label_name):
    s = oriented(data[score_col], higher_is_suspicious)
    flag = (s >= threshold).astype(int)
    is_h = data[CASE_TYPE_COL] == POSITIVE_CASE_TYPE
    print(f"\nBaseline ({label_name}), threshold {threshold:.4f}:")
    print(f"  detection rate on {POSITIVE_CASE_TYPE}: {flag[is_h].mean():.3f}" if is_h.any() else "  no positive cases in test")
    print("  false-flag rate by case_type:")
    print(flag[~is_h].groupby(data.loc[~is_h, CASE_TYPE_COL]).mean())
    return flag

print("\n--- Baseline variant A: honest-only calibration (naive detector) ---")
talk_flag_a = score_single_axis(test, TALK_SCORE_COL, frozen["talk_baseline_a_threshold"], frozen["talk_higher_is_suspicious"], "talk, variant A")
behav_flag_a = score_single_axis(test, BEHAVIOUR_SCORE_COL, frozen["behaviour_baseline_a_threshold"], frozen["behaviour_higher_is_suspicious"], "behaviour, variant A")

print("\n--- Baseline variant B: same nulls as primary rule (same label info) ---")
talk_flag_b = score_single_axis(test, TALK_SCORE_COL, frozen["talk_baseline_b_threshold"], frozen["talk_higher_is_suspicious"], "talk, variant B")
behav_flag_b = score_single_axis(test, BEHAVIOUR_SCORE_COL, frozen["behaviour_baseline_b_threshold"], frozen["behaviour_higher_is_suspicious"], "behaviour, variant B")

print("\nBaseline A uses honest-only nulls; baseline B uses the primary rule's nulls.")
print("The observed case-type counts determine each method's trade-off.")

# =============================================================
# PART 5: RATES BY CASE TYPE WITH WILSON 95% INTERVALS, SAVED TO CSV
# =============================================================
# Intervals use one row per parent run and can be wide for small groups.
# Stop if there are repeated run IDs: row-level Wilson intervals would
# otherwise incorrectly treat rows from the same run as independent.

if test[RUN_ID_COL].astype(str).duplicated().any():
    raise ValueError(
        "Wilson intervals here require exactly one row per test run. "
        "Repeated run IDs were found. Define a run-level flag before "
        "calculating these intervals."
    )


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate_table(data, flag, name):
    rows = []
    for ct, idx in data.groupby(CASE_TYPE_COL).groups.items():
        k, n = int(flag.loc[idx].sum()), len(idx)
        lo, hi = wilson(k, n)
        rows.append({
            "method": name,
            "case_type": ct,
            "flagged": k,
            "n": n,
            "rate": round(k / n, 3),
            "ci_low": round(lo, 3),
            "ci_high": round(hi, 3),
        })
    return pd.DataFrame(rows)


tables = pd.concat([
    rate_table(test, test["flag"], "primary two-signal"),
    rate_table(test, talk_flag_a, "talk only, baseline A"),
    rate_table(test, behav_flag_a, "behaviour only, baseline A"),
    rate_table(test, talk_flag_b, "talk only, baseline B"),
    rate_table(test, behav_flag_b, "behaviour only, baseline B"),
], ignore_index=True)

if frozen.get("ordinary_score_col"):
    ordinary = frozen["ordinary_score_col"]
    assert test[ordinary].notna().all(), "ordinary baseline missing on test"
    test["ordinary_calibrated_flag"] = (test[ordinary] >= frozen["ordinary_threshold"]).astype(int)
    test["ordinary_fixed_flag"] = (test[ordinary] >= frozen["ordinary_fixed_threshold"]).astype(int)
    tables = pd.concat([tables,
        rate_table(test, test["ordinary_calibrated_flag"], "ordinary direct judge, same-null calibration"),
        rate_table(test, test["ordinary_fixed_flag"], "ordinary direct judge, fixed majority")], ignore_index=True)

print(tables.to_string(index=False))

tables.to_csv(
    os.path.join(OUTDIR, "rates_by_case_type.csv"),
    index=False,
)

pd.crosstab(
    test[CASE_TYPE_COL],
    test["case_label"],
).to_csv(os.path.join(OUTDIR, "crosstab_primary.csv"))

test.to_csv(os.path.join(OUTDIR, "test_predictions.csv"), index=False)
print("\nevaluate_rule.py complete.")