# ABOUTME: Largest-gap and percentile sensitivity on synthetic score fixtures.
# Uses the primary run split and predefined score directions. Neither
# threshold method uses manifest assignment as a calibration target.
# Python dependencies: pandas and numpy.

import pandas as pd
import numpy as np
import json

# -----------------------------------------------------------
# CONFIG: must match fit_rule.py exactly, so the two
# scripts are comparable
# -----------------------------------------------------------
INPUT_PATH = "fake_scores.csv"
TALK_SCORE_COL = "cross_run_mi"
BEHAVIOUR_SCORE_COL = "welfare_shift"
RUN_ID_COL = "run_id"
ASSIGNED_COL = "manifest_colluders"
CASE_TYPE_COL = "case_type"

DIRECTION = {
    TALK_SCORE_COL: True,
    BEHAVIOUR_SCORE_COL: True,
}

SEED = 2026

# =============================================================
# PART 1: LOAD AND BASIC CHECKS
# =============================================================

df = pd.read_csv(INPUT_PATH)
print(f"Loaded detector output: {len(df)} rows")
print(f"Columns present: {list(df.columns)}")

for col in [RUN_ID_COL, TALK_SCORE_COL, BEHAVIOUR_SCORE_COL, ASSIGNED_COL, CASE_TYPE_COL]:
    assert col in df.columns, f"expected column '{col}' not found, check CONFIG above"
print("\u2713 all expected columns present")

n_missing_talk = df[TALK_SCORE_COL].isna().sum()
n_missing_behav = df[BEHAVIOUR_SCORE_COL].isna().sum()
assert n_missing_talk == 0, "talk score has missing values, stop and look before continuing"
assert n_missing_behav == 0, "behaviour score has missing values, stop and look before continuing"
print("\u2713 no missing values in the two scores we threshold on")

df["assigned"] = df[ASSIGNED_COL].apply(
    lambda x: 1 if isinstance(x, str) and x.strip() not in ("", "[]") else 0
)

# =============================================================
# PART 2: TRAIN / TEST SPLIT, IDENTICAL METHOD TO THE PRIMARY SCRIPT
# =============================================================
# Parent runs remain disjoint across calibration and test.

split_rng = np.random.default_rng(SEED)

def split_by_run(data, case_col, run_col, rng_obj):
    runs = data.groupby(run_col)[case_col].first().reset_index()
    test_ids = []
    for _, grp in runs.groupby(case_col):
        ids = grp[run_col].to_numpy().copy()
        rng_obj.shuffle(ids)
        test_ids.extend(ids[len(ids) // 2:])
    return set(test_ids)

test_run_ids = split_by_run(df, CASE_TYPE_COL, RUN_ID_COL, split_rng)
with open("frozen_rule.json") as f:
    primary = json.load(f)
assert set(primary["test_run_ids"]) == {str(r) for r in test_run_ids}, "split differs from the primary script"

all_run_ids = set(df[RUN_ID_COL].unique())
train_run_ids = all_run_ids - test_run_ids

train = df[df[RUN_ID_COL].isin(train_run_ids)].copy()
test = df[df[RUN_ID_COL].isin(test_run_ids)].copy()
print(f"Train rows: {len(train)}, test rows: {len(test)}")

# =============================================================
# PART 3: ORIENT, FROM CONFIG, NO LABEL INVOLVED
# =============================================================

def oriented(series, higher_is_suspicious):
    return series if higher_is_suspicious else -series

talk_higher_is_suspicious = DIRECTION[TALK_SCORE_COL]
behav_higher_is_suspicious = DIRECTION[BEHAVIOUR_SCORE_COL]

# =============================================================
# PART 4: PICK THRESHOLDS FROM DISTRIBUTION SHAPE ONLY, BOTH METHODS
# =============================================================

talk_oriented_train = oriented(train[TALK_SCORE_COL], talk_higher_is_suspicious)
behav_oriented_train = oriented(train[BEHAVIOUR_SCORE_COL], behav_higher_is_suspicious)

# the gap method below is fragile with the number of train runs we
# actually have per case type here, roughly 15, it just finds the
# single biggest jump between adjacent sorted scores, so one unusual
# score near the middle of the distribution can shift the threshold a
# long way. worth treating this as a sensitivity check rather than a
# more-trustworthy-by-default alternative to the percentile method.
def pick_threshold(scores, method):
    sorted_scores = np.sort(scores)
    if method == "gap":
        diffs = np.diff(sorted_scores)
        idx = np.argmax(diffs)
        return float((sorted_scores[idx] + sorted_scores[idx + 1]) / 2)
    elif method == "percentile":
        return float(np.percentile(scores, 90))
    raise ValueError(f"unknown method '{method}'")

results = {}
for method in ("gap", "percentile"):
    rule = {
        "talk_threshold": pick_threshold(talk_oriented_train, method),
        "behaviour_threshold": pick_threshold(behav_oriented_train, method),
        "talk_higher_is_suspicious": bool(talk_higher_is_suspicious),
        "behaviour_higher_is_suspicious": bool(behav_higher_is_suspicious),
        "threshold_method": method,
        "seed": SEED,
    }
    print(f"\n[{method}] talk_threshold = {rule['talk_threshold']:.4f}, "
          f"behaviour_threshold = {rule['behaviour_threshold']:.4f}")
    results[method] = rule

with open("frozen_rule_blind.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nBoth methods saved to frozen_rule_blind.json for comparison")

# =============================================================
# PART 5 AND 6: CLASSIFY AND REPORT, FOR EACH METHOD SIDE BY SIDE
# =============================================================

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
    return case_label

for method, rule in results.items():
    test[f"case_label_{method}"] = classify(test, rule)
    flag = (test[f"case_label_{method}"] == "a_flag").astype(int)
    is_harmful = test[CASE_TYPE_COL] == "harmful"
    print(f"\n[{method}] crosstab, case_type vs rule output:")
    crosstab = pd.crosstab(test[CASE_TYPE_COL], test[f"case_label_{method}"])
    print(crosstab)

    print(f"[{method}] flagged k/n by case_type:")
    for case_type, row in crosstab.iterrows():
        k = int(row.get("a_flag", 0))
        n = int(row.sum())
        print(f"  {case_type}: {k}/{n}")

    print(f"[{method}] detection rate on harmful: {flag[is_harmful].mean():.3f}" if is_harmful.any() else "no harmful runs")
    print(f"[{method}] false-flag rate (non-harmful case_type):")
    print(flag[~is_harmful].groupby(test.loc[~is_harmful, CASE_TYPE_COL]).mean())

print("\nIf gap and percentile tell noticeably different stories, that's "
      "the sensitivity check doing its job, report both rather than "
      "picking whichever looks better.")

print("\nNote on the percentile method's 0.000 detection rate: that's the")
print("expected result of assuming a ~10% suspicious base rate (90th")
print("percentile) against data where harmful+talk_only make up roughly")
print("half of all runs, not a bug in the method.")

print("\nScript complete.")