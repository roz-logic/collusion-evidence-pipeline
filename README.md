# Auditing the Evidence Chain for LLM-Agent Collusion

Apart AI Collusion Research Sprint · Track 2.3 · October 2026

This study examines whether recorded communication and actions support an inference of collusion. It connects a source-validity audit, process-level assessment of historical auctions, and a frozen-rule evaluation. The analysis separates **expressed willingness, selected participation, environment execution and economic outcomes**.

## Study and findings

| Component | Evidence | Main observation |
| --- | --- | --- |
| Source-validity audit | Pinned starter code and historical logs | Stored text and assigned roles do not establish message delivery or a negotiated agreement. |
| Complete-run process audit | 3 auctions, 90 rounds, 450 seller decisions | A communicated withdrawal plan leads to selected requests that the environment fails to execute. |
| Frozen-rule pilot | 16 additional focal cases from distinct parent runs; 8 calibration / 8 test | The audit rule flags one fewer provisional negative case than an ordinary direct judge. |

On the eight held-out focal cases:

| Method | Provisional positives detected | Provisional negatives flagged |
| --- | ---: | ---: |
| Evidence audit + two-signal rule | 2/2 | 1/6 |
| Ordinary direct judge, same-null calibration | 2/2 | 2/6 |
| Participation score alone | 2/2 | 1/6 |

References are AI-assisted source judgments fixed before scoring and not independently human-validated. The sample is small and purposively selected. Participation-only matches the combined rule; an additional benefit from the AND operator is not established. [The report](results/real_calibration/report.md) provides case definitions, intervals, errors and limitations.

## Materials

- [Manuscript](paper/main.tex) and [bibliography](paper/refs.bib).
- [Submission materials and offline Run Atlas](artifacts/submission-review.zip). Open `Run Atlas.html` after extraction.
- [Channel and execution audit](docs/channel_validity_audit.md), with pinned source-line links.
- [Complete-run process results](docs/auction-pilot-results.md).
- [Evaluation data flow](docs/auction-pipeline-integration.md) and [process protocol with exact judge prompts](docs/auction-process-evaluation.md).
- [Market and assessment diagrams](docs/auction-diagrams-en.html).
- [Real-source scores, frozen rule and test exports](results/real_calibration/).
- **[Read the submitted paper (PDF)](paper/Auditing_the_Evidence_Chain_for_LLM_Agent_Collusion.pdf)**.


## Reproduce the held-out results

The saved-score replay requires Python, pandas and NumPy. It makes no model calls and writes to a separate output directory:

```bash
uv run --with pandas --with numpy python fit_rule.py \
  --input results/real_calibration/scores_complete.csv \
  --out /tmp/replayed_rule.json \
  --talk-col intent_score --behaviour-col action_score \
  --ordinary-col ordinary_score --positive-case-type coordinated_action \
  --threshold-policy empirical \
  --split-file results/real_calibration/frozen_protocol.json

uv run --with pandas --with numpy python evaluate_rule.py \
  --input results/real_calibration/scores_complete.csv \
  --rule /tmp/replayed_rule.json --outdir /tmp/replayed_test
```

The recorded rule was [committed before evaluation](https://github.com/roz-logic/collusion-evidence-pipeline/commit/db7c217afaaf970f9d168e6321fe018fc866c0f5). Its input hash, split and execution receipts are preserved. Verification covers held-out-score independence, rejection of changed input bytes, 31 integrity tests, and reproduction of all 20 original synthetic rate-table rows.

The `historical_audit` modules provide source preparation, judge execution, score summaries and saved-rating analysis. Fresh inference uses an authenticated Codex CLI; saved results can be inspected and replayed without it. `results/fake_debug/` contains synthetic implementation checks; `frozen_rule_blind.py` evaluates gap/percentile sensitivity on those fixtures.

## Scope

The target is supported coordinated **action selection** at a focal seller-round. Successful execution and causal harm require separate evidence. Opening-proposal controls describe that moment, not the entire subsequent run. The study does not establish general detector performance, causal oversight effects or validated early warning.

## Authors and attribution

**Oz** (`roz-logic`) — statistical pipeline and integrity checks. **Shuhao Yao** (`Dexter-Yao`) — source/process audit and real-data integration. **Lucette** (`cookkatelyn01-ui`) — manuscript structure, editing and formatting.

AI assistance is disclosed in the [manuscript](paper/main.tex). The code retains the original [MIT licence](LICENSE); [source attribution](THIRD_PARTY_NOTICES.md) identifies third-party material.
