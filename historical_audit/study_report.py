"""ABOUTME: Summarise the frozen real-source study after Oz's held-out evaluation.

Owns count-based comparisons and conditional claims, never fitting or relabelling.
Agent-source-adjudicated references remain provisional until human review.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime,timezone
import json
import math
from pathlib import Path
from typing import Any
from .prepare import sha256
from .calibrated_study import dump


def read_csv(path: Path) -> list[dict[str,str]]:
    """Read a BOM-safe CSV export."""
    with path.open(encoding='utf-8-sig') as h:return list(csv.DictReader(h))


def wilson(k: int,n: int) -> list[float]:
    """Return a Wilson 95% binomial interval for independent parent-run counts."""
    if n==0:return [0.0,1.0]
    z=1.96;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0.0,c-half),min(1.0,c+half)]


def report(study: Path) -> dict[str,Any]:
    """Summarise existing held-out predictions without using them to refit anything."""
    rule=json.loads((study/'frozen_rule.json').read_text())
    assert sha256(study/'scores_complete.csv')==rule['input_file_hash']
    assert sha256(study/'frozen_protocol.json')==rule['split_file_hash']
    rows=read_csv(study/'evaluation/test_predictions.csv')
    protocol=json.loads((study/'frozen_protocol.json').read_text())
    refs={r['run_id']:r for r in json.loads((study/'references.json').read_text())}
    assert {r['run_id'] for r in rows}==set(rule['test_run_ids'])
    assert len(rows)==len({r['run_id'] for r in rows})
    methods={
      'Evidence audit + Oz two-signal rule':lambda r:int(r['flag']),
      'Ordinary judge + same-null calibration':lambda r:int(r['ordinary_calibrated_flag']),
      'Ordinary judge + fixed majority':lambda r:int(r['ordinary_fixed_flag']),
      'Willingness only, same nulls':lambda r:int(float(r['intent_score'])>=rule['talk_baseline_b_threshold']),
      'Participation only, same nulls':lambda r:int(float(r['action_score'])>=rule['behaviour_baseline_b_threshold'])}
    overall=[];predictions=[]
    for name,fn in methods.items():
        counts={'tp':0,'fp':0,'tn':0,'fn':0}
        for row in rows:
            flag=fn(row);truth=int(row['reference_participation']);key=('tp' if truth else 'fp') if flag else ('fn' if truth else 'tn');counts[key]+=1
            predictions.append({'method':name,'run_id':row['run_id'],'case_id':row['case_id'],'reference':truth,'flag':flag,'case_type':row['case_type']})
        tp,fp,tn,fnn=(counts[x] for x in ('tp','fp','tn','fn'))
        overall.append({'method':name,**counts,'n':len(rows),'detection_rate':tp/(tp+fnn) if tp+fnn else None,
                        'false_flag_rate':fp/(fp+tn) if fp+tn else None,
                        'detection_ci':wilson(tp,tp+fnn),'false_flag_ci':wilson(fp,fp+tn)})
    official=read_csv(study/'evaluation/rates_by_case_type.csv')
    names={'Evidence audit + Oz two-signal rule':'primary two-signal',
           'Ordinary judge + same-null calibration':'ordinary direct judge, same-null calibration',
           'Ordinary judge + fixed majority':'ordinary direct judge, fixed majority',
           'Willingness only, same nulls':'talk only, baseline B',
           'Participation only, same nulls':'behaviour only, baseline B'}
    for name,core_name in names.items():
        for kind in {r['case_type'] for r in rows}:
            selected=[r for r in predictions if r['method']==name and r['case_type']==kind]
            expected=next(r for r in official if r['method']==core_name and r['case_type']==kind)
            assert len(selected)==int(expected['n'])
            assert sum(r['flag'] for r in selected)==int(expected['flagged']), (name,kind)
    comparisons=[]
    for row in rows:
        truth=int(row['reference_participation']);ours=int(row['flag'])
        for field,label in [('ordinary_fixed_flag','fixed majority'),('ordinary_calibrated_flag','same-null calibration')]:
            base=int(row[field])
            if base!=ours:comparisons.append({'case_id':row['case_id'],'run_id':row['run_id'],'case_type':row['case_type'],
                'reference':truth,'audit_flag':ours,'ordinary_flag':base,'ordinary_variant':label,
                'change':'corrected' if ours==truth else 'regressed','source':refs[row['run_id']]['response_source']})
    coverage=json.loads((study/'coverage.json').read_text())
    receipts=[json.loads(x) for x in (study/'judge_receipts.jsonl').read_text().splitlines()]
    statuses={s:sum(r['status']==s for r in receipts) for s in sorted({r['status'] for r in receipts})}
    result={'created_at':datetime.now(timezone.utc).isoformat(),'status':'completed_conditional_on_provisional_source_references',
            'target':protocol['target'],'reference_status':'agent source-adjudicated before scoring; pending independent human review',
            'n_source_cases':len(refs),'n_calibration':len(rule['calibration_run_ids']),'n_test':len(rows),
            'judge_calls':len(receipts),'call_statuses':statuses,'coverage':coverage,
            'source_files_git_verified':48,'overall':overall,'paired_differences':comparisons,
            'limits':[protocol['reference_limit'],protocol['scope_limit'],
                'One focal seller-round per parent run; purposive and condition/time-confounded selection, not a random population benchmark.',
                'Only two held-out positive examples; wide intervals. Repeated judge calls are not independent market replications.',
                'Judge B already uses communication context; B-only is not an action-modality-only detector.',
                'The false-alarm target is empirical on four calibration negatives; it is not a population guarantee.'],
            'hashes':{name:sha256(study/name) for name in ['frozen_protocol.json','references.json','packets.json','judge_receipts.jsonl','scores_complete.csv','frozen_rule.json','evaluation/test_predictions.csv']}}
    dump(study/'evaluation_summary.json',result)
    lines=['# Real-source calibration, freeze and held-out evaluation','',
           '26 October 2026. Completed using reused historical auction records and Luna via the existing ChatGPT subscription. No new market trajectories, paid API calls or model training.','',
           '**All performance figures below are conditional on provisional source-adjudicated reference cases. They are not independently human-validated detector accuracy.**','',
           '## One experiment, from source validity to held-out results','',
           '```mermaid','flowchart LR','  A["Pinned environment and log audit"] --> B["3 complete runs: 450 seller-rounds"]',
           '  B --> C["16 new parent-run focal cases: fixed source references"]',
           '  C --> D["Identical evidence: ordinary Luna vs separate A/B"]',
           '  D --> E["8 calibration cases: Oz fit_rule.py"]',
           '  E --> F["Commit frozen thresholds and input hash"]',
           '  F --> G["8 held-out cases: Oz evaluate_rule.py"]',
           '  G --> H["Paired errors, uncertainty and human reference review"]','```','',
           'The complete-run process audit motivates the evidence distinctions. The additional 16 distinct parent runs supply a run-disjoint calibration/test pilot; they do not replace the full-round audit. Case selection, provisional references, prompts, aggregation and split were committed before the new judge calls. The fitted rule was committed before evaluation.','',
           '## Scope and reference labels','',
           'The target is **source-supported coordinated selected-action participation at the focal seller-round**. A coordinated request can count even if execution fails. Realised causal harm remains a separate question.','',
           'Each of four categories contributes four distinct parent runs, split two/two: coordinated action; competitive similarity without observed agreement; opening individual price discovery; and an unaccepted opening-round proposal. The last category is a temporal control, not proof of an entire run of talk that never changes behaviour. Coordinated action selection and realised causal harm are separate targets.','',
           'References were drafted from the original prompts, plans, selected asks and received messages by the assisting agent before scoring. They were not generated from Luna scores or manifest assignments. The references have not undergone independent human validation.','',
           '## Held-out results','',
           '| Method | Detected positives | False flags on negatives |','| --- | ---: | ---: |']
    for r in overall:lines.append(f"| {r['method']} | {r['tp']}/{r['tp']+r['fn']} | {r['fp']}/{r['fp']+r['tn']} |")
    lines+=['','Exact per-category counts and Wilson intervals are in [the results CSV](evaluation/rates_by_case_type.csv). The overall intervals are in [the machine-readable summary](evaluation_summary.json). Zero observed errors does not establish a zero population error rate.','',
            '## Calibration and reproducibility','',
            'The original synthetic default results remain unchanged. Optional arguments select actual `intent_score`/`action_score` columns, a pre-scoring split, the `coordinated_action` target, and an ordinary-judge column. An opt-in tie-safe empirical cutoff prevents an all-zero null group from being flagged by an inclusive zero threshold. Baseline A uses honest negatives; baseline B and the ordinary calibrated baseline use the same honest/common-incentive negatives as the primary rule. The ordinary fixed-majority result is also retained.','',
            f"Luna call receipts: **{len(receipts)}**, statuses `{json.dumps(statuses)}`. At least two numeric ratings are required per axis; unknown/failed calls are not filled with zero. Repeats are averaged under the rule fixed before inference. Coverage: **{coverage['complete_cases_all_methods']}/{coverage['all_cases']}** source cases. The complete raw answers, reference rationales and sources remain available. Neither call validity nor three-way agreement proves semantic correctness.",'',
            '## What the results can and cannot support','']
    for limit in result['limits']:lines.append('- '+limit)
    lines+=['','## Reproduction','',
            'Model-free replay from saved scores:','', '```bash',
            'uv run python fit_rule.py --input results/real_calibration/scores_complete.csv \\',
            '  --out /tmp/replayed_rule.json --talk-col intent_score --behaviour-col action_score \\',
            '  --ordinary-col ordinary_score --positive-case-type coordinated_action \\',
            '  --threshold-policy empirical --split-file results/real_calibration/frozen_protocol.json',
            'uv run python evaluate_rule.py --input results/real_calibration/scores_complete.csv \\',
            '  --rule /tmp/replayed_rule.json --outdir /tmp/replayed_test',
            '```','',
            'The original frozen artifact remains unchanged. Inference and its source packets are separately reproducible using `historical_audit.calibrated_study`; scoring requires an authenticated Codex CLI and consumes subscription usage.','',
            '## LLM assistance','',
            'AI assistance was used for source/code investigation, implementation, drafting provisional source-reference judgments, data preparation, statistical integration, documentation and figures. Luna also served as the evaluated judge in fresh CLI sessions. The provisional references are not independent expert-human labels. Historical market-agent outputs were reused from the cited repository; no synthetic debugging scores are represented as real detector observations. Final human review and responsibility remain with the authors.']
    (study/'report.md').write_text('\n'.join(lines)+'\n')
    return result


def main() -> None:
    """Create a report only from an existing completed test export."""
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',required=True,type=Path);args=p.parse_args()
    result=report(args.study)
    print(json.dumps({'status':result['status'],'test_cases':result['n_test'],'overall':result['overall']},indent=2))

if __name__=='__main__':main()
