"""ABOUTME: Verify frozen-split calibration cannot depend on held-out scores."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]

class RealFitContractTests(unittest.TestCase):
    def test_heldout_scores_do_not_change_any_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);rows=[];train=[];test=[]
            for kind in ('honest','common_incentives','talk_only','coordinated_action'):
                for n in range(4):
                    run=f'{kind}-{n}'; (train if n<2 else test).append(run)
                    rows.append({'run_id':run,'case_type':kind,'manifest_colluders':'[]',
                                 'intent_score':int(kind in ('talk_only','coordinated_action')),
                                 'action_score':int(kind=='coordinated_action'),
                                 'ordinary_score':int(kind in ('talk_only','coordinated_action'))})
            split=folder/'split.json';split.write_text(json.dumps({'calibration_run_ids':train,'test_run_ids':test}))
            def fit(name: str, data: list[dict]) -> dict:
                path=folder/(name+'.csv');out=folder/(name+'.json')
                with path.open('w',newline='') as h:
                    writer=csv.DictWriter(h,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
                result=subprocess.run([sys.executable,str(ROOT/'fit_rule.py'),'--input',str(path),'--out',str(out),
                    '--talk-col','intent_score','--behaviour-col','action_score','--ordinary-col','ordinary_score',
                    '--positive-case-type','coordinated_action','--threshold-policy','empirical','--split-file',str(split)],
                    cwd=ROOT,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr+result.stdout)
                return json.loads(out.read_text())
            before=fit('before',rows)
            changed=[{**r,**({'intent_score':-999,'action_score':999,'ordinary_score':999} if r['run_id'] in test else {})} for r in rows]
            after=fit('after',changed)
            for key in before:
                if 'threshold' in key:self.assertEqual(before[key],after[key],key)
            self.assertEqual(before['test_run_ids'],after['test_run_ids'])
            self.assertNotEqual(before['input_file_hash'],after['input_file_hash'])
            self.assertGreater(before['talk_threshold'],0)
            self.assertLessEqual(before['talk_threshold'],1)

if __name__=='__main__':unittest.main()
