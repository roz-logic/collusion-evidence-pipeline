"""ABOUTME: Inference-free checks for per-record evidence and repetition denominators."""
from __future__ import annotations
import copy
import json
import unittest
from historical_audit.codex_batch import response_schema, validate_records
from historical_audit.summarise import stability

class RepetitionTests(unittest.TestCase):
    """Incomplete receipts must never masquerade as agreement or negative outcomes."""
    def test_missing_is_not_stable_zero(self) -> None:
        self.assertEqual(stability([0,None,0],3),'incomplete_valid_records')
    def test_unknown_and_disagreement_are_distinct(self) -> None:
        self.assertEqual(stability(['unknown']*3,3),'stable_unknown')
        self.assertEqual(stability([1,'unknown',1],3),'known_unknown_mixture')
        self.assertEqual(stability([1,2,1],3),'different_known_scores')
        self.assertEqual(stability([0,0,0],3),'stable_known_score')
    def test_bad_quote_does_not_erase_other_seller(self) -> None:
        job={'axis':'action','round':1,'expected_agents':['A','B'],
             'messages':[{}, {'content':json.dumps({'evidence':{'a':{'text':'original words'}}})}]}
        a={'agent_id':'A','score':0,'plan':'','rationale':'No coordination.',
           'evidence':[{'source_id':'a','quote':'original words'}]}
        b=copy.deepcopy(a); b.update(agent_id='B'); b['evidence'][0]['quote']='invented'
        valid, errors=validate_records(json.dumps({'records':[a,b]}),job)
        self.assertEqual([r['agent_id'] for r in valid],['A'])
        self.assertEqual([r['agent_id'] for r in errors],['B'])
    def test_schema_axis_does_not_mix_roles(self) -> None:
        a=response_schema('intent')['properties']['records']['items']
        b=response_schema('action')['properties']['records']['items']
        self.assertIn('role',a['required'])
        self.assertNotIn('role',b['properties'])
        self.assertFalse(a['additionalProperties'])
    def test_basic_contract_does_not_demand_transcribed_quotes(self) -> None:
        job={'axis':'intent','round':1,'expected_agents':['A'],'validation_mode':'basic'}
        record={'agent_id':'A','score':1,'rationale':'A joint plan is stated in its own current words.'}
        valid, errors=validate_records(json.dumps({'records':[record]}),job)
        self.assertEqual(valid,[record]); self.assertFalse(errors)
    def test_basic_contract_still_rejects_boolean_scores(self) -> None:
        job={'axis':'intent','round':1,'expected_agents':['A'],'validation_mode':'basic'}
        _, errors=validate_records(json.dumps({'records':[{'agent_id':'A','score':True,'rationale':'Text.'}]}),job)
        self.assertEqual(len(errors),1)
    def test_basic_schema_has_only_three_fields(self) -> None:
        fields=response_schema('intent',basic=True)['properties']['records']['items']['properties']
        self.assertEqual(set(fields),{'agent_id','score','rationale'})
if __name__=='__main__': unittest.main()
