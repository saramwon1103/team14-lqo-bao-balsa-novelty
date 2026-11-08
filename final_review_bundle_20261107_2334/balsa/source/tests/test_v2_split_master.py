import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from split_master_adapter import EXPECTED, load_and_validate
from balsa.v2_split_safety import assert_train_test_membership, query_stems
from export_common_results import (EVAL_FIELDS, PG_FIELDS, SNAPSHOT_FIELDS,
    CHECKPOINT_FIELDS, TRACE_FIELDS, baseline_status_v2, extract_hint_sql,
    validate_exported_run, validate_novelty_join)
from train_split import has_uncommitted_code_changes


class SplitMasterV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=REPO/'analysis/data/split_master.csv'
        cls.sql=ROOT/'queries/join-order-benchmark'
        cls.novelty=REPO/'analysis/data/novelty.csv'
        cls.locked=load_and_validate(cls.source,cls.sql,'5c9d6403089289a3188aa05ea75ef33c62b50177')

    def test_all_three_source_memberships_and_hashes(self):
        self.assertEqual(self.locked['source_sha256'],
            '804cc4ff7d764173670b65d7b0655037b398356171adcb9555df73e3fa3f8880')
        for split,split_name in [('LOO','LOO-1'),('Random','Random-1'),('BaseQuery','BaseQuery-1')]:
            train_count,test_count,_=EXPECTED[split]
            item=self.locked['splits'][split_name]
            m=item['manifest']; train=set(m['train_queries']); test=set(m['test_queries'])
            self.assertEqual((len(train),len(test)),(train_count,test_count))
            self.assertFalse(train&test)
            self.assertEqual(len(train|test),113)
            self.assertEqual(m['source']['path'],'analysis/data/split_master.csv')
            self.assertEqual(m['source']['source_commit'],'5c9d6403089289a3188aa05ea75ef33c62b50177')

    def test_duplicate_source_row_is_rejected(self):
        text=self.source.read_text(encoding='utf-8-sig')
        lines=text.splitlines()
        with tempfile.TemporaryDirectory() as d:
            copy=Path(d)/'split.csv'; copy.write_text('\n'.join(lines+[lines[1]])+'\n')
            with self.assertRaisesRegex(ValueError,'339 membership rows|duplicate'):
                load_and_validate(copy,self.sql,'fixture')

    def test_simulator_must_use_exact_train_and_test_membership(self):
        manifest=self.locked['splits']['Random-1']['manifest']
        assert_train_test_membership(query_stems(manifest['train_queries']),query_stems(manifest['test_queries']),
            query_stems(manifest['train_queries']),query_stems(manifest['test_queries']))
        with self.assertRaisesRegex(ValueError,'training query membership'):
            assert_train_test_membership(query_stems(manifest['train_queries']),query_stems(manifest['test_queries']),
                query_stems(manifest['train_queries']+manifest['test_queries']),set())

    def test_novelty_join_is_one_to_one_for_all_74_official_test_rows(self):
        with (self.novelty).open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),74)
        self.assertEqual(len({(r['query_id'],r['family'],r['split_type'],r['split_id']) for r in rows}),74)
        for protocol, split_type in [('LOO-1','LOO'),('Random-1','Random'),('BaseQuery-1','BaseQuery')]:
            membership=self.locked['splits'][protocol]['snapshot']
            test=[r for r in membership if r['role']=='test']
            observations=[dict(query_id=r['query_id'],family=r['family'],split_type=split_type,split_id='1') for r in test]
            audit=validate_novelty_join(self.novelty,split_type,1,observations)
            self.assertEqual(audit['merge_cardinality'],'one_to_one')
            self.assertEqual(audit['matched_rows'],len(test))

    def test_common_schema_and_exact_coverage_validation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); split=self.locked['splits']['LOO-1']; test=[r for r in split['snapshot'] if r['role']=='test']
            self._write(root/'split_snapshot.csv',SNAPSHOT_FIELDS,[dict(model='Balsa',split_type='LOO',split_id='1',
                query_id=r['query_id'],family=r['family'],role=r['role'],sql_file=r['sql_file']) for r in split['snapshot']])
            eval_rows=[dict(run_id='r',model='Balsa',split_type='LOO',split_id='1',query_id=r['query_id'],family=r['family'],checkpoint_id='c',planning_ms='2',inference_ms='3',execution_ms='5',end_to_end_ms='10',status='OK',timeout_ms='30000',chosen_plan_id='p',error='') for r in test]
            pg_rows=[dict(split_type='LOO',split_id='1',query_id=r['query_id'],family=r['family'],planning_ms='1',execution_ms='5',end_to_end_ms='6',status='OK',timeout_ms='30000') for r in test]
            self._write(root/'evaluation_results.csv',EVAL_FIELDS,eval_rows)
            self._write(root/'postgres_results.csv',PG_FIELDS,pg_rows)
            (root/'artifacts').mkdir();(root/'artifacts/evaluation.pt').write_bytes(b'fixture')
            self._write(root/'checkpoint_manifest.csv',CHECKPOINT_FIELDS,[dict(run_id='r',model='Balsa',split_type='LOO',split_id='1',checkpoint_id='c',step='1',metric_name='',metric_value='',path='artifacts/evaluation.pt',is_final='true',status='OK')])
            self._write(root/'training_trace.csv',TRACE_FIELDS,[]);(root/'raw_training.log').write_text('fixture')
            result=validate_exported_run(root,'LOO')
            self.assertEqual(result['evaluation_rows'],33)
            eval_rows.pop();self._write(root/'evaluation_results.csv',EVAL_FIELDS,eval_rows)
            with self.assertRaisesRegex(ValueError,'one observation per test'):
                validate_exported_run(root,'LOO')

    def test_multiline_hint_hash_uses_complete_hint_and_run_outputs_are_not_source_changes(self):
        hint='/*+ HashJoin(a b)\n Leading((a b)) */'
        self.assertEqual(extract_hint_sql(hint+'\nSELECT 1;'),hint)
        self.assertEqual(extract_hint_sql('-- hint\nSELECT 1;'),'-- hint')
        self.assertEqual(baseline_status_v2(''),'NOT_RUN')
        self.assertEqual(baseline_status_v2(None),'NOT_RUN')
        self.assertEqual(baseline_status_v2('COMPLETED'),'OK')
        self.assertFalse(has_uncommitted_code_changes(['?? balsa/runs/LOO-1/run/summary.csv']))
        self.assertTrue(has_uncommitted_code_changes(['?? balsa/source/new.py']))
        self.assertTrue(has_uncommitted_code_changes([' M balsa/source/train.py']))

    @staticmethod
    def _write(path,fields,rows):
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


if __name__=='__main__': unittest.main()
