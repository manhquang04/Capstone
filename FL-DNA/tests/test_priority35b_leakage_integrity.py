"""Synthetic evidence integrity failures; no scientific outputs are read."""
import json
import tempfile
import unittest
from pathlib import Path
from experiments import priority35b_leakage_integrity as audit


class Tests(unittest.TestCase):
    def test_finite_and_path_fail_closed(self):
        for value in (float('nan'), {'x': [float('inf')]}):
            with self.assertRaises(ValueError): audit.finite_tree(value)
        with tempfile.TemporaryDirectory() as tmp:
            for name in ('../x', '/x'):
                with self.assertRaises(ValueError): audit.child(tmp, name)
            self.assertEqual(audit.child(tmp, 'x'), Path(tmp).resolve() / 'x')

    def test_result_hash_identity_and_missing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); attempt = p / 'attempt_1'; attempt.mkdir()
            for name in audit.EXPECTED_FILES: (attempt / name).write_bytes(b'synthetic')
            config = dict(dataset='fixture', cell='ratio_batch1', index=0, arm='unprotected')
            result = dict(status='COMPLETED', config=config, freeze_sha256='fixture', threads=1,
                          inter_threads=1, source_ids=[7], attempt='attempt_1',
                          files={n:audit.sha(attempt/n) for n in audit.EXPECTED_FILES},
                          score=dict(feature_accuracy=.5, standardized_mse=1., bounded_mse=.5, assignment=[0]))
            target = dict(source_ids=[7])
            def save(): (p / 'result.json').write_text(json.dumps(result))
            save(); audit.verify_result(p, config, 'fixture', target)
            (attempt / 'server_receipt.pt').write_bytes(b'drift')
            with self.assertRaises(ValueError): audit.verify_result(p, config, 'fixture', target)
            (attempt / 'server_receipt.pt').write_bytes(b'synthetic')
            result['files'].pop('private_truth.pt'); save()
            with self.assertRaises(ValueError): audit.verify_result(p, config, 'fixture', target)

    def test_target_firewall(self):
        targets=[]; exclusions={d:[] for d in ('a','b','c')}
        for d, cells in [('a',['ratio_batch1','tableak_batch1','tableak_batch2']),
                         ('b',['ratio_batch1','tableak_batch1']),
                         ('c',['ratio_batch1','tableak_batch1','tableak_batch2'])]:
            source=0
            for cell in cells:
                for i in range(39):
                    size=2 if cell.endswith('batch2') else 1
                    ids=list(range(source,source+size));source+=size
                    targets.append(dict(dataset=d,cell=cell,index=i,source_ids=ids,indices=ids))
        self.assertEqual(len(audit.verify_targets(targets,exclusions)),312)
        exclusions['a']=[0]
        with self.assertRaises(ValueError): audit.verify_targets(targets,exclusions)


if __name__ == '__main__': unittest.main()
