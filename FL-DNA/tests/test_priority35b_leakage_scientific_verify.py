"""Synthetic coordinator safety tests; never touches measurement results."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from experiments import priority35b_leakage_scientific_verify as verify


class Tests(unittest.TestCase):
    def test_source_seal_drift_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'artifacts/priority35b/b3';out.mkdir(parents=True)
            source=root/'fixture.py';source.write_text('synthetic')
            (out/'seal.json').write_text(json.dumps(dict(sources={'fixture.py':verify.integrity.sha(source)})))
            self.assertEqual(verify.verify_additive_sources(root,['seal.json'])['source_count'],1)
            source.write_text('drift')
            with self.assertRaises(ValueError):verify.verify_additive_sources(root,['seal.json'])

    def test_no_live_before_scientific_helpers(self):
        with patch.object(verify.postexit,'recorded_pids',return_value={10}), \
             patch.object(verify.postexit,'process_snapshot',return_value=[(10,1,'Python worker')]), \
             patch.object(verify,'verify_additive_sources',side_effect=AssertionError('must not reach sources')):
            with self.assertRaises(ValueError):verify.verify('/synthetic/root')


if __name__=='__main__':unittest.main()
