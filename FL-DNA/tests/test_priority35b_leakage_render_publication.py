"""Synthetic rendering gates only; no production evidence or figures accessed."""
import unittest
from pathlib import Path
from experiments import priority35b_leakage_render_publication as r


class Tests(unittest.TestCase):
    def test_ready_requires_complete_stable_evidence(self):
        doc=dict(status='VERIFIED_PREPARATION_READY_FOR_RENDER',measurements=9984,
            scientific_files=49920,measurement_repeated=False,full_priority35b_complete=False)
        r.require_ready(doc)
        for key,value in (('measurements',9983),('scientific_files',49919),
                          ('measurement_repeated',True),('full_priority35b_complete',True),('status','COMPLETE')):
            bad=dict(doc);bad[key]=value
            with self.assertRaises(ValueError):r.require_ready(bad)

    def test_process_exit_and_path_bounds(self):
        r.require_actor_exit([(42,1,'Python unrelated')],43)
        with self.assertRaises(ValueError):r.require_actor_exit([(43,1,'unrelated reused PID')],43)
        with self.assertRaises(ValueError):r.require_actor_exit([],0)
        with self.assertRaises(ValueError):r.evidence_path('/tmp/p35b','reports/priority35b_report.md')
        with self.assertRaises(ValueError):r.evidence_path('/tmp/p35b','../escape')
        self.assertEqual(str(r.evidence_path('/tmp/p35b','artifacts/priority35b/b3/receipt.json')),
                         str(Path('/tmp/p35b/artifacts/priority35b/b3/receipt.json').resolve()))


if __name__=='__main__':unittest.main()
