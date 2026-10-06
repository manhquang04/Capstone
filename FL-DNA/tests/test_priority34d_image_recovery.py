import inspect
import unittest
import numpy as np
import torch
from experiments import priority34d_image_recovery as d


class ImageTests(unittest.TestCase):
    def test_source_disjoint(self):
        targets=d.reserve(set(range(4000)))
        ids=[i for rows in targets['groups'].values() for row in rows for i in row]
        self.assertEqual(len(ids),273)
        self.assertEqual(len(set(ids)),273)
        self.assertTrue(all(i>=4000 for i in ids))
        self.assertTrue(all(len(row)==4 for row in targets['groups']['trained_batch4']))

    def test_collect_nested(self):
        self.assertEqual(d.collect({'job':{'indices':[5,6]},'C2_batch4':[[7,8]],'targets':[9]}),{5,6,7,8,9})

    def test_insufficient(self):
        with self.assertRaises(ValueError):
            d.reserve(set(range(9800)))

    def matches(self,matched=True):
        return {setting+'::'+method+'::'+mechanism:dict(status='MATCHED' if matched else 'NOT_ASSESSABLE',sigma=.001)
                for setting in d.u.SETTINGS for method in d.u.DNA for mechanism in ('single','per_tensor')}

    def test_full_schedule(self):
        jobs,missing=d.build_jobs(d.reserve(set()),self.matches())
        self.assertEqual(len(jobs),1092)
        self.assertEqual(len({j['id'] for j in jobs}),1092)
        self.assertEqual(missing,[])

    def test_gate_reservations(self):
        jobs,missing=d.build_jobs(d.reserve(set()),self.matches(False))
        self.assertEqual(len(jobs),468)
        self.assertEqual(len(missing),16)

    def test_attacker_interface(self):
        fields=inspect.signature(d.reconstruct).parameters
        self.assertFalse({'truth','raw','noise_seed','private_seed'}&set(fields))
        self.assertIn('received',fields)

    def test_batch4_alignment(self):
        gen=torch.Generator().manual_seed(12)
        truth=torch.rand((4,3,32,32),generator=gen)
        estimate=truth[[2,0,3,1]]+.01
        means,records,pairing=d.score(truth,estimate)
        self.assertEqual(pairing,[1,3,0,2])
        self.assertEqual(len(records),4)
        self.assertTrue(all(np.isfinite(v) for v in means.values()))
        for key in means:
            self.assertAlmostEqual(means[key],np.mean([r[key] for r in records]))

    def test_fail_before_metric_clipping(self):
        with self.assertRaises(FloatingPointError):
            d.score(torch.zeros((1,3,32,32)),torch.full((1,3,32,32),float('nan')))

    def test_frozen_native_budget(self):
        self.assertEqual(d.native.IMAGE_CONFIG['max_iterations'],4800)
        self.assertEqual(d.native.IMAGE_CONFIG['restarts'],1)


if __name__=='__main__':
    unittest.main()
