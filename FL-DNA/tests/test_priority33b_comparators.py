import unittest
import torch
import numpy as np
from experiments import priority33b_comparators as p


class ComparatorTests(unittest.TestCase):
    def test_dp_lossless_and_finite(self):
        raw=[torch.tensor([1.,2.]),torch.tensor([3.])]
        for a,b in zip(p.dp_payload(raw,10.,0.,1),raw):torch.testing.assert_close(a,b)
        self.assertTrue(all(torch.isfinite(g).all() for g in p.dp_payload(raw,1.,.01,1)))

    def test_v2_projection_plans_are_live_and_payload_only(self):
        raw=[torch.arange(17,dtype=torch.float32)+1]
        q,plans=p.transformed_gradient(raw,"dna_v2_0p95")
        value=p.v2_sketch_loss(raw,q,plans)
        self.assertTrue(torch.isfinite(value))
        self.assertNotEqual(q[0].shape,raw[0].shape)
        self.assertTrue(all(set(plan)=={"padded_size","sampled","signs","scale"} for plan in plans))

    def test_block_debias_and_accounting(self):
        raw=[torch.arange(300,dtype=torch.float32)]
        corrected=p.debias_blocks(raw)[0]
        torch.testing.assert_close(corrected[:256],(raw[0][:256]-.08*raw[0][:256].mean())/.92)
        torch.testing.assert_close(corrected[256:],(raw[0][256:]-.08*raw[0][256:].mean())/.92)
        self.assertGreater(p.epsilon(.01)["epsilon"],0)


if __name__=="__main__":unittest.main()
