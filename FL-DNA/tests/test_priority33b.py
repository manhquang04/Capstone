"""Pre-run synthetic P33b representation, budget and observable-payload gates."""
import unittest
from unittest.mock import patch
from collections import OrderedDict
import numpy as np
import torch
from experiments import priority33b_core as c


class P33bTests(unittest.TestCase):
    def test_logical_blocks_and_train_only_selection(self):
        meta=dict(feature_names=["amount","payment=A","payment=B","frequency"],one_hot_levels={"payment":["A","B"]})
        blocks=c.logical_blocks(meta)
        self.assertEqual([b["indices"] for b in blocks],[[0],[1,2],[3]])
        self.assertEqual(c.select_blocks(blocks,[.1,.9,.8],2),[1])
        self.assertEqual(c.select_blocks(blocks,[.1,.9,.8],3),[1,2])

    def test_fractional_continuous_decode_preserved(self):
        dataset=c.PreparedDataset.__new__(c.PreparedDataset)
        dataset.train_features=OrderedDict(amount=None,payment=["A","B"])
        dataset.mean=torch.zeros(3)
        dataset.std=torch.ones(3)
        truth=torch.tensor([[.127,1.,0.],[.291,0.,1.]])
        decoded=dataset.decode_batch(truth)
        np.testing.assert_allclose(decoded[:,0].astype(float),truth[:,0].numpy())
        self.assertEqual(decoded[:,1].tolist(),["A","B"])

    def test_native_budget(self):
        config=c.official_config()
        self.assertEqual(config["max_iterations"],1500)
        self.assertEqual(config["post_selection"],30)
        self.assertEqual(config["pooling"],"median+softmax")
        self.assertFalse(config["perfect_pooling"])
        self.assertEqual(config["device"],"cpu")

    def test_payload_object_and_fail_closed(self):
        class Dataset:
            num_features=2
        payload=[torch.tensor([1.,2.])]
        labels=torch.zeros(8,dtype=torch.long)
        def fake(**kw):
            self.assertIs(kw["true_grad"],payload)
            self.assertEqual(kw["true_data"].shape,(8,2))
            loss=c.gia._cosine_similarity_loss(reconstruct_gradient=payload,true_grad=kw["true_grad"],device="cpu")
            self.assertAlmostEqual(float(loss),0.,places=6)
            rec=torch.zeros(8,2)
            return rec,[rec.clone() for _ in range(30)],[0.]*30
        with patch.object(c.gia,"invert_grad",fake):
            c.native_recover(None,Dataset(),payload,labels,1)
        with self.assertRaises(ValueError):
            c.native_recover(None,Dataset(),payload,labels,1,mode="wrong")
        with self.assertRaises(FloatingPointError):
            c.finite(torch.tensor([float("nan")]),"synthetic")

    def test_lossless_sketch_loss(self):
        observed=[torch.tensor([1.,2.]),torch.tensor([3.])]
        candidate=[torch.tensor([.7,2.1],requires_grad=True),torch.tensor([2.8],requires_grad=True)]
        value=c.v2_sketch_loss(candidate,observed,[torch.eye(2),torch.eye(1)])
        original=c.gia._cosine_similarity_loss(candidate,observed,"cpu")
        torch.testing.assert_close(value.reshape(-1),original.reshape(-1),atol=1e-7,rtol=1e-6)
        value.backward()
        self.assertTrue(all(torch.isfinite(t.grad).all() for t in candidate))

    def test_sign_test(self):
        self.assertEqual(c.exact_p(8,0),1/256)
        self.assertEqual(c.exact_p(0,0),1.)


if __name__=="__main__":
    unittest.main()
