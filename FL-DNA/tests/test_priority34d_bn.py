import unittest
import numpy as np
from experiments import priority34d_bn as d


class BNTests(unittest.TestCase):
    def test_reservations(self):
        source=np.arange(2000,dtype=np.int64)
        doc=d.reserve(source,set(range(100)))
        ids=[i for checkpoint in doc['groups'].values() for stage in checkpoint.values()
             for row in stage['source_ids'] for i in row]
        self.assertEqual(len(ids),1140)
        self.assertEqual(len(set(ids)),1140)
        self.assertFalse(set(ids)&set(range(100)))
        self.assertEqual(doc,d.reserve(source,set(range(100))))

    def test_insufficient_sources(self):
        with self.assertRaises(ValueError):
            d.reserve(np.arange(1140),{1})

    def test_nonfinite(self):
        with self.assertRaises(FloatingPointError):
            d.finite(np.array([np.inf]),'synthetic')

    def test_negative_bn(self):
        with self.assertRaises(ValueError):
            d.b.strict_state({'running_var':d.torch.tensor([-1.])},'synthetic')

    def test_sign(self):
        self.assertLess(d.b.sign(np.ones(8),np.zeros(8))['p'],.05)
        self.assertEqual(d.b.sign(np.ones(8),np.ones(8))['p'],1.)

    def test_write_firewall(self):
        with self.assertRaises(ValueError):
            d.write(d.ROOT/'artifacts/priority33a/forbidden.json',{})

    def test_v2_projection(self):
        raw=np.arange(16,dtype=np.float32)/10
        q,meta=d.b.v2.transform_update_array_v2(raw,d.b.V2,tensor_index=0,
                      quantization_seed=d.b.derive_seed(d.b.V2.seed,'priority24-v2',0))
        matrix=d.b.projection(meta)
        # The exact known-key projection is linear; no scaled transpose decoder.
        self.assertEqual(matrix.shape,(meta.sketch_size,meta.original_size))
        self.assertTrue(np.isfinite(matrix).all())


if __name__=='__main__':
    unittest.main()
