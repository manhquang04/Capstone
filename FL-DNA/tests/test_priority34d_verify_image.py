import unittest
from unittest.mock import patch
import numpy as np
from experiments import priority34d_verify_image as v


class ImageVerifierTests(unittest.TestCase):
    def test_finite_precision(self):
        v.compare([1.],[1.+1e-8])
        with self.assertRaises(ValueError):
            v.compare([np.inf],[1.])

    def test_plain_tie(self):
        self.assertEqual(v.selected_mode(dict(v1_debias=.1,plain=.1)),'plain')
        self.assertEqual(v.selected_mode(dict(v1_debias=.05,plain=.1)),'v1_debias')

    def test_tensor_plan_equality(self):
        plans=[dict(signs=v.r.torch.tensor([1.,-1.]),sampled=v.r.torch.tensor([1,3]),scale=2.)]
        clone=[{key:value.clone() if v.r.torch.is_tensor(value) else value for key,value in plans[0].items()}]
        self.assertTrue(v.same_plans(plans,clone))
        clone[0]['signs'][0]=-1
        self.assertFalse(v.same_plans(plans,clone))

    def test_noise_bit_exact(self):
        torch=v.r.torch
        values=[torch.arange(12,dtype=torch.float32).reshape(3,4),torch.tensor([.1,.2])]
        clip=dict(C=2.31657,clips=[1.731,.13])
        for sigma in (.0001,.003,.3):
            expected=v.expected_dp(values,clip,sigma,12345,'single')
            observed=v.r.u.old.p31.dp.add_clipped_noise(values,clip['C'],sigma,12345)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(expected,observed)))
            expected=v.expected_dp(values,clip,sigma,12345,'per_tensor')
            observed=v.r.u.old.per_tensor_noise(values,clip['clips'],sigma,12345)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(expected,observed)))

    def test_no_early_target_access(self):
        with patch.object(v.r,'read',side_effect=FileNotFoundError('not complete')):
            with patch.object(v.r.u.old.p31.dp,'CIFAR10') as dataset:
                with self.assertRaises(FileNotFoundError):
                    v.verify()
                dataset.assert_not_called()


if __name__=='__main__':
    unittest.main()
