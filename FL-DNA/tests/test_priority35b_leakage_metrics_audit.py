"""Synthetic cross-implementation scoring checks, no protected data analysis."""
import unittest
from collections import OrderedDict
from types import SimpleNamespace
from experiments import priority35b_leakage as b
from experiments import priority35b_leakage_metrics_audit as audit


class Tests(unittest.TestCase):
    def test_reference_alignment_and_postprocessing(self):
        data = SimpleNamespace(
            train_features=OrderedDict(x=None,c=['a','b']),
            train_feature_index_map=OrderedDict(x=('cont',[0]),c=('cat',[1,2])),
            standardized=True, standardized_continuous_bounds={'x':(-1,1)},
            decode_batch=lambda x:b.np.array([[float(r[0]),'a' if r[1]>r[2] else 'b'] for r in x],dtype=object),
            create_tolerance_map=lambda:[.319,'cat'])
        truth = b.torch.tensor([[0.,1.,0.],[1.,0.,1.]])
        for rec in (truth.flip(0), b.torch.tensor([[3.,0.,1.],[.5,1.,0.]])):
            original = rec.clone()
            native = b.metric(data,truth,rec)
            separate = audit.independently_score(data,truth,rec)
            self.assertEqual(native, separate)
            self.assertTrue(b.torch.equal(rec,original))

    def test_native_detailed_denominator_and_boundary(self):
        for tolerances, x, y in [([.319],[[0.]],[[.319]]),
                                 ([.319],[[0.]],[[.320]]),
                                 (['cat'],[['a']],[['b']]),
                                 ([.319,'cat'],[[0.,'a']],[[.320,'b']])]:
            x,y=b.np.array(x,dtype=object),b.np.array(y,dtype=object)
            expected=b.core.native.match_reconstruction_ground_truth(x,y,tolerances,return_indices=True)[1]
            self.assertEqual(audit.decoded_costs(x,y,tolerances)[0,0],expected[0])

    def test_corruption_fail_closed(self):
        with self.assertRaises(ValueError): audit.decoded_costs([[float('nan')]],[[0.]],[.319])
        with self.assertRaises(ValueError): audit.decoded_costs([[0.]],[[0.]],[-.1])
        stored=dict(feature_accuracy=1.,assignment=[0],standardized_mse=.1,bounded_mse=.1)
        rec={k:v for k,v in stored.items() if k!='bounded_mse'}
        self.assertEqual(audit.verify_score(stored,rec,.2),0.)
        rec['standardized_mse']=.11
        with self.assertRaises(ValueError): audit.verify_score(stored,rec,.2)


if __name__=='__main__': unittest.main()
