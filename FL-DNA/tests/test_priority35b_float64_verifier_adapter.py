import math
import unittest
from experiments import priority35b_float64_verifier_adapter as adapter


class Float64AdapterTests(unittest.TestCase):
    def test_one_ulp_large_mse(self):
        a=55262900.7920114;b=math.nextafter(a,math.inf)
        self.assertGreater(abs(a-b),1e-12)
        self.assertLessEqual(abs(a-b),adapter.tolerance(a,b,58))

    def test_real_mismatch_rejected_by_bound(self):
        self.assertGreater(abs(1.-1.001),adapter.tolerance(1.,1.001,952))
        with self.assertRaises(ValueError):adapter.tolerance(float('nan'),1.,58)

    def test_native_must_be_exact(self):
        stored=dict(feature_accuracy=1.,standardized_mse=1.,assignment=[0])
        adapter.check_native(stored,dict(stored))
        with self.assertRaises(ValueError):adapter.check_native(stored,dict(stored,standardized_mse=math.nextafter(1.,2.)))

    def test_original_accuracy_assignment_and_bounded_rules(self):
        original=adapter.metric.verify_score
        stored=dict(feature_accuracy=1.,standardized_mse=1.,assignment=[0],bounded_mse=.5)
        independent=dict(feature_accuracy=1.,standardized_mse=math.nextafter(1.,2.),assignment=[0])
        original(stored,independent,.5,mse_tolerance=adapter.tolerance(1.,independent['standardized_mse'],58))
        for bad in (dict(stored,feature_accuracy=.9),dict(stored,assignment=[1]),dict(stored,bounded_mse=.4)):
            with self.assertRaises(ValueError):original(bad,independent,.5,mse_tolerance=1e-12)


if __name__=='__main__':unittest.main()
