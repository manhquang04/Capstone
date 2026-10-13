"""Pure publication preflight tests; production audit is NOT invoked."""
import unittest
from experiments import priority35b_leakage_prepare_publication as p


class Tests(unittest.TestCase):
    def test_complete_stage_only(self):
        good=dict(stage='B3_MEASUREMENTS_FINISHED',done=9984,failed=0)
        p.require_finished(good,[],[])
        for bad in (dict(good,stage='B3_PROTECTED_RUNNING'),dict(good,done=9983),dict(good,failed=1)):
            with self.assertRaises(ValueError):p.require_finished(bad,[],[])
        with self.assertRaises(ValueError):p.require_finished(good,['lock'],[])
        with self.assertRaises(ValueError):p.require_finished(good,[],['failure'])

    def test_csv_preserves_types_and_fail_closed(self):
        header=('dataset','method','endpoint','n','mean_delta')
        row=p.numeric_csv('dataset,method,endpoint,n,mean_delta\nbaf,fixture,f1,11,-0.02\n',header,{'n'})[0]
        self.assertEqual(row['n'],11);self.assertEqual(row['mean_delta'],-.02)
        for text in ('dataset,method,endpoint,n,mean_delta\nbaf,fixture,f1,11,nan\n',
                     'dataset,method,endpoint,n\nbaf,fixture,f1,11\n'):
            with self.assertRaises(ValueError):p.numeric_csv(text,header,{'n'})


if __name__=='__main__':unittest.main()
