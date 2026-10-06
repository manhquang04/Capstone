import unittest
from experiments import priority34d_report_text as report
from experiments import priority34d_assemble_statistics as assembly
from tests.test_priority34d_assemble_statistics import fixture


class ReportTests(unittest.TestCase):
    def context(self):
        images,matches,previous=fixture()
        bundle=assembly.assemble(images,[],[],matches,previous)
        utility=dict(matches={key:dict(row,baseline_mean=.5,target_delta=-.01,
                                      threshold=-.015,sigma=None,next_sigma=None)
                              for key,row in matches.items()})
        clips={setting:dict(C=1.,clips=[.5,.4]) for setting in report.s.MODELS+('trained_batch4',)}
        return bundle,utility,clips,[],[],{str(seed):'failed qualification; no confirmatory' for seed in report.s.BN_SEEDS},{'synthetic only':'0'*64}

    def test_required_scope_and_fixed_family(self):
        text=report.render(*self.context())
        for phrase in ('not COMPLETE','Holm211','ranks13/27','not record-level','batch256',
                       'Private truth','NOT_ASSESSABLE','342044','342002'):
            self.assertIn(phrase,text)
        self.assertEqual(text.count('P34D::'),76)
        self.assertNotIn('verified final audit PASS',text)

    def test_hash_guard(self):
        args=list(self.context()); args[-1]={'bad':'not a hash'}
        with self.assertRaises(ValueError): report.render(*args)

    def test_interval_and_finite(self):
        text=report.interval(report.s.median_interval(list(range(39))))
        self.assertIn('ranks 13/27',text)
        with self.assertRaises(ValueError): report.number(float('nan'))


if __name__=='__main__': unittest.main()
