import unittest
from experiments import priority35b_leakage_statistics as s

class StatisticsTests(unittest.TestCase):
    def test_full_family_and_independent_recomputation(self):
        expected={tuple(k[:-1]) for k in s.design.family()}
        key=next(iter(expected));pairs={key:([1.]*20+[0.]*19,[0.]*39)}
        reasons={k:'unmatched or original ineligible slot' for k in expected-{key}}
        rows=s.evaluate(pairs,reasons)
        self.assertEqual(s.independently_verify(rows,pairs,reasons)['status'],'PASS')
        self.assertEqual(sum(r['status']=='NOT_ASSESSABLE' for r in rows),142)
        bad=[dict(r) for r in rows];bad[0]['holm_p']=.2
        with self.assertRaises(ValueError):s.independently_verify(bad,pairs,reasons)

    def test_missing_pair_never_NA_success(self):
        expected={tuple(k[:-1]) for k in s.design.family()};key=next(iter(expected))
        with self.assertRaises(ValueError):s.evaluate({key:([1.]*38,[0.]*38)},{k:'not matched' for k in expected-{key}})
        with self.assertRaises(ValueError):s.evaluate({}, {})

if __name__=='__main__':unittest.main()
