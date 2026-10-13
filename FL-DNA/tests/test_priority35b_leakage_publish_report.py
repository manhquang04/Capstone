import unittest
from experiments import priority35b_leakage_publish_report as p


class Tests(unittest.TestCase):
    def test_reading_requires_exact_full_csv_binding(self):
        inputs={'artifacts/priority35b/b3/synthetic/frontier.csv':'a'*64,
                'artifacts/priority35b/b3/synthetic/holm.csv':'b'*64}
        doc=dict(status='FULL_RESULTS_DESCRIPTIVE_READING',inputs=inputs,text='Synthetic interpretation, not results.')
        self.assertEqual(p.require_interpretation(doc,inputs),doc['text'])
        with self.assertRaises(ValueError):p.require_interpretation(doc,{**inputs,'other':'c'*64})
        doc['status']='PARTIAL'
        with self.assertRaises(ValueError):p.require_interpretation(doc,inputs)

    def test_missing_or_duplicate_report_section_fails(self):
        for value in ('',None,p.contract.guard.SECTION):
            with self.assertRaises(ValueError):p.require_interpretation(
                dict(status='FULL_RESULTS_DESCRIPTIVE_READING',inputs={},text=value),{})


if __name__=='__main__':unittest.main()
