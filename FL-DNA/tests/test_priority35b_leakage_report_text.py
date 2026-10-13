"""Report renderer syntax/safety tests with synthetic provenance only."""
import unittest
from experiments import priority35b_leakage_report_text as report
from experiments import priority35b_leakage_report_guard as guard


class Tests(unittest.TestCase):
    def test_table_escape_and_finite(self):
        rendered=report.table([{'a':'x|y\nz','b':0.125}],('a','b'))
        self.assertIn('x&#124;y z',rendered)
        self.assertIn('0.125',rendered)
        with self.assertRaises(ValueError):report.cell(float('nan'))
        with self.assertRaises(ValueError):report.table([{'a':1}],('a','b'))

    def test_missing_evidence_cannot_render(self):
        with self.assertRaises(ValueError):report.section({},dict.fromkeys(guard.REQUIRED,True))
        with self.assertRaises(ValueError):report.section({},dict.fromkeys(guard.REQUIRED,False))


if __name__=='__main__':unittest.main()
