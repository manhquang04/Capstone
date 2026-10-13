"""Synthetic report-byte checks only; no scientific results read or written."""
import unittest
from experiments import priority35b_leakage_report_guard as guard


class Tests(unittest.TestCase):
    def setUp(self):
        self.original = b'Historical readiness report.\n'
        self.digest = guard.sha(self.original)
        self.length = len(self.original)
        self.checks = dict.fromkeys(guard.REQUIRED, True)
        self.section = guard.SECTION + '\n\nSynthetic verification fixture.\n'

    def test_exact_prefix(self):
        proposed = guard.prepare_append(self.original, self.section, self.checks, self.digest, self.length)
        receipt = guard.verify_append(self.original, proposed, self.digest, self.length)
        self.assertEqual(proposed[:self.length], self.original)
        self.assertFalse(receipt['full_priority35b_complete'])

    def test_every_gate_required(self):
        for name in guard.REQUIRED:
            bad = dict(self.checks); bad[name] = False
            with self.assertRaises(ValueError):
                guard.prepare_append(self.original, self.section, bad, self.digest, self.length)
        bad = dict(self.checks); bad.pop('csv_recomputation')
        with self.assertRaises(ValueError):guard.require_publication_checks(bad)
        bad = dict(self.checks); bad['tests'] = 1
        with self.assertRaises(ValueError):guard.require_publication_checks(bad)

    def test_drift_and_duplicate_append(self):
        with self.assertRaises(ValueError):
            guard.prepare_append(b'X' + self.original[1:], self.section, self.checks, self.digest, self.length)
        proposed = guard.prepare_append(self.original, self.section, self.checks, self.digest, self.length)
        with self.assertRaises(ValueError):
            guard.prepare_append(proposed, self.section, self.checks, self.digest, self.length)

    def test_wrong_section(self):
        with self.assertRaises(ValueError):
            guard.prepare_append(self.original, 'Replacement report', self.checks, self.digest, self.length)
        with self.assertRaises(ValueError):
            guard.verify_append(self.original, self.original + b'changed', self.digest, self.length)


if __name__ == '__main__':unittest.main()
