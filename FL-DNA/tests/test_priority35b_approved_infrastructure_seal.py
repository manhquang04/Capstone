import unittest
from experiments.priority35b_approved_infrastructure_seal import verify_failure_allowlist


class ApprovedFailureTests(unittest.TestCase):
    def test_only_exact_preserved_allowlist(self):
        verify_failure_allowlist({'old':'sha'},{'old':'sha'})
        for actual in ({'old':'changed'},{'old':'sha','new':'failure'},{}):
            with self.assertRaises(ValueError):verify_failure_allowlist(actual,{'old':'sha'})

    def test_scientific_failure_never_implicitly_waived(self):
        with self.assertRaises(ValueError):verify_failure_allowlist({'jobs/x/failure.json':'sha'},{'old':'sha'})


if __name__=='__main__':unittest.main()
