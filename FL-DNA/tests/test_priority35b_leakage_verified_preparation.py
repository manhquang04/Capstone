"""Synthetic hash-ledger tests; production verification is not invoked."""
import unittest
from experiments import priority35b_leakage_verified_preparation as p


class Tests(unittest.TestCase):
    def test_complete_stability(self):
        hashes={str(i):'a'*64 for i in range(49920)}
        receipt=p.check_stability(hashes,dict(hashes))
        self.assertFalse(receipt['full_priority35b_complete'])
        self.assertFalse(receipt['measurement_repeated'])
        self.assertEqual(receipt['entries'],49920)

    def test_changed_or_missing_file_fail_closed(self):
        hashes={str(i):'a'*64 for i in range(49920)}
        changed=dict(hashes);changed['49919']='b'*64
        with self.assertRaises(ValueError):p.check_stability(hashes,changed)
        changed.pop('49919')
        with self.assertRaises(ValueError):p.check_stability(hashes,changed)


if __name__=='__main__':unittest.main()
