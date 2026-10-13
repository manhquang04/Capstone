"""Synthetic inventories only; no measured result or private receipt read."""
import hashlib
import unittest
from experiments import priority35b_leakage_manifest_audit as m


class Tests(unittest.TestCase):
    def test_all_hashes_and_real_mismatch(self):
        fake={'artifacts/priority35b/a':b'a','experiments/priority35b_b.py':b'b'}
        digests={p:hashlib.sha256(v).hexdigest() for p,v in fake.items()}
        self.assertEqual(m.verify_mapping(digests,lambda p:hashlib.sha256(fake[p]).hexdigest()),2)
        fake['experiments/priority35b_b.py']=b'changed'
        with self.assertRaises(ValueError):m.verify_mapping(digests,lambda p:hashlib.sha256(fake[p]).hexdigest())
        with self.assertRaises(ValueError):m.verify_mapping({'../outside':'a'*64},lambda p:'a'*64)
        with self.assertRaises(ValueError):m.verify_mapping({},lambda p:'a'*64)

    def test_full_measurement_coverage(self):
        jobs=[dict(dataset='fixture',cell='fixture',index=i,arm='fixture') for i in range(9984)]
        results=[dict(config=c,status='COMPLETED',attempt='attempt_1',files=dict.fromkeys(m.integrity.EXPECTED_FILES,'a'*64)) for c in jobs]
        paths=m.measurement_inventory(jobs,results)
        self.assertEqual(len(paths),49920)
        with self.assertRaises(ValueError):m.measurement_inventory(jobs,results[:-1])
        results[-1]=results[0]
        with self.assertRaises(ValueError):m.measurement_inventory(jobs,results)


if __name__=='__main__':unittest.main()
