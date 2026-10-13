"""No-live gate tests using fake process tables; no measurement data read."""
import unittest
from experiments import priority35b_leakage_postexit as p


class Tests(unittest.TestCase):
    def test_empty_and_unrelated(self):
        self.assertEqual(p.require_exit([(7,1,'unrelated'),(9,1,'own')],{100,101},self_pid=9),[100,101])

    def test_recorded_worker_and_descendant(self):
        for rows in ([(100,1,'unrelated PID reused')],[(102,100,'Python -c spawn_main')]):
            with self.assertRaises(ValueError):p.require_exit(rows,{100},self_pid=9)

    def test_case_insensitive_scientific_process(self):
        with self.assertRaises(ValueError):
            p.require_exit([(5,1,'Python PRIORITY35B_LEAKAGE.py --SUPERVISE')],{100},self_pid=9)


if __name__=='__main__':unittest.main()
