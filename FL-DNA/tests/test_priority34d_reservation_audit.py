import unittest
import numpy as np
from experiments.priority34d_reservation_audit import ids, check_groups, SETTINGS


class ReservationAuditTests(unittest.TestCase):
    def groups(self):
        order = np.random.default_rng(342600).permutation(range(10000)).tolist()
        groups = {}; position = 0
        for setting in SETTINGS:
            width = 4 if setting == 'trained_batch4' else 1
            groups[setting] = [order[position+i*width:position+(i+1)*width] for i in range(39)]
            position += 39*width
        return groups

    def test_nested(self):
        self.assertEqual(ids({'job': {'indices': [3, 5]}, 'C2_batch4': [[7, 9]], 'count': 17}), {3, 5, 7, 9})

    def test_candidate(self):
        self.assertEqual(len(check_groups(self.groups(), set())), 273)

    def test_historical(self):
        groups = self.groups()
        with self.assertRaises(ValueError):
            check_groups(groups, {groups[SETTINGS[0]][0][0]})

    def test_duplicate(self):
        groups = self.groups(); groups[SETTINGS[0]][1] = groups[SETTINGS[0]][0]
        with self.assertRaises(ValueError):
            check_groups(groups, set())

    def test_order(self):
        groups = self.groups(); groups[SETTINGS[0]].reverse()
        with self.assertRaises(ValueError):
            check_groups(groups, set())


if __name__ == '__main__':
    unittest.main()
