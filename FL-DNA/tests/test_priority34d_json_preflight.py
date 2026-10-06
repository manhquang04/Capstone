import unittest
from experiments.priority34d_json_preflight import compare, normalize_keys


class JsonPreflightTests(unittest.TestCase):
    def test_int_and_string_keys_equal(self):
        compare({'grid': {1: .125, 3: -.5}}, {'grid': {'1': .125, '3': -.5}})

    def test_float_and_string_keys_equal(self):
        compare({'grid': {.001: -.007}}, {'grid': {'0.001': -.007}})

    def test_real_mismatch_fails(self):
        with self.assertRaises(ValueError):
            compare({'grid': {1: .125}}, {'grid': {'1': .126}})

    def test_no_tolerance(self):
        with self.assertRaises(ValueError):
            compare({1: 1.}, {'1': 1.000000000000001})

    def test_both_sides_and_preserve_values(self):
        original = {1: {'value': [.1, .2]}}
        compare(original, {'1': {'value': [.1, .2]}})
        self.assertEqual(original, {1: {'value': [.1, .2]}})
        self.assertEqual(normalize_keys(original)['1']['value'], [.1, .2])

    def test_collision_fails(self):
        with self.assertRaises(ValueError):
            compare({1: .1, '1': .1}, {'1': .1})


if __name__ == '__main__':
    unittest.main()
