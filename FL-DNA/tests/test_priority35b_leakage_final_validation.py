import unittest
from experiments import priority35b_leakage_final_validation as v


class Tests(unittest.TestCase):
    def test_all_suites_are_separate_interpreters(self):
        commands=v.test_commands('/synthetic/python')
        self.assertEqual(len(commands),20)
        self.assertEqual(len({tuple(c) for c in commands}),20)
        for command in commands:
            self.assertEqual(command[:4],['/synthetic/python','-B','-m','unittest'])
            self.assertTrue(command[4].startswith('tests.test_priority35b_'))

    def test_missing_or_failed_test_output_fails_closed(self):
        self.assertEqual(v.successful_suite(0,'..\nRan 2 tests in 0.002s\n\nOK\n'),2)
        for code,output in ((1,'Ran 2 tests in 0.002s\nOK\n'),(0,''),(0,'Ran 0 tests in 0s\nOK\n'),
                            (0,'Ran 2 tests in 0s\nFAILED (failures=1)\n')):
            with self.assertRaises(ValueError):v.successful_suite(code,output)


if __name__=='__main__':unittest.main()
