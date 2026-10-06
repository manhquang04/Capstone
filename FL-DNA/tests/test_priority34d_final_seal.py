import tempfile
import unittest
from pathlib import Path
from experiments.priority34d_final_analysis import sha, write_once


class SealTests(unittest.TestCase):
    def test_hash_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'x';path.write_text('a'); first=sha(path)
            path.write_text('b'); self.assertNotEqual(first,sha(path))

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'x.json';write_once(path,{'a':1})
            with self.assertRaises(RuntimeError):write_once(path,{'a':2})

    def test_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):write_once(Path(folder)/'x.json',{'a':float('nan')})


if __name__=='__main__':unittest.main()
