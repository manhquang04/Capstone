import unittest
from unittest.mock import patch
import numpy as np
from experiments import priority34d_verify_utility as v


class UtilityVerifierTests(unittest.TestCase):
    def test_registry(self):
        self.assertEqual(v.registry(v.u.GRID,v.u.SETTINGS,('single','per_tensor'),True),v.u.jobs())
        self.assertEqual(len(v.registry(v.u.EXTENSION,['init342042'],['single'],False)),48)

    def test_independent_bracket(self):
        base=np.full(16,.5); dna=np.full(16,.48)
        dp={.001:np.full(16,.49),.01:np.full(16,.47)}
        result=v.bracket(base,dna,dp)
        self.assertEqual(result['status'],'MATCHED')
        self.assertEqual(result['sigma'],.001)
        self.assertEqual(result['next_sigma'],.01)
        native=v.u.bracket(float(base.mean()),float(np.mean(dna-base)),{s:float(np.mean(a-base)) for s,a in dp.items()})
        native['grid_means']={str(s):a for s,a in native['grid_means'].items()}
        self.assertEqual(result,native)

    def test_extension_and_floor(self):
        base=np.full(16,.5)
        self.assertTrue(v.bracket(base,base,{.3:base})['extension_needed'])
        low=np.full(16,.39)
        self.assertFalse(v.bracket(low,low,{.3:low})['extension_needed'])

    def test_invalid_pairing(self):
        for values in ([.5]*15,[np.nan]*16,[1.1]*16):
            with self.assertRaises(ValueError): v.paired_vector(values)

    def test_no_early_data(self):
        with patch.object(v.u,'read',side_effect=FileNotFoundError('not complete')):
            with patch.object(v.u.old.p31,'utility_data') as data:
                with self.assertRaises(FileNotFoundError): v.verify()
                data.assert_not_called()


if __name__=='__main__': unittest.main()
