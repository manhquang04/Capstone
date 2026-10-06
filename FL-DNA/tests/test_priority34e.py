import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from dataclasses import asdict
import numpy as np
from dna_encoder import transform_defense_v2 as v
from experiments.priority34e_seed_search import recognize,ranks
from experiments.priority34e import SEEDS,KS
from experiments import priority34a_local_bn as a

class Preflight(unittest.TestCase):
    def test_completed_resume_hash_gate(self):
        from experiments import priority34e_utility as utility
        from experiments.priority34e import write
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);artifact=folder/'artifact.bin';artifact.write_bytes(b'validated')
            config={'K':10,'training':{'rounds':50}}
            client={split:{name:.5 for name in ('f1','auc_roc','pr_auc')} for split in ('validation','test')}
            result={'status':'COMPLETED','config_sha256':a.p.canonical_sha(config),'no_bn_transmitted':True,
                'payload_assertions':500,'clients':[client]*10,'torch_threads':1,'torch_interop_threads':1,'device':'cpu',
                'artifact_sha256':{'artifact.bin':a.p.sha(artifact)},**client}
            with patch('experiments.priority34e.OUT',folder),patch.object(utility,'OUT',folder):
                write(folder/'result.json',result)
                self.assertTrue(utility.valid_result(folder/'result.json',config))
                artifact.write_bytes(b'corrupt')
                with self.assertRaises(ValueError):utility.valid_result(folder/'result.json',config)
    def test_seed_false_positive_is_not_success(self):
        result=ranks([0,0,0],1)
        self.assertFalse(result['recognition_validated'])
    def test_write_firewall(self):
        from experiments.priority34e import write
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):write(Path(directory)/'outside.json',{})
    def test_statistics_complete_pairs(self):
        from experiments.priority34e_analysis import summary
        result=summary(list(range(11)))
        self.assertEqual(result['mean_delta'],5.)
        self.assertLess(result['ci95'][0],5.)
        self.assertGreater(result['ci95'][1],5.)
        with self.assertRaises(ValueError):summary([0]*10)
        with self.assertRaises(ValueError):summary([float('nan')]*11)
    def test_true_fingerprint(self):
        _,metadata=v.transform_update_array_v2(np.arange(128,dtype=np.float32),v.DNATransformV2Config(seed=37))
        public=asdict(metadata);public.pop('seed')
        self.assertTrue(recognize(37,public))
        self.assertFalse(recognize(38,public))
    def test_firewall(self):
        with self.assertRaises(ValueError):recognize(1,{'true_seed':1})
    def test_unique_rank_and_false_positive(self):
        self.assertTrue(ranks([0,1,0],1)['recognition_validated'])
        bad=ranks([1,1,0],1)
        self.assertEqual(bad['false_positives'],1)
        self.assertFalse(bad['recognition_validated'])
        self.assertEqual(ranks([1,0,0],1)['true_rank'],2)
    def test_partition_generalization(self):
        labels=np.array([1]*41+[0]*1000,dtype=np.float32)
        cats=np.array(['fraud']*41+['a']*500+['b']*500)
        for k in KS:
            parts=a.p._mild_non_iid_client_indices(a.p.pd.DataFrame({'type':cats}),labels,k,343000)
            self.assertEqual(len(np.unique(np.concatenate(parts))),len(labels))
            counts=[int(labels[x].sum()) for x in parts]
            self.assertLessEqual(max(counts)-min(counts),1)
            for category,primary in [('a',0),('b',1)]:
                count=sum(cats[i]==category and labels[i]==0 for i in parts[primary])
                self.assertLessEqual(abs(count-275),1)
    def test_no_BN_payload(self):
        model=a.p.FraudMLP(58);bn,allowed=a.domains(model)
        self.assertFalse(set(allowed)&bn)
        self.assertEqual(len(SEEDS),11)
    def test_norm_and_grid_not_key_evidence(self):
        q=np.arange(61,dtype=float);norms=[]
        for candidate in (1,2,3):
            derived=v._derive_seed(candidate,0);sampled=v._sampled_indices(64,61,derived)
            lifted=np.zeros(64);lifted[sampled]=np.sqrt(64/61)*q
            decoded=v._fwht_normalized(lifted)*v._signs(64,derived)
            norms.append(np.linalg.norm(decoded))
        np.testing.assert_allclose(norms,norms[0],atol=1e-12,rtol=1e-12)
if __name__=='__main__':unittest.main()
