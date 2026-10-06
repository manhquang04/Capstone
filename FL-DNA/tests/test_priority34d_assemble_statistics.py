import unittest
import numpy as np
from experiments import priority34d_assemble_statistics as a


def fixture():
    rows=[]; source=0
    for setting in a.s.MODELS+('trained_batch4',):
        for target in range(39):
            ids=list(range(source,source+(4 if setting=='trained_batch4' else 1))); source+=len(ids)
            for arm,offset in [('unprotected',0.),(a.s.DNA[0],-.2),(a.s.DNA[1],-.1)]:
                rows.append(dict(setting=setting,arm=arm,target=target,indices=ids,
                                 metrics=dict(psnr_db=10+target*.01+offset,ssim=.3,mse=.1)))
    matches={setting+'::'+dna+'::'+mechanism:dict(status='NOT_ASSESSABLE')
             for setting in a.s.MODELS+('trained_batch4',) for dna in a.s.DNA
             for mechanism in ('single','per_tensor')}
    previous=[dict(hypothesis_id='old'+str(i),p_raw=.5) for i in range(135)]
    return rows,matches,previous


class AssemblyTests(unittest.TestCase):
    def test_fixed_family_and_gated_reservations(self):
        images,matches,previous=fixture()
        result=a.assemble(images,[],[],matches,previous)
        self.assertEqual(len(result['rows']),76)
        self.assertEqual(len(result['combined']),211)
        self.assertTrue(result['independent_sign_holm_verified'])
        pooled=[r for r in result['rows'] if r['setting']=='pooled_initializations' and r['comparator']=='unprotected']
        self.assertTrue(all(r['n']==117 for r in pooled))
        gated=[r for r in result['rows'] if r['status']=='NOT_ASSESSABLE']
        self.assertTrue(all(r['p_raw']==1 for r in gated))

    def test_missing_arm_fails_not_gated(self):
        images,_,_=fixture()
        with self.assertRaises(ValueError): a.image_registry(images[:-1])

    def test_pairing_and_duplicate_fail(self):
        images,_,_=fixture()
        with self.assertRaises(ValueError): a.image_registry(images+[images[0]])
        images[1]=dict(images[1],indices=[99999])
        with self.assertRaises(ValueError): a.image_registry(images)

    def test_bn_missing_pairs(self):
        with self.assertRaises(ValueError): a.bn_registry([], [342000])
        self.assertEqual(a.bn_registry([],[]),{})

    def test_separate_holm_and_sign_check(self):
        p=[.9,.001,.001,.04,1.]
        np.testing.assert_array_equal(a.independent_holm(p),a.s.holm(p))
        images,matches,previous=fixture()
        result=a.assemble(images,[],[],matches,previous)
        result['rows'][0]['p_raw']=.9
        with self.assertRaises(ValueError): a.verify_statistics(result['rows'],result['combined'])


if __name__=='__main__': unittest.main()
