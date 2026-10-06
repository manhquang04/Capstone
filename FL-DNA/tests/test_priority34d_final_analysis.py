import unittest
from experiments.priority34d_final_analysis import descriptive
from experiments.priority34d_statistics import MODELS, DNA


class FinalAnalysisTests(unittest.TestCase):
    def endpoints(self):
        rows=[]; offset=0
        for setting in MODELS+('trained_batch4',):
            width=4 if setting=='trained_batch4' else 1
            for index in range(39):
                ids=list(range(offset+index*width,offset+(index+1)*width))
                for arm in ('unprotected',*DNA):
                    rows.append(dict(setting=setting,arm=arm,target=index,indices=ids,
                        metrics=dict(psnr_db=10.+index,ssim=.1+index*.001,mse=.5-index*.001)))
            offset+=39*width
        return rows

    def test_complete_pooled(self):
        summaries,effects=descriptive(self.endpoints())
        pooled=[r for r in summaries if r['setting']=='pooled_initializations']
        self.assertEqual(len(pooled),3)
        self.assertTrue(all(r['metrics']['psnr_db']['n']==117 for r in pooled))
        self.assertTrue(all(r['descriptive_only'] for r in effects))

    def test_missing_pair_fails(self):
        with self.assertRaises(ValueError): descriptive(self.endpoints()[1:])

    def test_paired_effect_zero(self):
        _,effects=descriptive(self.endpoints())
        self.assertTrue(all(r['metrics']['mse']['median']==0 for r in effects))


if __name__=='__main__': unittest.main()
