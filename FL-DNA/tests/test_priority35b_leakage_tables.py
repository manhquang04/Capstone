"""Synthetic complete table roundtrip and corruption rejection."""
import unittest
from experiments import priority35b_leakage_tables as tables


def fixture():
    d=tables.design
    arms={dataset:[dict(method=m) for m in ('unprotected',*d.TRANSFORMS,*['fixture_'+str(i) for i in range(29)])] for dataset in d.DATASETS}
    rows=[dict(dataset=dataset,cell=cell,index=i,arm=arm['method'],feature_accuracy=i/39.,
               standardized_mse=i/10.,bounded_mse=min(i/10.,1.),prior_mse=1.)
          for dataset,cell in d.QUALIFIED for i in range(39) for arm in arms[dataset]]
    matching=[dict(dataset=dataset,transform=t,clipping=clip,status='MATCHED_CLIP_ONLY_NO_DP',matched_arm='fixture_0')
              for dataset in d.DATASETS for t in d.TRANSFORMS for clip in d.CLIPS]
    return rows,arms,matching


class Tests(unittest.TestCase):
    def test_full_roundtrip_and_independent_recomputation(self):
        rows,arms,matches=fixture();output=tables.prepare(rows,arms,matches)
        csvs=[tables.csv_string(output[k]) for k in ('per_target','frontier','holm')]
        self.assertEqual(tables.verify_csvs(*csvs,arms,matches)['holm'],144)
        output['frontier'][0]['median']+=.1
        with self.assertRaises(ValueError):tables.verify_csvs(csvs[0],tables.csv_string(output['frontier']),csvs[2],arms,matches)

    def test_missing_and_nonfinite_fail_closed(self):
        rows,arms,matches=fixture()
        with self.assertRaises(ValueError):tables.prepare(rows[:-1],arms,matches)
        rows[0]['feature_accuracy']=float('nan')
        with self.assertRaises(ValueError):tables.prepare(rows,arms,matches)


if __name__=='__main__':unittest.main()
