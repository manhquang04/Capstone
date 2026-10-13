"""Complete synthetic9984-job data join; never reads measurement files."""
import unittest
from experiments import priority35b_leakage_data_bridge as b


def fixture():
    d=b.design
    arms={}
    for dataset in d.DATASETS:
        arms[dataset]=[dict(method=m,C=None,clipping=None,epsilon=None,noise_sd=0.) for m in ('unprotected',*d.TRANSFORMS)]
        arms[dataset]+=[dict(method=f'local_dp_{c}_eps{e}',C=.01,clipping=c,epsilon=e,noise_sd=.01)
                        for c in d.CLIPS for e in d.EPSILONS]
        arms[dataset].append(dict(method='b2b_global_c001_clip_only',C=.01,clipping='global',epsilon=None,noise_sd=0.))
        for tag,C in (('c01',.1),('cmed_half',.5),('cmed',1.)):
            arms[dataset].append(dict(method=f'b2b_global_{tag}_clip_only',C=C,clipping='global',epsilon=None,noise_sd=0.))
            arms[dataset]+=[dict(method=f'b2b_global_{tag}_eps{e}',C=C,clipping='global',epsilon=e,noise_sd=.01)
                            for e in (10,1000,1000000)]
    results=[]
    for dataset,cell in d.QUALIFIED:
        for i in range(39):
            for arm in arms[dataset]:
                results.append(dict(status='COMPLETED',config=dict(dataset=dataset,cell=cell,index=i,arm=arm['method']),
                    threads=1,inter_threads=1,source_ids=[i],reconstruction_seed=i,selected_variant='plain',
                    score=dict(feature_accuracy=i/39.,standardized_mse=.5,bounded_mse=.5),
                    controls=dict(mean_mode_metric=dict(standardized_mse=1.),scores=dict(mean_mode=50.,empirical_mean=25.))
                    if arm['method']=='unprotected' else None))
    records=b.result_rows(results,arms)
    frontier=[dict(dataset=dataset,cell=cell,arm=arm['method'],metric=m,n=39,ranks=[13,27],
                   median=19/39.,interval=[12/39.,26/39.])
              for dataset,cell in d.QUALIFIED for arm in arms[dataset] for m in b.tables.METRICS]
    utility=[dict(dataset=dataset,method='baseline' if arm['method']=='unprotected' else arm['method'],
                  endpoint=m,n=11,df=10,mean_delta=.02,ci_low=.01,ci_high=.03)
             for dataset in d.DATASETS for arm in arms[dataset] for m in ('f1','auc_roc','pr_auc')]
    selection=dict(panels=[dict(dataset=dataset,cell='ratio_batch1',protected_results_inspected=False,
                    unprotected_medians={cell:19/39. for dd,cell in d.QUALIFIED if dd==dataset}) for dataset in d.DATASETS])
    return results,records,arms,frontier,utility,selection


class Tests(unittest.TestCase):
    def test_complete_join_and_independent_data(self):
        _,records,arms,frontier,utility,selection=fixture()
        rows,controls=b.prepare_figure(records,arms,frontier,utility,selection)
        self.assertEqual(len(rows),96)
        self.assertEqual((rows[0]['x'],rows[0]['x_low'],rows[0]['x_high']),(-.02,-.03,-.01))
        self.assertEqual(controls['baf'],dict(mean_mode=.5,empirical_marginal=.25))
        self.assertEqual(b.independently_verify_figure(rows,controls,records,utility)['rows'],96)
        rows[0]['y']+=.01
        with self.assertRaises(ValueError):b.independently_verify_figure(rows,controls,records,utility)

    def test_incomplete_pair_selection_and_utility_fail_closed(self):
        results,records,arms,frontier,utility,selection=fixture()
        with self.assertRaises(ValueError):b.result_rows(results[:-1],arms)
        results[-1]['source_ids']=[999]
        with self.assertRaises(ValueError):b.result_rows(results,arms)
        with self.assertRaises(ValueError):b.prepare_figure(records,arms,frontier,utility[:-1],selection)
        selection['panels'][0]['cell']='tableak_batch1'
        with self.assertRaises(ValueError):b.prepare_figure(records,arms,frontier,utility,selection)


if __name__=='__main__':unittest.main()
