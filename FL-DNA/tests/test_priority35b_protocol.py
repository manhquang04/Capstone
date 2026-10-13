"""Synthetic preflight and read-only historical evidence, never measurements."""
import json
import math
import unittest
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from experiments import priority35b_design as d
from experiments import priority35b_firewall as fw


class DesignTests(unittest.TestCase):
    def values(self, x): return dict.fromkeys(d.SEEDS, x)
    def grid(self, x): return {e: self.values(x) for e in d.EPSILONS}

    def test_noise_exact_p34b_1_3_10(self):
        from experiments import priority34b_local_dp as old
        frozen = json.loads((old.OUT/'execution_freeze.json').read_text())['accounting']
        for row in frozen:
            new = d.account(row['target_epsilon'])
            self.assertEqual(new['noise_sd'], row['noise_sd'])
            self.assertEqual(new['sigma_sensitivity'], old.account(row['target_epsilon'])['sigma_sensitivity'])
        self.assertEqual(old.old.SEEDS, list(d.SEEDS))

    def test_account_all_points_independent_bound(self):
        sds = []
        for e in d.EPSILONS:
            row = d.account(e); alpha = row['alpha']; sigma = row['sigma_sensitivity']
            bound = 50*alpha/(2*sigma**2) + math.log(1e5)/(alpha-1)
            self.assertLessEqual(abs(bound-e), 16*math.ulp(float(e)))
            self.assertEqual(row['sensitivity'], .02); sds.append(row['noise_sd'])
        self.assertTrue(all(a>b for a,b in zip(sds,sds[1:])))
        for e in [float('nan'), float('inf'), 0, True, 2]:
            with self.assertRaises(ValueError): d.account(e)

    def test_match_bracket_and_equality(self):
        grid = self.grid(.4); grid[100] = self.values(.495)
        result = d.match(self.values(.5), grid)
        self.assertEqual((result['epsilon'], result['next_noisier_epsilon']), (100,30))
        self.assertTrue(result['nonmonotonic'])  # never force monotonicity

    def test_match_no_bracket_missing_and_bad(self):
        for x in [.1,.9]: self.assertEqual(d.match(self.values(.5),self.grid(x))['status'],'NOT_ASSESSABLE')
        grid=self.grid(.2); del grid[3]
        with self.assertRaises(ValueError): d.match(self.values(.5),grid)
        grid=self.grid(.2); grid[3][321000]=float('nan')
        with self.assertRaises(ValueError): d.match(self.values(.5),grid)
        with self.assertRaises(ValueError): d.f1_values({321000:.5})

    def test_intervals_signs_ties_independent(self):
        row=d.paired(list(range(39)),[19.]*39)
        self.assertEqual(row['interval'],[-7.,7.]); self.assertEqual(row['ties'],1)
        self.assertEqual(row['raw_p_lower'],binomtest(19,38,.5,alternative='greater').pvalue)
        self.assertEqual(d.exact_tail(39,0),2**-39)
        self.assertEqual(d.paired([1.]*39,[1.]*39)['raw_p_lower'],1)
        with self.assertRaises(ValueError): d.interval39([0.]*38)
        with self.assertRaises(ValueError): d.interval39([float('nan')]*39)

    def test_holm_144_independent_numpy(self):
        self.assertEqual(len(d.family()),144); self.assertEqual(len(set(d.family())),144)
        p=np.array([.0001,.001,.02]+[1.]*141)
        order=np.argsort(p,kind='stable'); expected=np.ones(144)
        expected[order]=np.minimum(1,np.maximum.accumulate(p[order]*np.arange(144,0,-1)))
        np.testing.assert_array_equal(d.holm(p),expected)
        with self.assertRaises(ValueError): d.holm([1.]*143)

    def test_bounded_score(self):
        self.assertEqual(d.bounded_mse(8,3),3); self.assertEqual(d.bounded_mse(1,3),1)
        for x in [-1,float('inf'),float('nan')]:
            with self.assertRaises(ValueError): d.bounded_mse(x,3)

    def test_panel_choice_ties_and_unqualified(self):
        row=d.select_estimator('baf',{c:[.8]*39 for c in d.CELLS})
        self.assertEqual(row['cell'],'ratio_batch1'); self.assertFalse(row['protected_results_inspected'])
        with self.assertRaises(ValueError): d.select_estimator('ieee_cis',{c:[.8]*39 for c in d.CELLS})
        self.assertEqual(d.FIGURE['width_cm'],8.8)

    def test_utility_loss_not_clamped_and_interval(self):
        row=d.utility_interval(self.values(.6),self.values(.5))
        self.assertLess(row['figure_loss_mean'],0); self.assertEqual(row['df'],10)
        self.assertEqual(row['figure_loss_interval'],[-row['delta_interval'][1],-row['delta_interval'][0]])

    def test_firewall_overlap_duplicate_missing_namespace(self):
        rows=[[i] for i in range(39)]
        self.assertTrue(fw.assert_fresh('baf',rows,[100],range(200)))
        with self.assertRaises(ValueError): fw.assert_fresh('baf',rows,[1],range(200))
        with self.assertRaises(ValueError): fw.assert_fresh('baf',[[0]]*39,[],range(200))
        with self.assertRaises(ValueError): fw.assert_fresh('baf',rows[:-1],[],range(200))
        with self.assertRaises(ValueError): fw.sources({'targets':[{'source_ids':[1]}]},'ambiguous.json')

    def test_firewall_actual_sealed_p24_35(self):
        audit=fw.audit(); self.assertEqual(audit['new_real_target_allocations'],0)
        self.assertTrue(any('priority35a/qualification/reservations/' in r['path'] for r in audit['inventory']))
        for dataset in d.DATASETS:
            doc=json.loads((fw.previous.ROOT/f'artifacts/priority35a/qualification/reservations/{dataset}.json').read_text())
            ids=fw.previous.extract_ids(doc)
            self.assertTrue(set(ids)<=set(audit['exclusions'][dataset]))

    def test_fail_closed_bn_and_threads(self):
        from experiments import priority34a_local_bn as old
        old.torch.set_num_threads(1)
        model=old.p.FraudMLP(13).cpu(); old.guard(model,'synthetic valid')
        bn=next(m for m in model.modules() if isinstance(m,old.torch.nn.BatchNorm1d))
        bn.running_var.fill_(-1)
        with self.assertRaises(ValueError):old.guard(model,'synthetic negative')
        bn.running_var.fill_(float('nan'))
        with self.assertRaises(ValueError):old.guard(model,'synthetic nonfinite')
        self.assertEqual(old.torch.get_num_threads(),1)

    def test_per_tensor_joint_sensitivity_and_inward_rounding(self):
        from experiments import priority34b_local_dp as old
        from collections import OrderedDict
        state=OrderedDict(a=old.torch.tensor([3.,4.]),b=old.torch.tensor([12.]))
        for mechanism in d.CLIPS:
            clipped, receipt=old.ORIGINAL_CLIP(state,mechanism)
            self.assertLessEqual(receipt['after_norm'],.01)
            expected=.01/math.sqrt(2) if mechanism=='per_tensor' else .01
            self.assertEqual(receipt['tensor_radius'],expected)
            if mechanism=='per_tensor':
                self.assertTrue(all(float(x.double().norm())<=expected for x in clipped.values()))
            self.assertEqual(d.account(10)['noise_sd'],old.account(10)['noise_sd'])

    def test_all_sign_count_pairs_and_order_coverage(self):
        for n in range(1,40):
            for wins in range(n+1):
                self.assertAlmostEqual(d.exact_tail(wins,n-wins),
                    binomtest(wins,n,.5,alternative='greater').pvalue,places=15)
        coverage=1-2*sum(math.comb(39,k) for k in range(13))/2**39
        self.assertGreaterEqual(coverage,.95)

    def test_selected_panel_invalid_individual_value(self):
        values={c:[.8]*39 for c in d.CELLS};values['ratio_batch1'][0]=1.1
        with self.assertRaises(ValueError):d.select_estimator('baf',values)
        with self.assertRaises(ValueError):d.f1_values({float(s):.5 for s in d.SEEDS})


if __name__=='__main__':unittest.main()
