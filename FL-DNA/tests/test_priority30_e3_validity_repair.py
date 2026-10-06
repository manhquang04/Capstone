"""Fail-closed utility gate and bracketing tests, synthetic inputs only."""
import importlib.util
from pathlib import Path

def module():
    path=Path(__file__).resolve().parents[1]/'experiments/priority30_native_defenses/repair_e3_validity.py'
    spec=importlib.util.spec_from_file_location('repair',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod.torch.set_num_threads(1)
    return mod

def test_majority_classifier_is_rejected():
    m=module()
    assert not m.baseline_gate([{'validation_accuracy':.754316,'test_accuracy':.754316}]*16,.754316,.754316)['passed']
    assert m.baseline_gate([{'validation_accuracy':.85,'test_accuracy':.85}]*16,.754316,.754316)['passed']
    assert not m.baseline_gate([{'validation_accuracy':.81,'test_accuracy':.81}]*16,.79,.79)['passed']

def test_bracketing_requires_upper_failure():
    m=module();grid=[.001,.01,.1]
    assert m.select_bracket(grid,{.001:0,.01:-.01,.1:-.08},-.02)==(.01,True)
    assert m.select_bracket(grid,{s:0 for s in grid},-.02)==(.1,False)
    assert m.select_bracket(grid,{s:-.1 for s in grid},-.02)==(None,False)

def test_exact_sign_and_global_holm():
    module()
    from experiments.priority30_native_defenses.analyze_s0u_pairs import exact_one_sided_p, holm_adjust
    from scipy.stats import binom
    for wins,losses in [(39,0),(37,2),(2,37),(16,23)]:
        assert abs(exact_one_sided_p(wins,losses)-binom.sf(wins-1,wins+losses,.5))<1e-15
    assert holm_adjust([('a',.001),('b',.04),('c',.06)])=={'a':.003,'b':.08,'c':.08}
