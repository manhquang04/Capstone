"""Post-hoc audit-only adapter; preserves every frozen source/measurement."""
import json
import math
from pathlib import Path
import numpy as np
from experiments import priority35b_leakage_metrics_audit as metric
from experiments import priority35b_leakage_verified_preparation as wrapper

ROOT=wrapper.prepare.ROOT
FREEZE='artifacts/priority35b/b3/float64_verifier_infrastructure_freeze.json'


def tolerance(stored,independent,n):
    if type(n) is not int or n<1:raise ValueError('encoded scalar count')
    u=float(np.finfo(np.float64).eps)
    if n*u>=1 or not all(math.isfinite(x) and x>=0 for x in (stored,independent)):
        raise ValueError('finite nonnegative MSE and valid rounding bound')
    return 1e-12+2*(n*u/(1-n*u))*max(stored,independent)


def check_native(stored,native):
    if any(stored[key]!=native[key] for key in ('feature_accuracy','standardized_mse','assignment')):
        raise ValueError('stored score must equal native recomputation exactly')


def install():
    from experiments import priority35b_leakage as driver
    original_score=metric.independently_score;original_verify=metric.verify_score
    evidence={}
    def score(data,truth,recovered):
        result=original_score(data,truth,recovered)
        evidence[id(result)]=(truth.numel(),driver.metric(data,truth,recovered),result)
        return result
    def verify(stored,recomputed,prior_mse,mse_tolerance=1e-12):
        if mse_tolerance!=1e-12:raise ValueError('unexpected caller tolerance override')
        n,native,_=evidence.pop(id(recomputed))
        check_native(stored,native)
        bound=tolerance(stored['standardized_mse'],recomputed['standardized_mse'],n)
        return original_verify(stored,recomputed,prior_mse,mse_tolerance=bound)
    metric.independently_score=score;metric.verify_score=verify


def run():
    seal=wrapper.prepare.integrity.read(Path(ROOT)/FREEZE)
    for path,digest in seal['files'].items():
        if wrapper.prepare.integrity.sha(wrapper.prepare.integrity.child(ROOT,path))!=digest:
            raise ValueError('float64 adapter freeze drift: '+path)
    wrapper.prepare.no_live(ROOT);wrapper.verify_sources(ROOT)
    install()
    return wrapper.run(ROOT)


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
