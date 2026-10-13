"""Original bootstrap and independent sorted-interpolation cross-check."""
import numpy as np
from experiments.analyze_rq3_benchmark import _bootstrap_p95_ci

def quantile_sorted(values, q, axis=-1):
    ordered = np.sort(values, axis=axis)
    h = (ordered.shape[axis]-1)*q
    lo, hi = int(np.floor(h)), int(np.ceil(h))
    return np.take(ordered, lo, axis=axis)*(1-(h-lo)) + np.take(ordered, hi, axis=axis)*(h-lo)

def independent_ci(values, seed, samples=10000):
    indices = np.random.default_rng(seed).integers(0,len(values),size=(samples,len(values)))
    estimates = quantile_sorted(np.asarray(values)[indices], .95, axis=1)
    return float(quantile_sorted(estimates,.025)), float(quantile_sorted(estimates,.975))

def summary(values, seed, independent=False):
    values = np.asarray(values,dtype=np.float64)
    if len(values)!=50 or not np.isfinite(values).all():
        raise ValueError("expected 50 finite paired observations")
    ci = independent_ci(values,seed) if independent else _bootstrap_p95_ci(values,seed=seed,samples=10000)
    return {"mean":float(np.mean(values)), "median":float(np.median(values)),
            "p95":float(quantile_sorted(values,.95) if independent else np.quantile(values,.95)),
            "lower":ci[0],"upper":ci[1]}

def cell_summary(rows, raw, profile, method, count, phase, measure, independent=False):
    field = {"RSS":"rss_peak_bytes","tracemalloc":"trace_peak_bytes","psutil":"psutil_peak_bytes"}[measure]
    a=np.array([r[phase][field] for r in rows],dtype=np.float64)
    b=np.array([r[phase][field] for r in raw],dtype=np.float64)
    base=int.from_bytes(f"{profile}|{method}|{count}".encode(),"little")%(2**32)
    offsets=(8,3,4) if phase=="client" else (9,5,6)
    result={}
    for name,vector,offset in zip(["absolute_mb","increase_mb","relative"],[a/1024**2,(a-b)/1024**2,(a-b)/np.maximum(b,1)],offsets):
        result.update({name+"_"+k:v for k,v in summary(vector,base+offset,independent).items()})
    result["absolute_limit_mb"]=512 if phase=="client" else 1024
    result["absolute_pass"]=result["absolute_mb_upper"]<=result["absolute_limit_mb"]
    result["relative_pass"]=result["relative_upper"]<=.25
    result["memory_pass"]=result["absolute_pass"] and result["relative_pass"]
    return result
