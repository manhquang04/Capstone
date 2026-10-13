import numpy as np
import pytest
from collections import OrderedDict
from experiments.priority36b_memory import rss_bytes,finite,prepare_client,bench
from experiments.priority36b_statistics import independent_ci,summary
from experiments.analyze_rq3_benchmark import _bootstrap_p95_ci

def test_units():
    assert rss_bytes(1024,"darwin")==1024
    assert rss_bytes(1024,"linux")==1048576
    with pytest.raises(ValueError): rss_bytes(1,"win32")

def test_failclosed():
    with pytest.raises(ValueError): finite([{"a":np.array([np.nan])}])
    with pytest.raises(ValueError): finite([{"bn.running_var":np.array([-1.])}])

@pytest.mark.parametrize("seed",[0,271828,9999])
def test_bootstrap(seed):
    x=np.random.default_rng(seed).normal(size=50)
    assert np.allclose(independent_ci(x,seed),_bootstrap_p95_ci(x,seed=seed,samples=10000),rtol=1e-12,atol=1e-12)
    assert summary(np.ones(50),seed)["upper"]==1

@pytest.mark.parametrize("method",["RAW_FLOAT32","DNA_TRANSFORM_TRANSPORT","DNA_TRANSFORM_V2_TRANSPORT","DP_CLIP_NOISE"])
def test_roundtrip(method):
    import json
    from pathlib import Path
    cfg=json.loads(Path("protocols/config/rq3_v2_benchmark.json").read_text())
    result=prepare_client(cfg,method,3,17)()
    decoded=[bench._deserialize_v2(p) if method=="DNA_TRANSFORM_V2_TRANSPORT" else bench._deserialize_binary(p) for p in result["payloads"]]
    assert bench._equal_bits(bench._aggregate(decoded),bench._aggregate(result["expected"]))

def test_default_fixture_identity():
    import json
    from pathlib import Path
    cfg=json.loads(Path("protocols/config/rq3_v2_benchmark.json").read_text())
    updates=bench._input_updates(3,13,1618033)
    for method in ["RAW_FLOAT32","DNA_TRANSFORM_TRANSPORT","DNA_TRANSFORM_V2_TRANSPORT"]:
        got=prepare_client(cfg,method,3,17)()
        for i,state in enumerate(updates):
            seed=bench._seed(1184685071,i,17)
            if method=="RAW_FLOAT32": payload=bench._serialize_binary(state)
            elif method=="DNA_TRANSFORM_TRANSPORT": payload=bench._serialize_binary(bench._transform(state,bench.DNATransformConfig(**cfg["dna_transform"]),seed))
            else: payload,_=bench._serialize_v2(state,bench.DNATransformV2Config(**cfg["dna_transform_v2"],seed=seed),17)
            assert got["payloads"][i]==payload
