"""Prospective configuration/provenance freeze, no benchmark measurements."""
import json
import platform
import subprocess
import sys
from pathlib import Path
from experiments.priority36b_memory import ROOT,OUT,DATASETS,METHODS,sha,write,event
from experiments.priority36b_finish import compile_sources

def main():
    if (OUT/"freeze.json").exists(): raise FileExistsError("do not replace freeze")
    original=ROOT/"artifacts/priority32_multidataset/repair_trainable_raw_bn_20261002/rq3"
    inputs=[ROOT/"protocols/amendments/2026-10-12_priority36b_rss_supplement.md",ROOT/"protocols/config/rq3_v2_benchmark.json",ROOT/"artifacts/priority32_multidataset/data_audit.json"]
    inputs += [ROOT/p for p in ["reports/rq3_optimized_implementation_report_20260916.md","reports/priority32_repaired_trainable_rq2_rq3_report_20261002.md"]]
    for dataset,dim in [("paysim",13),("ieee_cis",476),("baf",58)]:
        source=original/dataset/"config.json"
        cfg=json.loads(source.read_text())
        assert cfg["benchmark"]["input_dim"]==dim
        assert cfg["benchmark"]["methods"]==(["RAW_FLOAT32","DP_CLIP_NOISE"] if dataset=="paysim" else METHODS)
        cfg["benchmark"]["methods"]=METHODS
        cfg["benchmark"]["cells"]=36
        assert cfg["benchmark"]["warmup_repetitions"]==10 and cfg["benchmark"]["measured_repetitions"]==50
        write(OUT/"configs"/(dataset+".json"),cfg)
        # Semantic equality to the shape-matched original, including seeds/network.
        assert cfg==json.loads((OUT/"configs"/(dataset+".json")).read_text())
        inputs += [source,original/dataset/"analysis/acceptance_decision.json",OUT/"configs"/(dataset+".json")]
    test=subprocess.run([sys.executable,"-B","-m","pytest","-q","-p","no:cacheprovider","tests/test_priority36b_memory.py"],cwd=ROOT,capture_output=True,text=True)
    write(OUT/"preflight_tests.json",{"returncode":test.returncode,"stdout":test.stdout,"stderr":test.stderr,"command":test.args})
    assert test.returncode==0
    compile_sources()
    diff=subprocess.run(["git","diff","--check"],cwd=ROOT,capture_output=True,text=True)
    write(OUT/"preflight_diff.json",{"returncode":diff.returncode,"stdout":diff.stdout,"stderr":diff.stderr}); assert diff.returncode==0
    # Imported local dependency closure plus all newly written analysis/tests.
    sources={Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,"__file__",None) and str(m.__file__).endswith(".py") and Path(m.__file__).resolve().is_relative_to(ROOT) and ".venv" not in str(m.__file__)}
    sources.update((ROOT/"experiments").glob("priority36b_*.py")); sources.update((ROOT/"tests").glob("test_priority36b*.py"))
    write(OUT/"environment.json",{"platform":platform.platform(),"python":sys.version,"cpu_only":True,"threads_per_process":1,"ru_maxrss_units":"bytes" if sys.platform=="darwin" else "KiB"})
    inputs.append(OUT/"environment.json")
    inputs.append(ROOT/"results/rq3/optimized_implementation_20260916/acceptance_decision.json")
    freeze={"sources":{str(p.relative_to(ROOT)):sha(p) for p in sorted(sources)},"inputs":{str(p.relative_to(ROOT)):sha(p) for p in sorted(set(inputs))}}
    write(OUT/"freeze.json",freeze)
    write(OUT/"checklist.json",{"amendment":"DONE","implementation":"DONE","preflight_freeze":"DONE","measurements":"PENDING","independent_csv_verification":"PENDING","report_manifest":"PENDING"})
    event("PREFLIGHT_FROZEN",freeze_sha256=sha(OUT/"freeze.json"),tests=10,cells=108,phase_observations=12960)
    print(json.dumps({"freeze_sha256":sha(OUT/"freeze.json"),"sources":len(sources),"inputs":len(inputs),"tests":10}))

if __name__=="__main__": main()
