"""Final P33b byte, pairing, metric, source/firewall and statistics checks."""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.priority33b_core import *
from experiments.priority33b_qualify import checklist
from experiments.analyze_priority33b import analyze,report


def main():
    checklist(5,"in_progress")
    statistics,gates=analyze()
    for name in ("execution_freeze.json","comparator_execution_freeze.json"):
        frozen=json.loads((OUT/name).read_text())
        sources=frozen.get("source_sha256",frozen.get("sources"))
        for path,digest in sources.items():assert sha(ROOT/path)==digest,f"frozen source changed: {path}"
    reloaded=0
    for dataset in ("ieee_cis","baf"):
        adapter=json.loads((OUT/dataset/"adapter.json").read_text())
        for name,digest in adapter["prepared_sha256"].items():assert sha(PREPARED/dataset/name)==digest
        for name,digest in adapter["raw_sha256"].items():assert sha(ROOT/name)==digest
        manifest=json.loads((OUT/dataset/"targets.json").read_text())
        ids=[i for batches in manifest["source_ids"].values() for batch in batches for i in batch]
        assert len(ids)==len(set(ids))==568
        assert sha(OUT/dataset/"targets.json")==sha(OUT.parent/dataset/"targets.json")
        data=PreparedDataset(dataset)
        for path in (OUT/dataset).glob("**/result.json"):
            doc=json.loads(path.read_text())
            if doc["status"]!="COMPLETED" or "accuracy" not in doc:continue
            folder=path.parent
            truth=torch.load(folder/"scoring_truth.pt",map_location="cpu",weights_only=False)
            rec=torch.load(folder/"reconstruction.pt",map_location="cpu",weights_only=False)
            calculated=measure(data,truth["truth"],rec["reconstruction"])
            assert abs(calculated["accuracy_percent"]-doc["accuracy"]["accuracy_percent"])<1e-10
            assert truth["source_ids"]==doc["source_ids"]
            receipt=torch.load(folder/"server_receipt.pt",map_location="cpu",weights_only=False)
            assert "truth" not in receipt
            for key in ("payload","gradient"):
                if key in receipt:
                    for value in receipt[key]:finite(value,"reloaded receipt")
            assert len(rec["ensemble"])==len(rec["objective_losses"] if "objective_losses" in rec else rec["objectives"])==30
            reloaded+=1
    report_path=report(statistics,gates)
    tests=subprocess.run([sys.executable,"-B","-m","unittest","tests.test_priority33b","tests.test_priority33b_comparators","-v"],cwd=ROOT,capture_output=True,text=True)
    assert tests.returncode==0,tests.stderr
    source_files=[*ROOT.glob("experiments/*priority33b*.py"),*ROOT.glob("tests/*priority33b*.py")]
    compiled=subprocess.run([sys.executable,"-B","-m","py_compile",*[str(path) for path in source_files]],cwd=ROOT,capture_output=True,text=True)
    assert compiled.returncode==0,compiled.stderr
    diff=subprocess.run(["git","diff","--check"],cwd=ROOT,capture_output=True,text=True)
    assert diff.returncode==0,diff.stdout+diff.stderr
    check=dict(at=now(),py_compile="PASS",git_diff_check="PASS",unit_tests="PASS",unit_test_output=tests.stderr,
        frozen_sources="PASS",raw_prepared_data="PASS",source_pairing="PASS",reloaded_metrics=reloaded,
        independent_statistics="PASS",family_size=16)
    write(OUT/"final_checks.json",check)
    checklist(5,"completed")
    status="complete_with_explicit_NOT_ASSESSABLE" if any(t["status"]!="ASSESSABLE" for t in statistics["tests"]) else "complete"
    progress=dict(stage=status,dataset="all",method="all",done=reloaded,total=reloaded,failed=0,last_update=now(),eta_minutes=0)
    for folder in (OUT,OUT.parent):
        write(folder/"progress.json",progress)
        append(folder/"progress.log",progress)
    files=[path for path in OUT.parent.rglob("*") if path.is_file() and path.name not in ("sha256_manifest.json","COMPLETE.json")]
    files += source_files + list((ROOT/"protocols/amendments").glob("*priority33b*.md"))+[report_path]
    hashes={str(path.relative_to(ROOT)):sha(path) for path in sorted(set(files))}
    write(OUT.parent/"sha256_manifest.json",dict(at=now(),hashes=hashes))
    for name,digest in hashes.items():assert sha(ROOT/name)==digest
    write(OUT.parent/"COMPLETE.json",dict(at=now(),report=str(report_path.relative_to(ROOT)),report_sha256=sha(report_path),
        manifest_sha256=sha(OUT.parent/"sha256_manifest.json"),status=status,independently_verified=True))
    print(json.dumps(check))


if __name__=="__main__":main()
