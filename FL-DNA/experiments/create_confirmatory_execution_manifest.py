"""Create a compact integrity manifest for completed RQ1/RQ2 execution."""
from __future__ import annotations
import hashlib,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def tree(path):
    h=hashlib.sha256(); files=sorted(p for p in path.rglob("*") if p.is_file()); total=0
    for p in files:
        rel=str(p.relative_to(path)); digest=sha(p); size=p.stat().st_size; total+=size
        h.update(rel.encode()); h.update(b"\0"); h.update(str(size).encode()); h.update(b"\0"); h.update(digest.encode()); h.update(b"\n")
    return {"path":str(path.relative_to(ROOT)),"file_count":len(files),"bytes":total,"tree_sha256":h.hexdigest()}

def main():
    key=[ROOT.parent/"PROJECT.md",ROOT/"protocols/confirmatory_freeze_manifest_2026-09-13.json",ROOT/"protocols/confirmatory_execution_report_2026-09-13.md",ROOT/"protocols/amendments/2026-09-13_rq2_multiplicity_and_execution_authorization.md",ROOT/"protocols/config/rq1_confirmatory.yaml",ROOT/"protocols/config/rq2_confirmatory.yaml",ROOT/"artifacts/rq1/confirmatory_freeze_20260913/rq1_confirmatory_targets.pt",ROOT/"artifacts/rq2/confirmatory_freeze_20260913/per_seed_contract.json",ROOT/"artifacts/rq1/confirmatory_run_20260913/execution_summary.json",ROOT/"artifacts/rq2/confirmatory_run_20260913/execution_summary_rebuilt.json",ROOT/"results/rq1/confirmatory_20260913/rq1_summary.json",ROOT/"results/rq1/confirmatory_20260913/rq1_secondary_holm.json",ROOT/"results/rq2/confirmatory_20260913/rq2_summary.json"]
    def label(p):
        try: return str(p.relative_to(ROOT))
        except ValueError: return str(p.relative_to(ROOT.parent))
    report={"schema_version":1,"created_at":datetime.now(timezone.utc).isoformat(),"scope":"completed single RQ1/RQ2 confirmatory execution","rq3_status":"HELD","git_head_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"key_files":[{"path":label(p),"bytes":p.stat().st_size,"sha256":sha(p)} for p in key],"artifact_trees":[tree(ROOT/"artifacts/rq1/confirmatory_run_20260913"),tree(ROOT/"artifacts/rq2/confirmatory_run_20260913"),tree(ROOT/"results/rq1/confirmatory_20260913"),tree(ROOT/"results/rq2/confirmatory_20260913")]}
    out=ROOT/"protocols/confirmatory_execution_manifest_2026-09-13.json"; out.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
