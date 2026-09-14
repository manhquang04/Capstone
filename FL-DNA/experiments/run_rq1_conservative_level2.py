"""Run realization-known Level 2 on the frozen conservative n=39 target."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from dna_encoder.transform_defense import DNATransformConfig
from experiments import fraud_fl_common as common
from experiments.run_phase3_adam_ladder import capture
from experiments.run_phase3_full_client import checksum, dump
from experiments.run_phase4_dna_level2_direct_inversion import _evaluate_group_with_recovered_signal, _gate, _recover_state, _write_csv
from privacy.seed_manager import derive_seed

TARGET=ROOT/"artifacts/rq1/confirmatory_freeze_20260913/rq1_confirmatory_targets.pt"
TARGET_SHA="caee0024c318c2874037382996b843ead8024b02b132185e502e245d3266267f"
RAW_ROOT=ROOT/"artifacts/rq1/confirmatory_run_20260913/raw"
DNA_RUN_SEED=1184685071


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",type=Path,default=ROOT/"artifacts/rq1/group2_conservative_level2_20260913"); args=parser.parse_args()
    import hashlib
    if hashlib.sha256(TARGET.read_bytes()).hexdigest()!=TARGET_SHA: raise RuntimeError("conservative target hash mismatch")
    args.output_dir.mkdir(parents=True,exist_ok=False); torch.set_num_threads(1); common.DEVICE=torch.device("cpu")
    groups=torch.load(TARGET,map_location="cpu",weights_only=False); selected=[]; transform_rows=[]; block_rows=[]
    for group_id,group in enumerate(groups):
        group_run=RAW_ROOT/f"group_{group_id}_run"; baseline=json.loads((group_run/"baseline_gate.json").read_text()); report_dir=next(group_run.glob("harddiff_reparam_*")); report=json.loads((report_dir/"harddiff_reparam_report.json").read_text()); restarts=report["restarts"]
        local_seed=derive_seed(baseline["protocol"]["run_seed"],"local",group_id,baseline["protocol"]["batch_size"])
        model,criterion,_,y,batches,rng,observed=capture(group,local_seed,baseline["protocol"]["batch_size"])
        config=DNATransformConfig(256,0.08,0.88,0.45,derive_seed(DNA_RUN_SEED,"phase4-dna-transform",TARGET.name,group_id))
        recovered,tensor_rows,matrix_rows=_recover_state(observed,config)
        transform_rows.extend({"group_id":group_id,**row} for row in tensor_rows); block_rows.extend({"group_id":group_id,**row} for row in matrix_rows)
        selected.append(_evaluate_group_with_recovered_signal(report_dir,group,group_id,restarts,model,criterion,y,batches,rng,observed,recovered))
    errors=[row["relative_l2_recovery_error"] for row in transform_rows]; conditions=[row["condition_number"] for row in block_rows]
    result={"target":str(TARGET),"target_sha256":TARGET_SHA,"groups":len(groups),"attacker_level":"Level 2 realization-known direct linear inversion","dna_run_seed":DNA_RUN_SEED,"gate":_gate(selected),"recovery":{"tensors":len(transform_rows),"blocks":len(block_rows),"full_rank_blocks":sum(row["rank"]==row["block_size"] for row in block_rows),"max_condition_number":max(conditions),"median_condition_number":float(np.median(conditions)),"max_relative_l2_recovery_error":max(errors),"mean_relative_l2_recovery_error":float(np.mean(errors)),"max_abs_recovery_error":max(row["max_abs_recovery_error"] for row in transform_rows)},"selected":selected,"dataset_sha256":checksum(ROOT/"datasets/creditcard.csv"),"interpretation":"attacker-favorable exact-realization sensitivity; does not replace Level 1"}
    dump(args.output_dir/"level2_report.json",result); _write_csv(args.output_dir/"selected.csv",selected); _write_csv(args.output_dir/"transform_stats.csv",transform_rows); _write_csv(args.output_dir/"block_stats.csv",block_rows); print(json.dumps(result,indent=2))


if __name__=="__main__": main()
