"""Strip unused encoder diagnostics; replay the same fixed BN recovery without new training."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from experiments import priority33a_bn_mean as a
from experiments.priority33a_audit import OUT, write, append, sha, now

torch.set_num_threads(1)


def server_receipt(original):
    if original["kind"] == "v1":
        return dict(kind="v1", q=original["q"])
    if original["kind"] == "v2":
        assert isinstance(original["metadata"], a.v2.DNATransformV2Metadata)
        return dict(kind="v2", q=original["q"], metadata=original["metadata"])
    raise ValueError("unknown server receipt; no fallback")


def recover_server_payload(model, receipt):
    if receipt["kind"] == "v1":
        assert set(receipt) == {"kind", "q"}, "v1 must not expose encoder diagnostics"
    elif receipt["kind"] == "v2":
        assert set(receipt) == {"kind", "q", "metadata"}
        assert isinstance(receipt["metadata"], a.v2.DNATransformV2Metadata)
    else:
        raise ValueError("unknown server receipt; no fallback")
    return a.recover_payload(model, receipt)


def run():
    outcomes = json.loads((OUT / "A2_COMPLETE.json").read_text())["datasets"]
    records = []
    for dataset, outcome in outcomes.items():
        if outcome["status"] != "CONFIRMATORY_COMPLETE":
            continue
        model = a.load_model(dataset)
        _, std = a.population(dataset)
        captures = torch.load(OUT / "A2" / dataset / "n39/captures.pt", map_location="cpu")
        for method in a.METHODS:
            for i, capture in enumerate(captures):
                original_folder = OUT / "A2" / dataset / "confirmatory" / method / f"target_{i:02d}"
                folder = OUT / "receipt_contract_replay" / dataset / method / f"target_{i:02d}"
                old = json.loads((original_folder / "result.json").read_text())
                result_path = folder / "result.json"
                if result_path.exists():
                    doc = json.loads(result_path.read_text())
                    assert doc["source_ids"] == old["source_ids"] and doc["original_payload_sha256"] == sha(original_folder / "transmitted_payload.pt")
                    assert doc["server_receipt_sha256"] == sha(folder / "server_observable_receipt.pt")
                else:
                    original = torch.load(original_folder / "transmitted_payload.pt", map_location="cpu")
                    received = server_receipt(original)
                    recovered = recover_server_payload(model, received)
                    value = a.mse_std(recovered, capture["true_mean"], std)
                    assert np.isfinite(value) and np.isclose(value, old["defended_mse"], rtol=1e-8, atol=1e-10)
                    for comparison in old["comparators"].values():
                        if comparison["status"] == "VALID":
                            assert np.sign(value-comparison["dp_mse"]) == np.sign(old["defended_mse"]-comparison["dp_mse"])
                    folder.mkdir(parents=True, exist_ok=True)
                    torch.save(received, folder / "server_observable_receipt.pt")
                    doc = dict(dataset=dataset, method=method, target=i, source_ids=old["source_ids"], status="COMPLETED",
                        defended_mse=value, original_defended_mse=old["defended_mse"], absolute_difference=abs(value-old["defended_mse"]),
                        original_payload_sha256=sha(original_folder / "transmitted_payload.pt"), server_receipt_sha256=sha(folder / "server_observable_receipt.pt"),
                        comparators=old["comparators"], received_keys=sorted(received), device="cpu", torch_threads=1,
                        unused_encoder_diagnostics_removed=method == a.METHODS[0], original_scores_superseded=False, original_v1_receipt_contract_superseded=method == a.METHODS[0])
                    write(result_path, doc)
                records.append(doc)
                progress = dict(stage="server_receipt_contract_replay", dataset=dataset, method=method, done=len(records), total=78, failed=0,
                    started_at=None, last_update=now(), eta_minutes=None)
                write(OUT / "progress.json", progress)
                append(OUT / "progress.log", progress)
    write(OUT / "RECEIPT_CONTRACT_COMPLETE.json", dict(at=now(), jobs=len(records), max_absolute_score_difference=max((r["absolute_difference"] for r in records), default=0),
        all_pairwise_signs_unchanged=True, new_training_jobs=0, code_sha256=sha(__file__), records=records))
    append(OUT / "runs.jsonl", dict(at=now(), event="server_receipt_contract_replay_complete", jobs=len(records), new_training_jobs=0,
        command=[sys.executable, "-B", str(Path(__file__).resolve())]))


if __name__ == "__main__":
    run()
