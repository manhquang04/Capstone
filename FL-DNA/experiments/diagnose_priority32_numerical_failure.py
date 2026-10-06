"""Faithful validation-only diagnostic replay; never repairs model tensors."""
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import priority32_multidataset as p


def main():
    destination = p.OUT / "diagnosis_20261002"
    destination.mkdir(exist_ok=True)
    result = destination / "baf_v1_seed321001.json"
    evidence_path = destination / "evidence.json"
    if evidence_path.exists():
        raise RuntimeError("diagnostic already recorded; do not silently replay")
    freeze = json.loads((p.OUT / "execution_freeze.json").read_text())
    cfg = p.job_config("baf", "dna_v1_conservative", 321001,
                       freeze["chosen"]["baf"]["training"])
    cfg["quality_validation_only"] = True  # forbids any test evaluation
    original_average = p.fed_avg
    original_probabilities = p.probabilities
    records, activations = [], []

    def inspect_average(states, counts):
        state = original_average(states, counts)
        records.append({"round": len(records) + 1, "running_var": {
            key: {"min": float(value.min()), "negative": int((value < 0).sum()),
                  "finite": bool(p.torch.isfinite(value).all())}
            for key, value in state.items() if key.endswith("running_var")}})
        return state

    def inspect_validation(model, array):
        p.torch.save(model.state_dict(), destination / "final_state.pt")
        hooks = []
        def hook(name):
            def capture(module, inputs, output):
                if len(activations) < len(model.network):
                    activations.append({"layer": name, "finite_input": bool(p.torch.isfinite(inputs[0]).all()),
                                        "finite_output": bool(p.torch.isfinite(output).all())})
            return capture
        for name, module in model.network.named_children():
            hooks.append(module.register_forward_hook(hook("network." + name)))
        try:
            return original_probabilities(model, array)
        finally:
            for handle in hooks:
                handle.remove()

    p.fed_avg, p.probabilities = inspect_average, inspect_validation
    p.event("diagnostic_replay_started", config=cfg, output=str(destination),
            amendment="protocols/amendments/2026-10-02_priority32_numerical_failure_diagnosis.md")
    error = None
    try:
        p.train_job(cfg, str(result))
    except Exception:
        error = traceback.format_exc()
    finally:
        p.fed_avg, p.probabilities = original_average, original_probabilities
    p.write(evidence_path, {"diagnostic_only": True, "config": cfg,
                           "rounds": records, "validation_layers": activations,
                           "exception": error, "finished_at": p.now(),
                           "torch_threads": p.torch.get_num_threads()})
    paths = [evidence_path, destination / "final_state.pt", result.with_suffix(".rounds.jsonl"),
             Path(__file__), p.ROOT / "protocols/amendments/2026-10-02_priority32_numerical_failure_diagnosis.md"]
    p.write(destination / "hashes.json", {str(path): p.sha(path) for path in paths if path.exists()})
    p.event("diagnostic_replay_finished", exception=error, evidence=str(evidence_path))


if __name__ == "__main__":
    main()
