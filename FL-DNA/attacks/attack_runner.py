"""Run gradient inversion attack comparisons for PaySim FL defenses."""

from __future__ import annotations

import copy
import csv
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attacks.gradient_inversion import (
    GradientInversionConfig,
    gradient_inversion_attack,
    parameter_gradients,
)
from attacks.inversion_metrics import reconstruction_metrics
from attacks.pseudo_image import save_pseudo_image
from data.load_creditcard import load_creditcard_data
from dna_encoder.encoder import DNAEncoder
from dna_encoder.transform_defense import DNATransformConfig, transform_update_array
from experiments.fraud_fl_common import (
    BATCH_SIZE,
    LOCAL_EPOCHS,
    NUM_CLIENTS,
    RANDOM_SEED,
    build_loss,
    fed_avg,
    set_random_seed,
    train_local_model,
)
from models.fraud_mlp import FraudMLP

OUTPUT_DIR = Path(
    os.environ.get(
        "ATTACK_OUTPUT_DIR",
        str(PROJECT_ROOT / "artifacts" / "gradient_inversion"),
    )
)
DETAILS_PATH = OUTPUT_DIR / "metrics_details.json"
SUMMARY_CSV_PATH = OUTPUT_DIR / "metrics_summary.csv"
SUMMARY_JSON_PATH = OUTPUT_DIR / "metrics_summary.json"

ATTACK_ITERATIONS = int(os.environ.get("ATTACK_ITERATIONS", "300"))
ATTACK_NUM_SAMPLES = int(os.environ.get("ATTACK_NUM_SAMPLES", "3"))
ATTACK_WARMUP_ROUNDS = int(os.environ.get("ATTACK_WARMUP_ROUNDS", "3"))
ATTACK_LR = float(os.environ.get("ATTACK_LR", "0.05"))
ATTACK_L2 = float(os.environ.get("ATTACK_L2", "0.0001"))
DP_CLIP_NORM = float(os.environ.get("DP_CLIP_NORM", "100.0"))
DP_NOISE_MULTIPLIER = float(os.environ.get("DP_NOISE_MULTIPLIER", "0.0005"))

DNA_TRANSFORM_CONFIG = DNATransformConfig(
    block_size=int(os.environ.get("DNA_TRANSFORM_BLOCK_SIZE", "256")),
    mix_ratio=float(os.environ.get("DNA_TRANSFORM_MIX", "0.05")),
    keep_ratio=float(os.environ.get("DNA_TRANSFORM_KEEP", "0.90")),
    shrink_factor=float(os.environ.get("DNA_TRANSFORM_SHRINK", "0.50")),
    seed=RANDOM_SEED,
)

RAW_VISIBLE_METHODS = (
    "FL_Baseline",
    "FL_DNA",
    "FL_DP",
    "FL_DNA_TransformDefense",
)
SECURE_AGG_METHODS = (
    "FL_SecureAgg",
    "FL_DNA_SecureAgg",
    "FL_DNA_TransformDefense_SecureAgg",
)


def main() -> None:
    set_random_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client_loaders, validation_loader, _, input_dim, pos_weight, metadata = load_creditcard_data(
        batch_size=BATCH_SIZE,
        num_clients=NUM_CLIENTS,
    )
    model = _train_warmup_model(client_loaders, input_dim, pos_weight)
    criterion = build_loss(pos_weight)
    features, labels = _select_attack_samples(validation_loader, ATTACK_NUM_SAMPLES)
    attack_config = GradientInversionConfig(
        iterations=ATTACK_ITERATIONS,
        learning_rate=ATTACK_LR,
        l2_weight=ATTACK_L2,
        optimizer=os.environ.get("ATTACK_OPTIMIZER", "adam"),
        seed=RANDOM_SEED,
    )

    details: list[dict[str, object]] = []
    dna_encoder = DNAEncoder(key=b"0" * 32)

    print("Gradient Inversion Attack Evaluation")
    print(f"Samples: {len(features)} | iterations: {ATTACK_ITERATIONS} | warmup rounds: {ATTACK_WARMUP_ROUNDS}")
    print(f"DP config: clip_norm={DP_CLIP_NORM}, noise_multiplier={DP_NOISE_MULTIPLIER}")
    print(f"DNA transform config: {DNA_TRANSFORM_CONFIG}")

    for sample_number, (feature, label) in enumerate(zip(features, labels), start=1):
        feature = feature.reshape(1, -1).clone()
        label = label.reshape(1, 1).clone()
        true_gradients = parameter_gradients(model, criterion, feature, label, create_graph=False)
        save_pseudo_image(
            OUTPUT_DIR / f"sample{sample_number:03d}_original.png",
            feature.numpy().reshape(-1),
        )

        for method in RAW_VISIBLE_METHODS:
            observed_gradients, defense_metadata = _apply_observed_gradient_defense(
                method,
                true_gradients,
                dna_encoder,
                sample_number,
            )
            result = gradient_inversion_attack(
                copy.deepcopy(model),
                criterion,
                observed_gradients,
                label,
                input_dim,
                GradientInversionConfig(
                    **{**attack_config.__dict__, "seed": RANDOM_SEED + sample_number}
                ),
            )
            reconstructed = result.reconstructed_input.numpy().reshape(-1)
            metric_values = reconstruction_metrics(feature.numpy().reshape(-1), reconstructed)
            detail = {
                "method": method,
                "sample_id": sample_number,
                "label": float(label.item()),
                "threat_model": "raw_visible_update",
                "attack_applicability": "direct_server_side",
                "best_attack_loss": result.best_loss,
                "loss_history": result.loss_history,
                **metric_values,
                **defense_metadata,
            }
            details.append(detail)
            _save_attack_artifacts(method, sample_number, reconstructed, result.loss_history)
            _print_detail(detail)

        for method in SECURE_AGG_METHODS:
            not_applicable = {
                "method": method,
                "sample_id": sample_number,
                "label": float(label.item()),
                "threat_model": "secure_aggregation_server_side",
                "attack_applicability": "not_directly_applicable",
                "server_sees_individual_raw_updates": False,
                "note": "Server observes only aggregated masked update under the true SecureAgg threat model.",
            }
            details.append(not_applicable)

            leak_method = method.replace("SecureAgg", "PreAggregationLeakage")
            base_method = _secureagg_base_method(method)
            observed_gradients, defense_metadata = _apply_observed_gradient_defense(
                base_method,
                true_gradients,
                dna_encoder,
                sample_number,
            )
            result = gradient_inversion_attack(
                copy.deepcopy(model),
                criterion,
                observed_gradients,
                label,
                input_dim,
                GradientInversionConfig(
                    **{**attack_config.__dict__, "seed": RANDOM_SEED + 10_000 + sample_number}
                ),
            )
            reconstructed = result.reconstructed_input.numpy().reshape(-1)
            metric_values = reconstruction_metrics(feature.numpy().reshape(-1), reconstructed)
            detail = {
                "method": leak_method,
                "sample_id": sample_number,
                "label": float(label.item()),
                "threat_model": "analysis_only_pre_aggregation_leakage",
                "attack_applicability": "upper_bound_if_raw_client_update_leaks",
                "server_sees_individual_raw_updates": True,
                "best_attack_loss": result.best_loss,
                "loss_history": result.loss_history,
                **metric_values,
                **defense_metadata,
            }
            details.append(detail)
            _save_attack_artifacts(leak_method, sample_number, reconstructed, result.loss_history)
            _print_detail(detail)

    summary = _summarize(details)
    _save_outputs(details, summary, metadata.feature_names)
    print(f"Saved attack details: {DETAILS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved attack summary: {SUMMARY_CSV_PATH.relative_to(PROJECT_ROOT)}")


def _train_warmup_model(client_loaders, input_dim: int, pos_weight: torch.Tensor) -> FraudMLP:
    model = FraudMLP(input_dim)
    sample_counts = [len(loader.dataset) for loader in client_loaders]
    for _ in range(ATTACK_WARMUP_ROUNDS):
        local_states = []
        for loader in client_loaders:
            local_model = copy.deepcopy(model)
            train_local_model(local_model, loader, pos_weight, LOCAL_EPOCHS)
            local_states.append(local_model.state_dict())
        model.load_state_dict(fed_avg(local_states, sample_counts))
    return model


def _select_attack_samples(validation_loader, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    features, labels = validation_loader.dataset.tensors
    positive_indices = torch.nonzero(labels.reshape(-1) == 1.0, as_tuple=False).reshape(-1)
    if positive_indices.numel() >= count:
        chosen = positive_indices[:count]
    else:
        fallback = torch.arange(min(count, len(labels)))
        chosen = torch.unique(torch.cat([positive_indices, fallback]))[:count]
    return features[chosen].clone(), labels[chosen].clone()


def _apply_observed_gradient_defense(
    method: str,
    gradients: list[torch.Tensor],
    dna_encoder: DNAEncoder,
    sample_number: int,
) -> tuple[list[torch.Tensor], dict[str, object]]:
    if method in {"FL_Baseline", "FL_SecureAgg"}:
        return [gradient.detach().clone() for gradient in gradients], {
            "defense": "none",
            "server_sees_individual_raw_updates": method == "FL_Baseline",
        }
    if method in {"FL_DNA", "FL_DNA_SecureAgg"}:
        encoded = []
        for gradient in gradients:
            array = gradient.detach().numpy().astype(np.float32, copy=False)
            payload = dna_encoder.encode_array(array)
            restored = dna_encoder.decode_array(payload, array.shape)
            encoded.append(torch.from_numpy(restored.copy()).to(dtype=gradient.dtype))
        return encoded, {
            "defense": "lossless_dna_encode_decode",
            "server_sees_individual_raw_updates": method == "FL_DNA",
        }
    if method == "FL_DP":
        return _apply_dp_to_gradients(gradients, sample_number)
    if method in {"FL_DNA_TransformDefense", "FL_DNA_TransformDefense_SecureAgg"}:
        transformed = []
        stats = []
        for tensor_index, gradient in enumerate(gradients):
            array = gradient.detach().numpy().astype(np.float32, copy=False)
            transformed_array, stat = transform_update_array(
                array,
                DNA_TRANSFORM_CONFIG,
                tensor_index=tensor_index,
            )
            transformed.append(torch.from_numpy(transformed_array.copy()).to(dtype=gradient.dtype))
            stats.append(stat)
        return transformed, {
            "defense": "dna_transform_defense",
            "server_sees_individual_raw_updates": method == "FL_DNA_TransformDefense",
            "dna_transform_relative_l2_delta": float(np.mean([s.relative_l2_delta for s in stats])),
            "dna_transform_cosine_similarity": float(np.mean([s.cosine_similarity for s in stats])),
        }
    raise ValueError(f"Unknown attack method: {method}")


def _apply_dp_to_gradients(
    gradients: list[torch.Tensor],
    sample_number: int,
) -> tuple[list[torch.Tensor], dict[str, object]]:
    flat = torch.cat([gradient.detach().reshape(-1) for gradient in gradients])
    norm = torch.linalg.vector_norm(flat)
    clip_factor = min(1.0, DP_CLIP_NORM / max(float(norm.item()), 1e-12))
    noise_std = DP_NOISE_MULTIPLIER * DP_CLIP_NORM
    generator = torch.Generator().manual_seed(RANDOM_SEED + 2_000 + sample_number)
    defended = []
    for gradient in gradients:
        clipped = gradient.detach() * clip_factor
        noise = torch.randn(clipped.shape, generator=generator, dtype=clipped.dtype) * noise_std
        defended.append(clipped + noise)
    return defended, {
        "defense": "gradient_clipping_gaussian_noise",
        "server_sees_individual_raw_updates": True,
        "dp_clip_norm": DP_CLIP_NORM,
        "dp_noise_multiplier": DP_NOISE_MULTIPLIER,
        "dp_noise_std": noise_std,
        "gradient_norm_before_clip": float(norm.item()),
        "gradient_norm_after_clip": float(norm.item()) * clip_factor,
    }


def _secureagg_base_method(method: str) -> str:
    if method == "FL_SecureAgg":
        return "FL_Baseline"
    if method == "FL_DNA_SecureAgg":
        return "FL_DNA"
    if method == "FL_DNA_TransformDefense_SecureAgg":
        return "FL_DNA_TransformDefense"
    raise ValueError(f"Unknown SecureAgg method: {method}")


def _save_attack_artifacts(
    method: str,
    sample_number: int,
    reconstructed: np.ndarray,
    loss_history: list[float],
) -> None:
    safe_method = method.lower()
    save_pseudo_image(
        OUTPUT_DIR / f"{safe_method}_sample{sample_number:03d}_reconstructed.png",
        reconstructed,
    )
    _save_loss_curve(
        OUTPUT_DIR / f"{safe_method}_sample{sample_number:03d}_loss_curve.png",
        loss_history,
    )


def _save_loss_curve(path: Path, loss_history: list[float]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".mplconfig"))
    import matplotlib.pyplot as plt

    plt.figure(figsize=(4, 2.5))
    plt.plot(loss_history)
    plt.xlabel("iteration")
    plt.ylabel("gradient match loss")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _print_detail(detail: dict[str, object]) -> None:
    print(
        f"{detail['method']} sample={detail['sample_id']} "
        f"mse={detail['mse']:.6f} psnr={detail['psnr']:.3f} "
        f"ssim={detail['ssim']:.3f} cosine={detail['cosine_similarity']:.3f}"
    )


def _summarize(details: list[dict[str, object]]) -> list[dict[str, object]]:
    metric_names = [
        "mse",
        "psnr",
        "ssim",
        "cosine_similarity",
        "pearson_correlation",
        "sign_match_ratio",
        "best_attack_loss",
    ]
    grouped: OrderedDict[str, list[dict[str, object]]] = OrderedDict()
    for detail in details:
        if "mse" not in detail:
            grouped.setdefault(str(detail["method"]), []).append(detail)
            continue
        grouped.setdefault(str(detail["method"]), []).append(detail)

    rows = []
    for method, items in grouped.items():
        numeric_items = [item for item in items if "mse" in item]
        if not numeric_items:
            rows.append(
                {
                    "method": method,
                    "threat_model": items[0]["threat_model"],
                    "attack_applicability": items[0]["attack_applicability"],
                    "samples": len(items),
                    "note": items[0].get("note", ""),
                }
            )
            continue
        row: dict[str, object] = {
            "method": method,
            "threat_model": numeric_items[0]["threat_model"],
            "attack_applicability": numeric_items[0]["attack_applicability"],
            "samples": len(numeric_items),
            "note": "",
        }
        for metric_name in metric_names:
            values = np.array([float(item[metric_name]) for item in numeric_items], dtype=np.float64)
            row[f"mean_{metric_name}"] = float(np.mean(values))
            row[f"std_{metric_name}"] = float(np.std(values))
        rows.append(row)
    return rows


def _save_outputs(
    details: list[dict[str, object]],
    summary: list[dict[str, object]],
    feature_names: list[str],
) -> None:
    payload = {
        "config": {
            "dataset": "PaySim",
            "feature_names": feature_names,
            "attack": "known-label gradient matching",
            "iterations": ATTACK_ITERATIONS,
            "num_samples": ATTACK_NUM_SAMPLES,
            "warmup_rounds": ATTACK_WARMUP_ROUNDS,
            "optimizer": os.environ.get("ATTACK_OPTIMIZER", "adam"),
            "tabular_psnr_ssim_note": (
                "PSNR and SSIM are computed on deterministic pseudo-images made by "
                "reshaping normalized tabular feature vectors. Feature MSE, cosine, "
                "Pearson, and sign-match are the primary tabular metrics."
            ),
            "secureagg_note": (
                "Secure Aggregation rows marked not_directly_applicable reflect the "
                "true server-side threat model: individual client updates are hidden. "
                "PreAggregationLeakage rows are analysis-only upper bounds."
            ),
        },
        "summary": summary,
        "details": details,
    }
    DETAILS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fieldnames = sorted({key for row in summary for key in row})
    with SUMMARY_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
