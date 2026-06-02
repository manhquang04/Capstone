"""Run the complete preliminary FL-DNA research pipeline with realtime logs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "logs"
FRAUD_SUMMARY_PATH = PROJECT_ROOT / "results" / "fraud" / "comparison_summary.json"
PRIVACY_SUMMARY_PATH = PROJECT_ROOT / "results" / "mnist" / "privacy_summary.json"
SEPARATOR = "=" * 60
STEPS = (
    ("Fraud FL Baseline", "experiments/run_fraud_fl_baseline.py"),
    ("Fraud FL + DNA Encoder", "experiments/run_fraud_fl_dna.py"),
    ("Fraud FL + Differential Privacy", "experiments/run_fraud_fl_dp.py"),
    ("Compare Fraud Results", "experiments/compare_fraud_results.py"),
    ("Train MNIST Model", "experiments/train_mnist_model.py"),
    ("Extract Raw MNIST Gradient", "attacks/gradient_extraction.py"),
    ("Encode and Decode DNA Gradient", "attacks/dna_gradient.py"),
    ("Create DP Gradient", "attacks/dp_gradient.py"),
    ("Run Gradient Inversion Attack", "attacks/gradient_inversion.py"),
    ("Evaluate Reconstruction", "attacks/evaluate_reconstruction.py"),
    ("Compare Privacy Results", "experiments/compare_privacy_results.py"),
)


class TeeLogger:
    """Write pipeline messages to terminal and the current run log."""

    def __init__(self, log_path: Path) -> None:
        self.log_file = log_path.open("w", encoding="utf-8")

    def write(self, message: str = "") -> None:
        print(message, flush=True)
        self.log_file.write(message + "\n")
        self.log_file.flush()

    def write_stream_line(self, line: str) -> None:
        print(line, end="", flush=True)
        self.log_file.write(line)
        self.log_file.flush()

    def close(self) -> None:
        self.log_file.close()


def run_step(
    step_number: int,
    name: str,
    script: str,
    environment: dict[str, str],
    logger: TeeLogger,
) -> bool:
    """Run one script and stream combined stdout/stderr in realtime."""
    logger.write(SEPARATOR)
    logger.write(f"STEP {step_number}/{len(STEPS)}: {name}")
    logger.write(f"Script: {script}")
    logger.write(SEPARATOR)
    started = perf_counter()
    process = subprocess.Popen(
        [sys.executable, "-u", script],
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        assert process.stdout is not None
        for line in process.stdout:
            logger.write_stream_line(line)
        return_code = process.wait()
    except KeyboardInterrupt:
        logger.write("\nINTERRUPTED")
        logger.write(f"Script: {script}")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise

    runtime = perf_counter() - started
    if return_code != 0:
        logger.write("FAILED")
        logger.write(f"Script: {script}")
        logger.write(f"Return code: {return_code}")
        logger.write(f"Runtime: {runtime:.2f} seconds")
        return False

    logger.write("COMPLETED")
    logger.write(f"Runtime: {runtime:.2f} seconds")
    logger.write()
    return True


def metric_text(value) -> str:
    return "N/A" if value is None else f"{value:.6f}"


def read_json(path: Path, logger: TeeLogger):
    if not path.is_file():
        logger.write(f"Summary file not found: {path.relative_to(PROJECT_ROOT)}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def print_final_summary(total_runtime: float, log_path: Path, logger: TeeLogger) -> None:
    """Print saved fraud and privacy summaries after all steps complete."""
    logger.write(SEPARATOR)
    logger.write("FINAL SUMMARY")
    logger.write(SEPARATOR)
    logger.write()
    logger.write("Fraud Results")
    fraud_summary = read_json(FRAUD_SUMMARY_PATH, logger)
    if fraud_summary:
        logger.write(
            f"{'Method':<12} {'F1':>10} {'AUC':>10} {'Accuracy':>10} "
            f"{'Precision':>10} {'Recall':>10}"
        )
        final_metrics = fraud_summary.get("final_metrics", {})
        for method in ("Baseline", "DNA", "DP"):
            metrics = final_metrics.get(method)
            if not metrics:
                logger.write(f"{method:<12} Missing metrics")
                continue
            f1 = metrics.get("f1_score", metrics.get("f1"))
            auc = metrics.get("auc_roc", metrics.get("auc"))
            logger.write(
                f"{method:<12} {metric_text(f1):>10} {metric_text(auc):>10} "
                f"{metric_text(metrics.get('accuracy')):>10} "
                f"{metric_text(metrics.get('precision')):>10} "
                f"{metric_text(metrics.get('recall')):>10}"
            )

    logger.write()
    logger.write("Privacy Results")
    privacy_summary = read_json(PRIVACY_SUMMARY_PATH, logger)
    if privacy_summary:
        logger.write(f"{'Method':<8} {'PSNR':>12} {'SSIM':>12} {'MSE':>12}")
        metrics_by_method = privacy_summary.get("metrics", {})
        for method in ("RAW", "DNA", "DP"):
            metrics = metrics_by_method.get(method)
            if not metrics:
                logger.write(f"{method:<8} Missing metrics")
                continue
            logger.write(
                f"{method:<8} {metric_text(metrics.get('psnr')):>12} "
                f"{metric_text(metrics.get('ssim')):>12} "
                f"{metric_text(metrics.get('mse')):>12}"
            )

    logger.write()
    logger.write(f"Total Runtime: {total_runtime:.2f} seconds")
    logger.write(f"Log file: {log_path.relative_to(PROJECT_ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use three fraud FL rounds and 500 inversion iterations.",
    )
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"run_{timestamp}.log"
    logger = TeeLogger(log_path)
    pipeline_started = perf_counter()

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    if args.quick:
        environment["QUICK"] = "1"
    else:
        environment.pop("QUICK", None)

    try:
        logger.write(f"FL-DNA preliminary pipeline mode: {'QUICK' if args.quick else 'FULL'}")
        logger.write(f"Log file: {log_path.relative_to(PROJECT_ROOT)}")
        logger.write()
        for step_number, (name, script) in enumerate(STEPS, start=1):
            if not run_step(step_number, name, script, environment, logger):
                logger.write("Pipeline stopped after failed step.")
                return 1

        print_final_summary(perf_counter() - pipeline_started, log_path, logger)
        return 0
    except KeyboardInterrupt:
        logger.write("Pipeline interrupted by user.")
        logger.write(f"Log file: {log_path.relative_to(PROJECT_ROOT)}")
        return 130
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
