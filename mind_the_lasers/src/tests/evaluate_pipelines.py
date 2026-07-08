"""
mind_the_lasers/src/tests/evaluate_pipelines.py

Synchronous K-Fold Cross-Validation Evaluation Harness.
This script tests both BCI baselines against an offline .xdf dataset and 
generates comprehensive metrics reports using DecoderEvaluator.
"""

import os
import re
import json
import time
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from mind_the_lasers.src.pipeline.signal import EEGDataLoaderOffline, EEGWindow
from mind_the_lasers.src.pipeline.pipeline_config import build_pipeline
from mind_the_lasers.src.pipeline.smoothing import SmoothingController
from mind_the_lasers.src.pipeline.decoder_metrics import DecoderEvaluator

def get_next_run_number(baseline: str, output_dir: str = "mind_the_lasers/reports") -> str:
    """Scans the directory and returns the next auto-incremented filename."""
    os.makedirs(output_dir, exist_ok=True)
    pattern = re.compile(rf"^decoder_summary_{baseline}_(\d{{4}})\.json$")
    highest_num = 0
    
    for filename in os.listdir(output_dir):
        match = pattern.match(filename)
        if match:
            highest_num = max(highest_num, int(match.group(1)))
            
    return f"{highest_num + 1:04d}"

def aggregate_reports(reports: list[dict]) -> dict:
    """Averages rates/scores and sums counts across K-Folds."""
    k = len(reports)
    summary = {}

    # Mean Aggregations
    summary["overall_accuracy"] = sum(r["overall_accuracy"] for r in reports) / k
    summary["balanced_accuracy"] = sum(r["balanced_accuracy"] for r in reports) / k
    summary["macro_f1_score"] = sum(r["macro_f1_score"] for r in reports) / k
    summary["rejection_rate"] = sum(r["rejection_rate"] for r in reports) / k
    
    summary["false_activations_during_rest"] = {
        "count": sum(r["false_activations_during_rest"]["count"] for r in reports),
        "rate": sum(r["false_activations_during_rest"]["rate"] for r in reports) / k
    }

    # Sum Aggregations (Matrices and Counts)
    cm_sum = np.zeros((3, 3))
    for r in reports:
        cm_sum += np.array(r["confusion_matrix"])
    summary["confusion_matrix"] = cm_sum.tolist()

    summary["predictions_per_class"] = {
        cls: sum(r["predictions_per_class"].get(cls, 0) for r in reports)
        for cls in ["left", "rest", "right"]
    }

    # Dictionary Mean Aggregations
    for metric in ["class_wise_accuracy", "precision_per_class", "recall_per_class", "f1_score_per_class"]:
        summary[metric] = {
            cls: sum(r[metric].get(cls, 0) for r in reports) / k
            for cls in ["left", "rest", "right"]
        }

    # Confidence and Latency distributions
    if all("confidence_distribution" in r and r["confidence_distribution"] for r in reports):
        summary["confidence_distribution"] = {
            stat: sum(r["confidence_distribution"].get(stat, 0) for r in reports) / k
            for stat in ["mean", "median", "std", "min", "max"]
        }
        
    if all("decision_latency" in r and r["decision_latency"] for r in reports):
        summary["decision_latency"] = {
            stat: sum(r["decision_latency"].get(stat, 0) for r in reports) / k
            for stat in ["mean_ms", "95th_percentile_ms"]
        }

    # MTL Suitability Score (0-100)
    macro_f1 = summary["macro_f1_score"]
    false_act_rate = summary["false_activations_during_rest"]["rate"]
    min_f1 = min(summary["f1_score_per_class"].values())
    rej_rate = summary["rejection_rate"]

    score = (macro_f1 * 40) + ((1.0 - false_act_rate) * 30) + (min_f1 * 15) + ((1.0 - rej_rate) * 15)
    summary["mtl_suitability_score"] = score

    return summary

def main():
    # ==========================================
    # 1. Configuration
    # ==========================================
    dataset_path = r"D:/Programming/BCI_Practical/practical-ss26-team4/data/sub-P999/sub-P999_ses-S009_task-Default_run-001_eeg.xdf" 
    
    window_length = 1  # seconds
    stride = 0.1 # milliseconds

    fs = 250
    window_samples = int(fs * window_length)  # window
    stride_samples = int(fs * stride)  # stride (matching live LSL behavior)
    n_splits = 5                    # 5-Fold Cross Validation
    # baselines = ["csp-lda", "eegnet"]
    baselines = ["zp-csp-lda"]

    # Matches the default mapping in EEGDataLoaderOffline
    label_map = {0: "left", 1: "right", 2: "rest"}

    # ==========================================
    # 2. Data Loading & Preparation
    # ==========================================
    print(f"Loading dataset: {dataset_path}")
    data_loader = EEGDataLoaderOffline(data_path=dataset_path, window_length=window_length, stride=stride)
    
    # The loader naturally splits 50/50. We recombine it to perform custom K-Fold splitting.
    X_train, X_test, y_train, y_test = data_loader.load_data(test_size=0.2)
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    print(f"Total epochs available: {X.shape[0]}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # ==========================================
    # 3. Evaluation Loop
    # ==========================================
    for baseline in baselines:
        print(f"\n{'='*40}")
        print(f"Evaluating Baseline: {baseline.upper()}")
        print(f"{'='*40}")
        fold_reports = []

        run_suffix = get_next_run_number(baseline)
        out_filepath = os.path.join("mind_the_lasers/reports", f"decoder_summary_{baseline}_{run_suffix}.json")

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            print(f"\n--- Fold {fold}/{n_splits} ---")

            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]

            # Initialize isolated components
            pipeline = build_pipeline(baseline, window_samples, suffix="temp")
            smoothing = SmoothingController(window_size=5, confidence_threshold=0.50)
            evaluator = DecoderEvaluator()

            pipeline.data_loader = data_loader

            pipeline.calibrate(
                X_train,
                y_train,
            )
            # Simulate the continuous sliding window over the test epochs
            for trial_idx, (trial_data, label) in enumerate(zip(X_test_fold, y_test_fold)):
                ground_truth_str = label_map[label]
                n_samples = trial_data.shape[1]
                
                # Slide window across the 3-second epoch (mimicking LSL Streamer)
                for i in range(0, n_samples - window_samples + 1, stride_samples):
                    window_data = trial_data[:, i:i+window_samples]

                    window = EEGWindow(
                        data=window_data, 
                        sampling_rate=fs, 
                        timestamp=time.time(),
                        ground_truth=ground_truth_str
                    )

                    # 1. Inference
                    t_start = time.perf_counter()
                    probs = pipeline.process_window(window)
                    latency = time.perf_counter() - t_start

                    # 2. Smoothing
                    pred_label, conf, rejected = smoothing.process(probs)

                    # 3. Metric Logging
                    evaluator.log_step(
                        truth=ground_truth_str,
                        pred=pred_label,
                        confidence=conf,
                        rejected=rejected,
                        latency=latency
                    )

            # Generate report dict without saving to disk
            fold_reports.append(evaluator.generate_report(f"{baseline}_temp", save=False))

        # Aggregate and export
        summary_data = aggregate_reports(fold_reports)
        with open(out_filepath, "w") as f:
            json.dump(summary_data, f, indent=4)
            
        print(f"[{baseline.upper()}] Final MTL Score: {summary_data['mtl_suitability_score']:.2f}/100")
        print(f"Saved Metrics to: {out_filepath}")

        # Train final production model on 100% of data and save weights with the matched suffix
        print(f"Training final {baseline.upper()} model on full dataset...")
        final_pipeline = build_pipeline(baseline, window_samples, suffix=run_suffix)
        final_pipeline.data_loader = data_loader
        final_pipeline.calibrate(X, y)
        print(f"Saved Final Weights with suffix: _{run_suffix}")

if __name__ == "__main__":
    main()