"""
mind_the_lasers/src/tests/evaluate_pipelines.py

Synchronous K-Fold Cross-Validation Evaluation Harness.
This script tests both BCI baselines against an offline .xdf dataset and 
generates comprehensive metrics reports using DecoderEvaluator.
"""

import os
import time
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from mind_the_lasers.src.pipeline.signal import EEGDataLoaderOffline, EEGWindow
from mind_the_lasers.src.pipeline.pipeline_config import build_pipeline
from mind_the_lasers.src.pipeline.smoothing import SmoothingController
from mind_the_lasers.src.pipeline.decoder_metrics import DecoderEvaluator

def main():
    # ==========================================
    # 1. Configuration
    # ==========================================
    dataset_path = r"C:\Users\marti\Documents\Programmieren\RCI\4Semester\BCI\practical-ss26-team4\data\sub-P999\our_structure\sub-P666_ses-S002_task-arrow_run-001_eeg.xdf" 
    
    fs = 250
    window_samples = int(fs * 1.0)  # 1-second window
    stride_samples = int(fs * 0.1)  # 100ms stride (matching live LSL behavior)
    n_splits = 5                    # 5-Fold Cross Validation
    baselines = ["csp-lda", "eegnet"]
    
    # Matches the default mapping in EEGDataLoaderOffline
    label_map = {0: "left", 1: "right", 2: "rest"}

    # ==========================================
    # 2. Data Loading & Preparation
    # ==========================================
    print(f"Loading dataset: {dataset_path}")
    data_loader = EEGDataLoaderOffline(data_path=dataset_path)
    
    # The loader naturally splits 50/50. We recombine it to perform custom K-Fold splitting.
    X_train, X_test, y_train, y_test = data_loader.load_data()
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

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            print(f"\n--- Fold {fold}/{n_splits} ---")
            
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]

            # Initialize isolated components
            pipeline = build_pipeline(baseline, window_samples)
            smoothing = SmoothingController(window_size=5, confidence_threshold=0.60)
            evaluator = DecoderEvaluator()

            # Calibrate the pipeline (This will temporarily overwrite the weights files)
            pipeline.calibrate(X_train_fold, y_train_fold, sampling_rate=fs)

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

            # Export the metrics for this specific fold
            report_name = f"{baseline}_fold_{fold}"
            evaluator.generate_report(report_name)

if __name__ == "__main__":
    main()