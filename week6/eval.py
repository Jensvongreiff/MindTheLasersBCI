import argparse
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score, 
    confusion_matrix, brier_score_loss
)

from week4.loading import load_bci2a_dataset
from week4.preprocessing import preprocess_bci2a_dataset

from week5.data_augmentation import augment_eeg_training_data

# ==========================================
# 1. THE MODEL DISPATCHER
# ==========================================
def get_model(model_name, mixup=False):
    if model_name.lower() == "lda":
        from pyriemann.estimation import Covariances
        from pyriemann.spatialfilters import CSP
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.pipeline import make_pipeline
        return make_pipeline(Covariances('oas'), CSP(4, log=True), LinearDiscriminantAnalysis())
    elif model_name.lower() == "eegnet":
        from week6.models import EEGNet, PyTorchClassifier
        # --- NEW: Pass mixup flag down to the wrapper ---
        return PyTorchClassifier(EEGNet, epochs=50, batch_size=16, lr=1e-3, mixup=mixup)
    elif model_name.lower() == "fbcnet":
        raise NotImplementedError("FBCNet is not yet implemented.")
    else:
        raise ValueError(f"Model '{model_name}' not recognized.")

# ==========================================
# 2. EVALUATION UTILS
# ==========================================
def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        mask = binids == i
        if np.any(mask):
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            weight = np.sum(mask) / len(y_prob)
            ece += weight * np.abs(bin_acc - bin_conf)
    return ece

# ==========================================
# 3. THE 9-SUBJECT EVALUATION ENGINE
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Universal BCI Evaluator")
    parser.add_argument("--model", type=str, default="eegnet")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--data", type=str, default="./data/bci2a_dataset")
    parser.add_argument("--protocol", type=str, default="baseline", choices=["baseline", "noisy", "cross-session", "cross-subject", "loso"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", type=str, default="./week5/results/undefined_model/undefined_protocol/undefined_run")
    
    # Hidden args for latency calculation logic
    parser.add_argument("--acq_delay", type=float, default=4.0)
    parser.add_argument("--meth_delay", type=float, default=0.0) 
    
    # --- Augmentation Flags ---
    parser.add_argument("--input_augment", action="store_true", help="Apply input-space data augmentation before training")
    parser.add_argument("--mixup_augment", action="store_true", help="Apply Mixup feature-space augmentation during training")
    parser.add_argument("--ICA", action="store_true", help="Apply ICA for data preprocessing")
    parser.add_argument("--classes", type=int, default=4, choices=[2, 4], help="Classify 2 (Left/Right) or 4 (All) motor tasks.")

    parser.add_argument("--save_weights", action="store_true", help="Save trained model weights")
    
    args = parser.parse_args()

    internal_protocol = "cross-subject" if args.protocol.lower() == "loso" else args.protocol.lower()
    seed_pool = [42, 97, 123, 1337, 2024]
    run_seeds = seed_pool[:args.seeds] if args.seeds <= 5 else list(range(args.seeds))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[+] Loading Data from '{args.data}' (ICA={args.ICA}). Please wait...")
    raw_dataset = load_bci2a_dataset(args.data, classes=args.classes)
    data_cache, _ = preprocess_bci2a_dataset(raw_dataset, apply_filter=True, apply_ica=args.ICA)

    subjects = [f"A{i:02d}" for i in range(1, 10)]
    all_metrics = []
    all_probs = {}

    print(f"\nEvaluating Model: {args.model.upper()} | Protocol: {internal_protocol.upper()}")
    
    for seed in run_seeds:
        np.random.seed(seed)
        
        for target_sub in subjects:
            
            # --- Dynamic Data Routing ---
            if internal_protocol == "baseline":
                X_full, y_full = data_cache[target_sub]["T"]["X"], data_cache[target_sub]["T"]["y"]
                split = int(0.8 * len(X_full))
                X_train, y_train = X_full[:split], y_full[:split]
                X_test, y_test = X_full[split:], y_full[split:]
            elif internal_protocol == "noisy":
                X_full, y_full = data_cache[target_sub]["T"]["X"], data_cache[target_sub]["T"]["y"]
                split = int(0.8 * len(X_full))
                X_train, y_train = X_full[:split], y_full[:split]
                noise = np.random.normal(0, 0.5, X_full[split:].shape)
                X_test = X_full[split:] + noise
                y_test = y_full[split:]
            elif internal_protocol == "cross-session":
                X_train, y_train = data_cache[target_sub]["T"]["X"], data_cache[target_sub]["T"]["y"]
                X_test, y_test = data_cache[target_sub]["E"]["X"], data_cache[target_sub]["E"]["y"]
            elif internal_protocol == "cross-subject":
                train_subs = [s for s in subjects if s != target_sub]
                X_train = np.vstack([data_cache[s]["T"]["X"] for s in train_subs])
                y_train = np.concatenate([data_cache[s]["T"]["y"] for s in train_subs])
                X_test, y_test = data_cache[target_sub]["T"]["X"], data_cache[target_sub]["T"]["y"]
            else:
                raise ValueError(f"Unknown Protocol: {internal_protocol}")

            # Dynamic Class Filtering
            if args.classes == 2:
                # Mask out classes 2 (Feet) and 3 (Tongue)
                train_mask = (y_train == 0) | (y_train == 1)
                test_mask = (y_test == 0) | (y_test == 1)
                
                X_train, y_train = X_train[train_mask], y_train[train_mask]
                X_test, y_test = X_test[test_mask], y_test[test_mask]
            
            y_train = y_train - np.min(y_train)
            y_test = y_test - np.min(y_test)

            # --- Input-Space Augmentation Routing ---
            if args.input_augment and not args.weights:
                original_len = len(X_train)
                X_train, y_train = augment_eeg_training_data(
                    X_train, y_train, noise_std=0.05, scale_std=0.1, augment_factor=1, seed=seed
                )
                print(f"    [Augmentation] {target_sub}: Expanded training set from {original_len} to {len(X_train)} trials.")

            # --- Instantiation & Pre-trained Bypass ---
            model = get_model(args.model, mixup=args.mixup_augment)
            t0_train = time.perf_counter()
            
            if args.weights:
                if hasattr(model, "load_weights"):
                    model.load_weights(
                        args.weights,
                        X_test,
                        nb_classes=args.classes
                    )
                else:
                    raise AttributeError(
                        f"Model '{args.model}' wrapper doesn't support weights."
                    )
            else:
                model.fit(X_train, y_train)

                # Save trained weights
                if args.save_weights and hasattr(model, "save_weights"):
                    weights_dir = out_dir / "weights"
                    weights_dir.mkdir(parents=True, exist_ok=True)

                    weight_path = weights_dir / f"{args.model}_{internal_protocol}_{target_sub}_seed_{seed}.pt"

                    model.save_weights(weight_path)

                    print(f"    [+] Saved weights to: {weight_path}")
                
            train_time = time.perf_counter() - t0_train if not args.weights else 0.0
            
            # --- Inference & Prediction ---
            t0_infer = time.perf_counter()
            probs = model.predict_proba(X_test)
            infer_time = time.perf_counter() - t0_infer
            
            y_pred = np.argmax(probs, axis=1)

            # Dynamic Probability Extraction for Calibration
            n_classes_tested = len(np.unique(y_test))
            if n_classes_tested != args.classes:
                print(f"    [Warning] Detected {n_classes_tested} unique classes in test set, but --classes={args.classes}. Verify if this is correct")
            
            if args.classes == 2:
                y_prob = probs[:, 1]
                ece_score = expected_calibration_error(y_test, y_prob)
                brier_score = brier_score_loss(y_test, y_prob)
            else:
                # Multi-class uses Top-1 Calibration
                y_prob = np.max(probs, axis=1)
                correctness_array = (y_pred == y_test).astype(int)
                ece_score = expected_calibration_error(correctness_array, y_prob)
                
                # Manual Brier Score for Multi-class
                y_test_onehot = np.eye(args.classes)[y_test]
                brier_score = np.mean(np.sum((probs - y_test_onehot)**2, axis=1))

            # --- Hardware Metadata Extraction ---
            if hasattr(model, "get_info"):
                info = model.get_info() # type: ignore
                device_str = info.get("device", "CPU")
                params_str = info.get("parameters", "N/A")
                batch_str = info.get("batch_size", "N/A")
            else:
                device_str, params_str, batch_str = "CPU", "N/A", "N/A"

            # --- Strict Latency Math (Acq + Meth + Comp) ---
            comp_delay = infer_time / len(y_test)
            total_delay = args.acq_delay + args.meth_delay + comp_delay

            # --- Metrics Calculation ---
            metrics = {
                "seed": seed,
                "subject": target_sub,
                "model": args.model.upper(),
                "device": device_str,
                
                # --- Track Augmentations in the CSV ---
                "input_augment": args.input_augment,
                "mixup_augment": args.mixup_augment,

                # Device Settings
                "parameters": params_str if params_str != "N/A" else 0,
                "batch_size": batch_str if batch_str != "N/A" else 0,
                
                # Latency
                "train_time_sec": train_time,
                "inference_time_sec": infer_time,
                "delay_acquisition_sec": args.acq_delay,
                "delay_methodological_sec": args.meth_delay,
                "delay_computational_sec": comp_delay,
                "delay_total_sec": total_delay,
                "inference_time_per_trial_ms": comp_delay * 1000,
                
                # Accuracy & Calibration
                "accuracy": accuracy_score(y_test, y_pred),
                "macro_f1": f1_score(y_test, y_pred, average="macro"),
                "balanced_acc": balanced_accuracy_score(y_test, y_pred),
                "brier_score": brier_score,
                "ece": ece_score
            }
            
            cm_labels = list(range(args.classes))
            cm = confusion_matrix(y_test, y_pred, labels=cm_labels)
            
            for i in range(args.classes):
                for j in range(args.classes):
                    metrics[f"cm_{i}{j}"] = int(cm[i, j])

            all_metrics.append(metrics)

            # Save correctness for Multi-Class Top-1 Calibration
            if args.classes == 2:
                all_probs[f"{target_sub}_seed_{seed}"] = {"y_true": y_test.tolist(), "y_prob": y_prob.tolist()}
            else:
                all_probs[f"{target_sub}_seed_{seed}"] = {"y_true": correctness_array.tolist(), "y_prob": y_prob.tolist()}

    # --- Save Raw Metrics ---
    df = pd.DataFrame(all_metrics)
    df.to_csv(out_dir / "metrics_per_seed.csv", index=False)
    with open(out_dir / "prob_data.json", "w") as f:
        json.dump(all_probs, f)
        
    # ==========================================
    # 4. FINAL PIPELINE REPORT (MEAN ± STD)
    # ==========================================
    # Calculate statistics across all subjects and seeds
    num_cols = df.select_dtypes(include=[np.number]).columns
    summary_stats = df[num_cols].agg(['mean', 'std']).T
    summary_stats['formatted'] = summary_stats.apply(lambda row: f"{row['mean']:.4f} ± {row['std']:.4f}", axis=1)
    summary_stats.to_csv(out_dir / "metrics_summary.csv")

    dev = df['device'].iloc[0]
    param = df['parameters'].iloc[0]
    batch = df['batch_size'].iloc[0]
    
    print("\n" + "="*50)
    print(" "*10 + "FINAL PIPELINE REPORT")
    print("="*50)
    print(f"Device: {dev} | Model: {args.model.upper()} | Parameters: {param} | Batch: {batch}")
    print(f"Protocol: {internal_protocol.upper()} | Seeds: {args.seeds} | Total Runs: {len(df)}")
    print("-"*50)
    print("[1] Accuracy Pillar (Mean ± Std)")
    print(f"    Accuracy:      {summary_stats.loc['accuracy', 'formatted']}")
    print(f"    Macro-F1:      {summary_stats.loc['macro_f1', 'formatted']}")
    print(f"    Balanced Acc:  {summary_stats.loc['balanced_acc', 'formatted']}")
    print("-"*50)
    print("[2] Reliability Pillar")
    print(f"    ECE:           {summary_stats.loc['ece', 'formatted']}")
    print(f"    Brier Score:   {summary_stats.loc['brier_score', 'formatted']}")
    print("-"*50)
    print("[3] Efficiency / Latency Pillar")
    print(f"    Train Time:    {summary_stats.loc['train_time_sec', 'formatted']} sec")
    print(f"    Total Delay:   {summary_stats.loc['delay_total_sec', 'formatted']} sec")
    print(f"    ├── Acq Delay:   {args.acq_delay:.3f} s")
    print(f"    ├── Meth Delay:  {args.meth_delay:.3f} s")
    print(f"    └── Comp Delay:  {summary_stats.loc['delay_computational_sec', 'formatted']} s")
    print("="*50)
    print(f"[+] Output saved to: {out_dir}/")

if __name__ == "__main__":
    main()