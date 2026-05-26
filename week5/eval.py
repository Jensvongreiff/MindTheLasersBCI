import argparse
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score, 
    confusion_matrix, brier_score_loss
)

# Import Week 4 tools
from week4.feature_extraction import load_features
from week4.classification import make_classifier

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Computes the Expected Calibration Error (ECE) for binary classification."""
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

def evaluate_single_seed(X_train, y_train, X_test, y_test, model_type, seed, acq_delay, meth_delay):
    """Trains and evaluates a model for a single random seed."""
    np.random.seed(seed)
    metrics = {"seed": seed}
    
    # ---------------------------------------------------------
    # 1. Model Setup & Device Logging (Generalization Pillar)
    # ---------------------------------------------------------
    if model_type == "mdm" or model_type == "lda":
        model = make_classifier() # From week 4
        metrics["device"] = "CPU"
        metrics["parameters"] = "N/A (sklearn)"
        metrics["batch_size"] = "Full Batch"
    else:
        # Placeholder for PyTorch models (EEGNet, FBCNet)
        metrics["device"] = "GPU (cuda) / CPU fallback"
        metrics["parameters"] = 1500 # Example placeholder
        metrics["batch_size"] = 64
        raise NotImplementedError("Deep Learning models not yet linked.")

    # ---------------------------------------------------------
    # 2. Efficiency Pillar (Latency & ITR)
    # ---------------------------------------------------------
    t0_train = time.perf_counter()
    model.fit(X_train, y_train)
    metrics["train_time_sec"] = time.perf_counter() - t0_train
    
    t0_infer = time.perf_counter()
    
    if hasattr(model, "predict_proba"):
        y_prob_full = model.predict_proba(X_test)
        y_prob = y_prob_full[:, 1] # Probability of class 1 (Right Hand)
        y_pred = np.argmax(y_prob_full, axis=1)
    else:
        y_pred = model.predict(X_test)
        y_prob = y_pred.astype(float) 

    metrics["inference_time_sec"] = time.perf_counter() - t0_infer
    
    # --- The Complete Latency Report ---
    metrics["delay_acquisition_sec"] = acq_delay
    metrics["delay_methodological_sec"] = meth_delay
    metrics["delay_computational_sec"] = metrics["inference_time_sec"] / len(y_test)
    metrics["delay_total_sec"] = (
        metrics["delay_acquisition_sec"] + 
        metrics["delay_methodological_sec"] + 
        metrics["delay_computational_sec"]
    )
    metrics["inference_time_per_trial_ms"] = metrics["delay_computational_sec"] * 1000

    # ---------------------------------------------------------
    # 3. Accuracy Pillar
    # ---------------------------------------------------------
    metrics["accuracy"] = accuracy_score(y_test, y_pred)
    metrics["macro_f1"] = f1_score(y_test, y_pred, average="macro")
    metrics["balanced_acc"] = balanced_accuracy_score(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    metrics["cm_00"], metrics["cm_01"] = cm[0, 0], cm[0, 1]
    metrics["cm_10"], metrics["cm_11"] = cm[1, 0], cm[1, 1]

    # ---------------------------------------------------------
    # 4. Reliability Pillar
    # ---------------------------------------------------------
    metrics["brier_score"] = brier_score_loss(y_test, y_prob)
    metrics["ece"] = expected_calibration_error(y_test, y_prob)
    
    prob_data = {"y_true": y_test.tolist(), "y_prob": y_prob.tolist()}

    return metrics, prob_data

def main():
    parser = argparse.ArgumentParser(description="BCI 4-Pillar Evaluation Framework")
    parser.add_argument("--model", type=str, default="lda", help="Model type: lda, mdm, eegnet")
    parser.add_argument("--weights", type=str, default=None, help="Path to pre-trained weights")
    parser.add_argument("--data", type=str, default="features/all_features.pkl", help="Path to feature dictionary")
    parser.add_argument("--protocol", type=str, default="cross-session", choices=["within-session", "cross-session"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 97, 123], help="List of random seeds")
    
    # Changed default output directory to be inside week5
    parser.add_argument("--out", type=str, default="week5/results", help="Output directory")
    
    # New Latency Arguments
    parser.add_argument("--acq_delay", type=float, default=4.0, 
                        help="Acquisition delay in seconds (e.g., standard BCI2a epoch is 4.0s)")
    parser.add_argument("--meth_delay", type=float, default=0.0, 
                        help="Methodological delay in seconds (e.g., IIR filter group delay or spatial filtering overhead)")
    
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.data}...")
    features = load_features(args.data)
    
    # For baseline demonstration, evaluating Subject A01 on CSP
    subject = "A01"
    method = "csp"
    
    if args.protocol == "cross-session":
        X_train, y_train = features[subject][method]["T"]["X"], features[subject][method]["T"]["y"]
        X_test, y_test = features[subject][method]["E"]["X"], features[subject][method]["E"]["y"]
    else:
        X_train, y_train = features[subject][method]["T"]["X"], features[subject][method]["T"]["y"]
        X_test, y_test = X_train, y_train 

    all_metrics = []
    all_probs = {}

    print(f"\nEvaluating {args.model.upper()} across {len(args.seeds)} seeds...")
    for seed in args.seeds:
        print(f"  Running seed {seed}...")
        # Passed the new delay arguments into the evaluator
        metrics, prob_data = evaluate_single_seed(
            X_train, y_train, X_test, y_test, 
            args.model, seed, args.acq_delay, args.meth_delay
        )
        all_metrics.append(metrics)
        all_probs[f"seed_{seed}"] = prob_data

    # Aggregate and Compute Mean/Std
    df = pd.DataFrame(all_metrics)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop("seed")
    summary_df = df[numeric_cols].agg(['mean', 'std']).T
    summary_df["formatted"] = summary_df.apply(lambda row: f"{row['mean']:.4f} ± {row['std']:.4f}", axis=1)

    print("\n--- Final 4-Pillar Summary ---")
    print(summary_df["formatted"])

    # Save to disk
    df.to_csv(out_dir / "metrics_per_seed.csv", index=False)
    summary_df.to_csv(out_dir / "metrics_summary.csv")
    with open(out_dir / "prob_data.json", "w") as f:
        json.dump(all_probs, f)
        
    print(f"\nOutputs saved to {out_dir}/")

if __name__ == "__main__":
    main()