import argparse
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # Suppress TF warnings if still installed

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score, 
    confusion_matrix, brier_score_loss
)

# Import loading/preprocessing and our new PyTorch model
from week4.loading import load_bci2a_dataset
from week4.preprocessing import preprocess_bci2a_dataset
from week5.models import EEGNet

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

def evaluate_single_seed(X_train, y_train, X_test, y_test, subject_id, seed, acq_delay, meth_delay):
    """Trains and evaluates a PyTorch model for a single random seed."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    metrics = {
        "seed": seed,
        "subject": subject_id,
        "device": str(device).upper()
    }

    # Reshape for PyTorch: (Batch, Channels, Height, Width) -> (Batch, 1, Channels, Timepoints)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
    
    # Initialize EEGNet dynamically
    model = EEGNet(nb_classes=2, Chans=X_train.shape[1], Samples=X_train.shape[2]).to(device)
    metrics["parameters"] = sum(p.numel() for p in model.parameters())
    metrics["batch_size"] = 16
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=metrics["batch_size"], shuffle=True)
    
    # ---------------------------------------------------------
    # Training Loop (Efficiency Pillar)
    # ---------------------------------------------------------
    t0_train = time.perf_counter()
    model.train()
    for epoch in range(50):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    metrics["train_time_sec"] = time.perf_counter() - t0_train
    
    # ---------------------------------------------------------
    # Inference & Latency
    # ---------------------------------------------------------
    t0_infer = time.perf_counter()
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        y_prob = probs[:, 1]
        y_pred = np.argmax(probs, axis=1)
    metrics["inference_time_sec"] = time.perf_counter() - t0_infer
    
    # Latency Breakdown
    metrics["delay_acquisition_sec"] = acq_delay
    metrics["delay_methodological_sec"] = meth_delay
    metrics["delay_computational_sec"] = metrics["inference_time_sec"] / len(y_test)
    metrics["delay_total_sec"] = (acq_delay + meth_delay + metrics["delay_computational_sec"])
    metrics["inference_time_per_trial_ms"] = metrics["delay_computational_sec"] * 1000

    # ---------------------------------------------------------
    # Accuracy & Reliability Pillars
    # ---------------------------------------------------------
    metrics["accuracy"] = accuracy_score(y_test, y_pred)
    metrics["macro_f1"] = f1_score(y_test, y_pred, average="macro")
    metrics["balanced_acc"] = balanced_accuracy_score(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    metrics["cm_00"], metrics["cm_01"] = cm[0, 0], cm[0, 1]
    metrics["cm_10"], metrics["cm_11"] = cm[1, 0], cm[1, 1]

    metrics["brier_score"] = brier_score_loss(y_test, y_prob)
    metrics["ece"] = expected_calibration_error(y_test, y_prob)
    
    prob_data = {"y_true": y_test.tolist(), "y_prob": y_prob.tolist()}

    return metrics, prob_data

def main():
    parser = argparse.ArgumentParser(description="BCI 9-Subject Evaluation Framework")
    parser.add_argument("--protocol", type=str, default="cross-session", choices=["baseline", "noisy", "cross-session", "cross-subject"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 97, 123])
    parser.add_argument("--out", type=str, default="week5/results")
    parser.add_argument("--acq_delay", type=float, default=4.0)
    parser.add_argument("--meth_delay", type=float, default=0.0)
    args = parser.parse_args()

    out_dir = Path(args.out) / args.protocol
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[+] Initializing Data Cache (ICA = True). This will take a moment...")
    raw_dataset = load_bci2a_dataset("data/bci2a_dataset")
    data_cache, _ = preprocess_bci2a_dataset(raw_dataset, apply_filter=True, apply_ica=True)
    print("[+] All 9 Subjects Loaded and Cleaned.\n")

    subjects = [f"A{i:02d}" for i in range(1, 10)]
    all_metrics = []
    all_probs = {}

    print(f"Executing Protocol: {args.protocol.upper()}")
    
    # ---------------------------------------------------------
    # The Evaluation Engine (Loops through all 9 Subjects)
    # ---------------------------------------------------------
    for seed in args.seeds:
        print(f"\n--- Random Seed: {seed} ---")
        
        for target_sub in subjects:
            print(f"  Evaluating Test Subject: {target_sub}...")
            
            # --- Dynamic Data Routing ---
            if args.protocol == "baseline":
                X_full, y_full = data_cache[target_sub]["T"]["X"], data_cache[target_sub]["T"]["y"]
                split = int(0.8 * len(X_full))
                X_train, y_train = X_full[:split], y_full[:split]
                X_test, y_test = X_full[split:], y_full[split:]
                
            elif args.protocol == "noisy":
                X_full, y_full = data_cache[target_sub]["T"]["X"], data_cache[target_sub]["T"]["y"]
                split = int(0.8 * len(X_full))
                X_train, y_train = X_full[:split], y_full[:split]
                noise = np.random.normal(0, 0.5, X_full[split:].shape)
                X_test = X_full[split:] + noise
                y_test = y_full[split:]
                
            elif args.protocol == "cross-session":
                X_train, y_train = data_cache[target_sub]["T"]["X"], data_cache[target_sub]["T"]["y"]
                X_test, y_test = data_cache[target_sub]["E"]["X"], data_cache[target_sub]["E"]["y"]
                
            elif args.protocol == "cross-subject":
                # Leave-One-Subject-Out Loop
                train_subs = [s for s in subjects if s != target_sub]
                X_train = np.vstack([data_cache[s]["T"]["X"] for s in train_subs])
                y_train = np.concatenate([data_cache[s]["T"]["y"] for s in train_subs])
                X_test, y_test = data_cache[target_sub]["T"]["X"], data_cache[target_sub]["T"]["y"]

            # Ensure 0-indexing for PyTorch Loss
            y_train = y_train - np.min(y_train)
            y_test = y_test - np.min(y_test)

            metrics, prob_data = evaluate_single_seed(
                X_train, y_train, X_test, y_test, 
                target_sub, seed, args.acq_delay, args.meth_delay
            )
            all_metrics.append(metrics)
            all_probs[f"{target_sub}_seed_{seed}"] = prob_data

    # Save to disk
    df = pd.DataFrame(all_metrics)
    df.to_csv(out_dir / "metrics_per_seed.csv", index=False)
    with open(out_dir / "prob_data.json", "w") as f:
        json.dump(all_probs, f)
        
    print(f"\n[+] Saved {len(df)} evaluation rows to {out_dir}/")

if __name__ == "__main__":
    main()