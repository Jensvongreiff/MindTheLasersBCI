import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from sklearn.calibration import calibration_curve

def plot_accuracy_pillar(df, model_name, protocol_name, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f"Accuracy Pillar | Model: {model_name} | Protocol: {protocol_name.title()}", fontsize=14, fontweight="bold")
    
    acc_metrics = df[["accuracy", "macro_f1", "balanced_acc"]].melt(var_name="Metric", value_name="Score")
    sns.boxplot(x="Metric", y="Score", data=acc_metrics, ax=ax, palette="Blues")
    sns.stripplot(x="Metric", y="Score", data=acc_metrics, ax=ax, color="black", alpha=0.4, jitter=True)
    
    ax.set_title("Variance Across 9 Subjects & Seeds", fontsize=12)
    ax.set_ylim(0.3, 1.0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_efficiency_pillar(df, model_name, protocol_name, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f"Efficiency Pillar | Model: {model_name} | Protocol: {protocol_name.title()}", fontsize=14, fontweight="bold")
    
    ax.scatter(df["train_time_sec"], df["inference_time_per_trial_ms"], color="darkorange", s=60, edgecolor="k", alpha=0.7)
    ax.set_title("Computational Latency", fontsize=12)
    ax.set_xlabel("Total Training Time (s)")
    ax.set_ylabel("Inference Time per Trial (ms)")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_confusion_matrix(df, model_name, protocol_name, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f"Bias & Errors | Model: {model_name} | Protocol: {protocol_name.title()}", fontsize=14, fontweight="bold")
    
    avg_cm = np.array([
        [df["cm_00"].mean(), df["cm_01"].mean()],
        [df["cm_10"].mean(), df["cm_11"].mean()]
    ])
    row_sums = avg_cm.sum(axis=1, keepdims=True)
    avg_cm_norm = avg_cm / row_sums
    
    sns.heatmap(avg_cm_norm, annot=True, fmt=".1%", cmap="Blues", 
                xticklabels=["Left", "Right"], yticklabels=["Left", "Right"], ax=ax)
                
    ax.set_title("Average Row-Normalized Confusion Matrix (9 Subjects)", fontsize=12)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_reliability_pillar(prob_data, ece_score, model_name, protocol_name, out_path):
    # Grab the first seed of A01 as a representative example
    rep_key = list(prob_data.keys())[0]
    y_true = np.array(prob_data[rep_key]["y_true"])
    y_prob = np.array(prob_data[rep_key]["y_prob"])
    
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle(f"Reliability Pillar | Model: {model_name} | Protocol: {protocol_name.title()}", fontsize=14, fontweight="bold")
    
    # Split layout: 3 parts curve, 1 part histogram
    gs = GridSpec(4, 1, figure=fig)
    ax_curve = fig.add_subplot(gs[:3, 0])
    ax_hist = fig.add_subplot(gs[3, 0], sharex=ax_curve)
    
    # The Calibration Curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    ax_curve.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    ax_curve.plot(prob_pred, prob_true, marker="s", color="tab:blue", linewidth=2, label="Right Hand Confidence")
    
    ax_curve.set_ylabel("Fraction of Positives (Real Accuracy)")
    ax_curve.set_title(f"Empirical Calibration Curve (Mean ECE across all subs: {ece_score:.3f})", fontsize=12)
    ax_curve.legend(loc="lower right")
    ax_curve.grid(True, linestyle="--", alpha=0.7)
    
    # The Density Histogram
    ax_hist.hist(y_prob, range=(0, 1), bins=10, histtype="bar", color="tab:blue", alpha=0.7, edgecolor="black")
    ax_hist.set_xlabel("Mean Predicted Probability (Confidence)")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(axis='y', linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_master_degradation(metrics_tracker, model_name, out_path):
    protocols = list(metrics_tracker.keys())
    means = [metrics_tracker[p]["Mean"] for p in protocols]
    worsts = [metrics_tracker[p]["Worst"] for p in protocols]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f"System Robustness | Average vs Worst-Subject Accuracy", fontsize=16, fontweight="bold")
    
    x = np.arange(len(protocols))
    width = 0.35
    
    ax.bar(x - width/2, means, width, label='Average Subject', color="tab:blue", edgecolor="black")
    ax.bar(x + width/2, worsts, width, label='Worst Subject', color="tab:red", edgecolor="black")
    
    ax.set_ylabel('Accuracy')
    ax.set_title(f"Model: {model_name} (Evaluated across 9 BCI Subjects)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(protocols)
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="upper right")
    ax.grid(axis='y', linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"\n[+] Master degradation summary saved to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="week5/results")
    parser.add_argument("--model", type=str, default="EEGNet")
    parser.add_argument("--protocols", type=str, default="all")
    parser.add_argument("--run", type=str, default="latest") # NEW: Strict run locking
    args = parser.parse_args()

    root_dir = Path(args.dir)
    if args.protocols.lower() == "all":
        target_protocols = ["baseline", "noisy", "cross-session", "cross-subject"]
    else:
        target_protocols = [p.strip() for p in args.protocols.split(",")]

    # --- 1. STRICT RUN CONSISTENCY CHECK ---
    target_run_name = args.run
    if target_run_name.lower() == "latest":
        all_valid_runs = []
        for p in target_protocols:
            p_dir = root_dir / p
            if p_dir.exists():
                all_valid_runs.extend([d for d in p_dir.iterdir() if d.is_dir() and (d / "metrics_per_seed.csv").exists()])
        
        if not all_valid_runs:
            print("[-] Error: No valid run folders found anywhere.")
            return
        
        # Find the absolute newest folder and lock in its name globally
        newest_run_dir = max(all_valid_runs, key=lambda d: d.stat().st_mtime)
        target_run_name = newest_run_dir.name
        print(f"[*] Auto-detected latest run: '{target_run_name}'. Enforcing across all protocols to prevent mixing!")

    metrics_tracker = {}

    # --- 2. GENERATE DASHBOARDS ---
    for protocol in target_protocols:
        res_path = root_dir / protocol / target_run_name
        
        # If the locked run name is missing for a protocol, it safely skips it instead of mixing older runs
        if not res_path.exists() or not (res_path / "metrics_per_seed.csv").exists():
            print(f"[-] Warning: Run '{target_run_name}' is missing for {protocol.upper()}. Skipping...")
            continue
            
        print(f"[*] Extracting Dashboards for: {protocol.upper()}...")
        df = pd.read_csv(res_path / "metrics_per_seed.csv")
        with open(res_path / "prob_data.json", "r") as f:
            prob_data = json.load(f)

        plot_accuracy_pillar(df, args.model, protocol, res_path / "accuracy_pillar.png")
        plot_efficiency_pillar(df, args.model, protocol, res_path / "efficiency_pillar.png")
        plot_confusion_matrix(df, args.model, protocol, res_path / "accuracy_confusion_matrix.png")
        
        mean_ece = df['ece'].mean()
        plot_reliability_pillar(prob_data, mean_ece, args.model, protocol, res_path / "reliability_pillar.png")
        
        # Track for Master Degradation Slide
        subject_means = df.groupby('subject')['accuracy'].mean()
        metrics_tracker[protocol.title()] = {
            "Mean": subject_means.mean(),
            "Worst": subject_means.min()
        }
        
    if len(metrics_tracker) > 1 and (args.protocols.lower() == "all" or len(target_protocols) > 1):
        # Dynamically append the run name to the master summary output path
        plot_master_degradation(metrics_tracker, args.model, root_dir / f"master_degradation_summary_{target_run_name}.png")

if __name__ == "__main__":
    main()