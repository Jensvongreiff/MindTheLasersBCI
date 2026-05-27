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
    fig.suptitle(f"Accuracy Pillar | Model: {model_name.upper()} | Protocol: {protocol_name.title()}", fontsize=14, fontweight="bold")
    
    acc_metrics = df[["accuracy", "macro_f1", "balanced_acc"]].melt(var_name="Metric", value_name="Score")
    sns.boxplot(x="Metric", y="Score", data=acc_metrics, ax=ax, palette="Blues")
    sns.stripplot(x="Metric", y="Score", data=acc_metrics, ax=ax, color="black", alpha=0.6)
    
    ax.set_title("Metric Variance Across Random Seeds", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_efficiency_pillar(df, model_name, protocol_name, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f"Efficiency Pillar | Model: {model_name.upper()} | Protocol: {protocol_name.title()}", fontsize=14, fontweight="bold")
    
    ax.scatter(df["train_time_sec"], df["inference_time_per_trial_ms"], color="darkorange", s=100, edgecolor="k")
    for i, row in df.iterrows():
        ax.annotate(f"Seed {int(row['seed'])}", (row["train_time_sec"], row["inference_time_per_trial_ms"]), 
                    textcoords="offset points", xytext=(0,10), ha='center')
                    
    ax.set_title("Computational Latency", fontsize=12)
    ax.set_xlabel("Total Training Time (s)")
    ax.set_ylabel("Inference Time per Trial (ms)")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_confusion_matrix(df, model_name, protocol_name, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f"Bias & Errors | Model: {model_name.upper()} | Protocol: {protocol_name.title()}", fontsize=14, fontweight="bold")
    
    avg_cm = np.array([
        [df["cm_00"].mean(), df["cm_01"].mean()],
        [df["cm_10"].mean(), df["cm_11"].mean()]
    ])
    
    # Row-Normalize
    row_sums = avg_cm.sum(axis=1, keepdims=True)
    avg_cm_norm = avg_cm / row_sums
    
    sns.heatmap(avg_cm_norm, annot=True, fmt=".1%", cmap="Blues", 
                xticklabels=["Left", "Right"], yticklabels=["Left", "Right"], ax=ax)
                
    ax.set_title("Average Row-Normalized Confusion Matrix", fontsize=12)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_dual_reliability(prob_data, ece_score, model_name, protocol_name, out_path):
    # Use the first seed as a representative sample for the curve/histogram
    first_seed_key = list(prob_data.keys())[0]
    y_true = np.array(prob_data[first_seed_key]["y_true"])
    y_prob_right = np.array(prob_data[first_seed_key]["y_prob"])
    
    # Mathematically derive Left Hand probabilities
    y_true_left = 1 - y_true
    y_prob_left = 1.0 - y_prob_right
    
    fig = plt.figure(figsize=(9, 8))
    fig.suptitle(f"Reliability Pillar | Model: {model_name.upper()} | Protocol: {protocol_name.title()}", fontsize=14, fontweight="bold")
    
    # Create the split layout: 3 parts curve, 1 part histogram
    gs = GridSpec(4, 1, figure=fig)
    ax_curve = fig.add_subplot(gs[:3, 0])
    ax_hist = fig.add_subplot(gs[3, 0], sharex=ax_curve)
    
    # --- The Calibration Curves ---
    prob_true_r, prob_pred_r = calibration_curve(y_true, y_prob_right, n_bins=10)
    prob_true_l, prob_pred_l = calibration_curve(y_true_left, y_prob_left, n_bins=10)
    
    ax_curve.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    ax_curve.plot(prob_pred_r, prob_true_r, marker="s", label="Right Hand (Class 1)", color="tab:blue")
    ax_curve.plot(prob_pred_l, prob_true_l, marker="^", label="Left Hand (Class 0)", color="tab:orange")
    
    ax_curve.set_ylabel("Fraction of Positives (Real Accuracy)")
    ax_curve.set_title(f"Dual-Hand Calibration Curves (Mean ECE: {ece_score:.3f})", fontsize=12)
    ax_curve.legend(loc="lower right")
    ax_curve.grid(True, linestyle="--", alpha=0.7)
    
    # --- The Probability Histogram ---
    ax_hist.hist(y_prob_right, range=(0, 1), bins=10, histtype="step", lw=2, color="tab:blue", label="Right Hand Guesses")
    ax_hist.hist(y_prob_left, range=(0, 1), bins=10, histtype="step", lw=2, color="tab:orange", label="Left Hand Guesses")
    ax_hist.set_xlabel("Mean Predicted Probability (Confidence)")
    ax_hist.set_ylabel("Count")
    ax_hist.legend(loc="upper center", ncol=2, fontsize=10)
    ax_hist.grid(axis='y', linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_master_degradation(metrics_tracker, model_name, out_path):
    protocols = list(metrics_tracker.keys())
    accs = [metrics_tracker[p]["Acc"] for p in protocols]
    eces = [metrics_tracker[p]["ECE"] for p in protocols]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.suptitle(f"Master Degradation Summary | Model: {model_name.upper()}", fontsize=16, fontweight="bold")
    
    # Axis 1: Accuracy (Blue Line)
    color1 = 'tab:blue'
    ax1.set_xlabel('Evaluation Protocol')
    ax1.set_ylabel('Mean Accuracy', color=color1)
    ax1.plot(protocols, accs, marker='o', color=color1, linewidth=3, markersize=10, label="Accuracy")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0.4, 1.0)
    ax1.grid(True, linestyle="--", alpha=0.7)
    
    # Axis 2: ECE (Red Line)
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Expected Calibration Error (ECE)', color=color2)
    ax2.plot(protocols, eces, marker='X', color=color2, linewidth=3, markersize=10, linestyle="--", label="ECE")
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0.0, max(0.3, max(eces) + 0.05))
    
    fig.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"\n[+] Master degradation summary saved to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate Individual Presentation Slides from eval.py outputs")
    parser.add_argument("--dir", type=str, default="week5/results", help="Root directory containing protocol sub-folders")
    parser.add_argument("--model", type=str, default="EEGNet", help="The model name to display on the dashboards")
    parser.add_argument("--protocols", type=str, default="all", help="Comma separated list of protocols, or 'all'")
    args = parser.parse_args()

    root_dir = Path(args.dir)
    
    if args.protocols.lower() == "all":
        target_protocols = ["baseline", "noisy", "cross-session", "cross-subject"]
    else:
        target_protocols = [p.strip() for p in args.protocols.split(",")]

    metrics_tracker = {}

    for protocol in target_protocols:
        res_path = root_dir / protocol
        
        if not res_path.exists():
            print(f"[-] Warning: Directory {res_path} not found. Skipping...")
            continue
            
        print(f"[*] Processing {protocol.upper()}...")
        
        # Load evaluation data
        df = pd.read_csv(res_path / "metrics_per_seed.csv")
        with open(res_path / "prob_data.json", "r") as f:
            prob_data = json.load(f)

        # 1. Accuracy Pillar
        plot_accuracy_pillar(df, args.model, protocol, res_path / "accuracy_pillar.png")
        
        # 2. Efficiency Pillar
        plot_efficiency_pillar(df, args.model, protocol, res_path / "efficiency_pillar.png")
        
        # 3. Bias & Error Matrix
        plot_confusion_matrix(df, args.model, protocol, res_path / "accuracy_confusion_matrix.png")
        
        # 4. Reliability Pillar (Dual Hand + Hist)
        mean_ece = df['ece'].mean()
        plot_dual_reliability(prob_data, mean_ece, args.model, protocol, res_path / "reliability_pillar.png")
        
        # Track for Master Degradation Plot
        metrics_tracker[protocol.title()] = {
            "Acc": df['accuracy'].mean(),
            "ECE": mean_ece
        }
        
    # Generate Master Degradation Plot if we tracked multiple protocols
    if len(metrics_tracker) > 1 and (args.protocols.lower() == "all" or len(target_protocols) > 1):
        plot_master_degradation(metrics_tracker, args.model, root_dir / "master_degradation_summary.png")

if __name__ == "__main__":
    main()