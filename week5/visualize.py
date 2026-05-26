import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.calibration import CalibrationDisplay

def plot_dashboard(results_dir):
    res_path = Path(results_dir)
    
    # Load evaluation data
    df = pd.read_csv(res_path / "metrics_per_seed.csv")
    with open(res_path / "prob_data.json", "r") as f:
        prob_data = json.load(f)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("BCI Evaluation Dashboard: 4-Pillars", fontsize=16, fontweight="bold")

    # --- 1. Accuracy Pillar: Boxplots ---
    ax = axs[0, 0]
    acc_metrics = df[["accuracy", "macro_f1", "balanced_acc"]].melt(var_name="Metric", value_name="Score")
    sns.boxplot(x="Metric", y="Score", data=acc_metrics, ax=ax, palette="Blues")
    sns.stripplot(x="Metric", y="Score", data=acc_metrics, ax=ax, color="black", alpha=0.6)
    ax.set_title("Accuracy Metrics Across Seeds")
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # --- 2. Reliability Pillar: Reliability Diagram ---
    ax = axs[0, 1]
    # We plot the calibration curve for the first seed as a representative example
    first_seed_key = list(prob_data.keys())[0]
    y_true = np.array(prob_data[first_seed_key]["y_true"])
    y_prob = np.array(prob_data[first_seed_key]["y_prob"])
    
    CalibrationDisplay.from_predictions(y_true, y_prob, n_bins=10, ax=ax, name=f"Model ({first_seed_key})")
    ax.set_title(f"Reliability Diagram (ECE: {df['ece'].mean():.3f})")
    ax.grid(True, linestyle='--', alpha=0.7)

    # --- 3. Efficiency Pillar: Training vs Inference Time ---
    ax = axs[1, 0]
    ax.scatter(df["train_time_sec"], df["inference_time_per_trial_ms"], color="darkorange", s=100, edgecolor="k")
    for i, row in df.iterrows():
        ax.annotate(f"Seed {int(row['seed'])}", (row["train_time_sec"], row["inference_time_per_trial_ms"]), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    ax.set_title("Computational Latency")
    ax.set_xlabel("Total Training Time (s)")
    ax.set_ylabel("Inference Time per Trial (ms)")
    ax.grid(True, linestyle='--', alpha=0.7)

    # --- 4. Accuracy Pillar: Average Confusion Matrix ---
    ax = axs[1, 1]
    avg_cm = np.array([
        [df["cm_00"].mean(), df["cm_01"].mean()],
        [df["cm_10"].mean(), df["cm_11"].mean()]
    ])
    # Normalize
    row_sums = avg_cm.sum(axis=1, keepdims=True)
    avg_cm_norm = avg_cm / row_sums
    
    sns.heatmap(avg_cm_norm, annot=True, fmt=".1%", cmap="Blues", 
                xticklabels=["Left", "Right"], yticklabels=["Left", "Right"], ax=ax)
    ax.set_title("Average Row-Normalized Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    out_img = res_path / "evaluation_dashboard.png"
    plt.savefig(out_img, dpi=200)
    print(f"Dashboard saved to {out_img}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Dashboard from eval.py outputs")
    parser.add_argument("--dir", type=str, default="week5_results", help="Directory containing eval.py outputs")
    args = parser.parse_args()
    
    plot_dashboard(args.dir)