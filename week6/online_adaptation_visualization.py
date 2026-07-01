from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# FILE PATHS
# ============================================================

SUMMARY_RESULTS = {
    "Cross-Session": {
        "Baseline": "./week6/results/eegnet/cross_session_baseline/metrics_summary.csv",
        "AdaBN": "./week6/results/eegnet/cross_session_adabn/online_adabn_summary.csv",
        "AdaBN Rolling": "./week6/results/eegnet/cross_session_adabn_buffer/online_adabn_summary.csv",
        "TENT": "./week6/results/eegnet/cross_session_tent/online_tent_summary.csv",
    },
    "Cross-Subject": {
        "Baseline": "./week6/results/eegnet/cross_subject_baseline/metrics_summary.csv",
        "AdaBN": "./week6/results/eegnet/cross_subject_adabn/online_adabn_summary.csv",
        "AdaBN Rolling": "./week6/results/eegnet/cross_subject_adabn_buffer/online_adabn_summary.csv",
        "TENT": "./week6/results/eegnet/cross_subject_tent/online_tent_summary.csv",
    },
}


PER_SUBJECT_RESULTS = {
    "Baseline": "./week6/results/eegnet/cross_session_baseline/metrics_per_seed.csv",
    "AdaBN": "./week6/results/eegnet/cross_session_adabn/online_adabn_metrics_per_seed.csv",
    "AdaBN Rolling": "./week6/results/eegnet/cross_session_adabn_buffer/online_adabn_metrics_per_seed.csv",
    "TENT": "./week6/results/eegnet/cross_session_tent/online_tent_metrics_per_seed.csv",
}

MOMENTUM_RESULTS = {
    "0.01": "./week6/results/eegnet/cross_session_adabn_momentum_001/online_adabn_summary.csv",
    "0.1": "./week6/results/eegnet/cross_session_adabn_momentum_01/online_adabn_summary.csv",
    "0.5": "./week6/results/eegnet/cross_session_adabn_momentum_05/online_adabn_summary.csv",
}

MOMENTUM_PER_SUBJECT_RESULTS = {
    "0.01": "./week6/results/eegnet/cross_session_adabn_momentum_001/online_adabn_metrics_per_seed.csv",
    "0.1": "./week6/results/eegnet/cross_session_adabn_momentum_01/online_adabn_metrics_per_seed.csv",
    "0.5": "./week6/results/eegnet/cross_session_adabn_momentum_05/online_adabn_metrics_per_seed.csv",
}

BUFFER_RESULTS = {
    "4": "./week6/results/eegnet/cross_session_adabn_buffer_4/online_adabn_summary.csv",
    "8": "./week6/results/eegnet/cross_session_adabn_buffer_8/online_adabn_summary.csv",
    "16": "./week6/results/eegnet/cross_session_adabn_buffer_16/online_adabn_summary.csv",
    "32": "./week6/results/eegnet/cross_session_adabn_buffer_32/online_adabn_summary.csv",
}

BUFFER_PER_SUBJECT_RESULTS = {
    "4": "./week6/results/eegnet/cross_session_adabn_buffer_4/online_adabn_metrics_per_seed.csv",
    "8": "./week6/results/eegnet/cross_session_adabn_buffer_8/online_adabn_metrics_per_seed.csv",
    "16": "./week6/results/eegnet/cross_session_adabn_buffer_16/online_adabn_metrics_per_seed.csv",
    "32": "./week6/results/eegnet/cross_session_adabn_buffer_32/online_adabn_metrics_per_seed.csv",
}

OUT_DIR = Path("./week6/results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def load_metric(summary_csv, metric="balanced_acc"):
    df = pd.read_csv(summary_csv, index_col=0)

    mean = df.loc[metric, "mean"]
    std = df.loc[metric, "std"]

    return mean, std


# ============================================================
# FIGURE 1
# MAIN METHOD COMPARISON
# ============================================================

def plot_main_comparison(metric="balanced_acc"):
    protocols = list(SUMMARY_RESULTS.keys())
    methods = list(SUMMARY_RESULTS[protocols[0]].keys())

    x = np.arange(len(protocols))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, method in enumerate(methods):
        means = []
        stds = []

        for protocol in protocols:
            csv_path = SUMMARY_RESULTS[protocol][method]

            mean, std = load_metric(csv_path, metric)

            means.append(mean)
            stds.append(std)

        ax.bar(
            x + i * width - (1.5 * width),
            means,
            width,
            yerr=stds,
            capsize=5,
            label=method,
        )

    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Online Adaptation under EEG Domain Shift")

    ax.set_xticks(x)
    ax.set_xticklabels(protocols)

    ax.set_ylim(0, 1)

    ax.legend()

    plt.tight_layout()

    save_path = OUT_DIR / "main_method_comparison.png"

    plt.savefig(save_path, dpi=300)

    plt.show()

    print(f"[+] Saved: {save_path}")


# ============================================================
# FIGURE 2
# SUBJECT-WISE IMPROVEMENT
# ============================================================

def plot_subject_improvement(metric="balanced_acc"):
    baseline_df = pd.read_csv(PER_SUBJECT_RESULTS["Baseline"])

    methods = [
        "AdaBN",
        "AdaBN Rolling",
        "TENT",
    ]

    subjects = sorted(baseline_df["subject"].unique())

    # Mean baseline per subject
    baseline_subject = (
        baseline_df.groupby("subject")[metric]
        .mean()
    )

    x = np.arange(len(subjects))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, method in enumerate(methods):
        method_df = pd.read_csv(PER_SUBJECT_RESULTS[method])

        method_subject = (
            method_df.groupby("subject")[metric]
            .mean()
        )

        improvement = (
            method_subject[subjects].values
            - baseline_subject[subjects].values
        )

        ax.bar(
            x + i * width - width,
            improvement,
            width,
            label=method,
        )

    ax.axhline(0, linestyle="--")

    ax.set_ylabel("Balanced Accuracy Improvement")
    ax.set_title("Per-Subject Online Adaptation Gain (Cross-Session)")

    ax.set_xticks(x)
    ax.set_xticklabels(subjects)

    ax.legend()

    plt.tight_layout()

    save_path = OUT_DIR / "subject_improvement_cross_session.png"

    plt.savefig(save_path, dpi=300)

    plt.show()

    print(f"[+] Saved: {save_path}")


# ============================================================
# FIGURE 3
# CALIBRATION COMPARISON
# ============================================================

def plot_calibration(metric="ece"):
    protocols = list(SUMMARY_RESULTS.keys())
    methods = list(SUMMARY_RESULTS[protocols[0]].keys())

    x = np.arange(len(protocols))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, method in enumerate(methods):
        means = []
        stds = []

        for protocol in protocols:
            csv_path = SUMMARY_RESULTS[protocol][method]

            mean, std = load_metric(csv_path, metric)

            means.append(mean)
            stds.append(std)

        ax.bar(
            x + i * width - (1.5 * width),
            means,
            width,
            yerr=stds,
            capsize=5,
            label=method,
        )

    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} Comparison")

    ax.set_xticks(x)
    ax.set_xticklabels(protocols)

    ax.legend()

    plt.tight_layout()

    save_path = OUT_DIR / f"{metric}_comparison.png"

    plt.savefig(save_path, dpi=300)

    plt.show()

    print(f"[+] Saved: {save_path}")



def plot_momentum_ablation(metric="balanced_acc"):
    momentums = list(MOMENTUM_RESULTS.keys())

    means = []
    stds = []

    for momentum in momentums:
        csv_path = MOMENTUM_RESULTS[momentum]

        mean, std = load_metric(csv_path, metric)

        means.append(mean)
        stds.append(std)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        momentums,
        means,
        yerr=stds,
        capsize=5,
    )

    ax.set_xlabel("BN Momentum")
    ax.set_ylabel("Balanced Accuracy")

    ax.set_title(
        "AdaBN Momentum Ablation (Cross-Session)"
    )

    ax.set_ylim(0, 1)

    plt.tight_layout()

    save_path = OUT_DIR / "momentum_ablation.png"

    plt.savefig(save_path, dpi=300)

    plt.show()

    print(f"[+] Saved: {save_path}")

def plot_momentum_ablation_per_subject(metric="balanced_acc"):
    subjects = None

    fig, ax = plt.subplots(figsize=(12, 6))

    x = None
    width = 0.25

    momentums = list(MOMENTUM_PER_SUBJECT_RESULTS.keys())

    for i, momentum in enumerate(momentums):
        df = pd.read_csv(
            MOMENTUM_PER_SUBJECT_RESULTS[momentum]
        )

        subject_means = (
            df.groupby("subject")[metric]
            .mean()
            .sort_index()
        )

        if subjects is None:
            subjects = subject_means.index.tolist()
            x = np.arange(len(subjects))

        ax.bar(
            x + i * width - width,
            subject_means.values,
            width,
            label=f"Momentum={momentum}",
        )

    ax.set_xlabel("Subject")
    ax.set_ylabel("Balanced Accuracy")

    ax.set_title(
        "AdaBN Momentum Ablation per Subject (Cross-Session)"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(subjects)

    ax.set_ylim(0, 1)

    ax.legend()

    plt.tight_layout()

    save_path = (
        OUT_DIR / "momentum_ablation_per_subject.png"
    )

    plt.savefig(save_path, dpi=300)

    plt.show()

    print(f"[+] Saved: {save_path}")


def plot_buffer_ablation(metric="balanced_acc"):
    buffer_sizes = list(BUFFER_RESULTS.keys())

    means = []
    stds = []

    for buffer_size in buffer_sizes:
        csv_path = BUFFER_RESULTS[buffer_size]

        mean, std = load_metric(csv_path, metric)

        means.append(mean)
        stds.append(std)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        buffer_sizes,
        means,
        yerr=stds,
        capsize=5,
    )

    ax.set_xlabel("Rolling Buffer Size")
    ax.set_ylabel("Balanced Accuracy")

    ax.set_title(
        "Rolling Buffer Ablation (Cross-Session)"
    )

    ax.set_ylim(0, 1)

    plt.tight_layout()

    save_path = OUT_DIR / "buffer_ablation.png"

    plt.savefig(save_path, dpi=300)

    plt.show()

    print(f"[+] Saved: {save_path}")

def plot_buffer_ablation_per_subject(
    metric="balanced_acc"
):
    subjects = None

    fig, ax = plt.subplots(figsize=(12, 6))

    x = None
    width = 0.2

    buffer_sizes = list(
        BUFFER_PER_SUBJECT_RESULTS.keys()
    )

    for i, buffer_size in enumerate(buffer_sizes):
        df = pd.read_csv(
            BUFFER_PER_SUBJECT_RESULTS[buffer_size]
        )

        subject_means = (
            df.groupby("subject")[metric]
            .mean()
            .sort_index()
        )

        if subjects is None:
            subjects = subject_means.index.tolist()
            x = np.arange(len(subjects))

        ax.bar(
            x + i * width - 1.5 * width,
            subject_means.values,
            width,
            label=f"Buffer={buffer_size}",
        )

    ax.set_xlabel("Subject")
    ax.set_ylabel("Balanced Accuracy")

    ax.set_title(
        "Rolling Buffer Ablation per Subject"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(subjects)

    ax.set_ylim(0, 1)

    ax.legend()

    plt.tight_layout()

    save_path = (
        OUT_DIR / "buffer_ablation_per_subject.png"
    )

    plt.savefig(save_path, dpi=300)

    plt.show()

    print(f"[+] Saved: {save_path}")




# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # plot_main_comparison(metric="balanced_acc")

    # plot_subject_improvement(metric="balanced_acc")

    # plot_calibration(metric="ece")

    # plot_momentum_ablation(metric="balanced_acc")

    # plot_momentum_ablation_per_subject(
    #     metric="balanced_acc"
    # )

    plot_buffer_ablation(metric="balanced_acc")

    plot_buffer_ablation_per_subject(
        metric="balanced_acc"
    )