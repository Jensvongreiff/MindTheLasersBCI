import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def find_latest_log():
    log_dir = (
        Path(__file__).resolve().parents[1]
        / "logs"
        / "training_logs"
    )

    csvs = sorted(
        log_dir.glob("*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not csvs:
        raise FileNotFoundError(
            f"No training logs found in {log_dir}"
        )

    return csvs[0]


def load_data(path):
    return pd.read_csv(path)


def compute_per_class_accuracy(df):
    result = {}

    for cls in ["LEFT", "RIGHT", "REST"]:
        subset = df[df["ground_truth_command"] == cls]

        if len(subset) == 0:
            result[cls] = 0

        else:
            result[cls] = (
                subset["correct"].mean() * 100
            )

    return result


def plot_dashboard(df):

    overall_accuracy = (
        df["correct"].mean() * 100
    )

    per_class = compute_per_class_accuracy(df)

    fig = plt.figure(
        figsize=(14, 8)
    )

    fig.suptitle(
        "Mind the Lasers - Training Summary",
        fontsize=18,
        fontweight="bold",
    )

    # ----------------------------------------------------------
    # Overall accuracy
    # ----------------------------------------------------------

    ax1 = plt.subplot2grid((2, 3), (0, 0))

    ax1.axis("off")

    ax1.text(
        0.5,
        0.65,
        f"{overall_accuracy:.1f}%",
        ha="center",
        va="center",
        fontsize=36,
        fontweight="bold",
    )

    ax1.text(
        0.5,
        0.28,
        "Overall Accuracy",
        ha="center",
        fontsize=16,
    )

    # ----------------------------------------------------------
    # Per-class accuracy
    # ----------------------------------------------------------

    ax2 = plt.subplot2grid((2, 3), (0, 1))

    classes = list(per_class.keys())
    values = list(per_class.values())

    ax2.bar(classes, values)

    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Per-Class Accuracy")

    for i, v in enumerate(values):
        ax2.text(
            i,
            v + 2,
            f"{v:.0f}%",
            ha="center",
        )

    # ----------------------------------------------------------
    # Confusion matrix
    # ----------------------------------------------------------

    ax3 = plt.subplot2grid((2, 3), (0, 2))

    labels = ["LEFT", "RIGHT", "REST"]

    confusion = pd.crosstab(
        df["ground_truth_command"],
        df["predicted_command"],
        dropna=False,
    )

    confusion = confusion.reindex(
        index=labels,
        columns=labels,
        fill_value=0,
    )

    im = ax3.imshow(confusion.values)

    ax3.set_xticks(np.arange(3))
    ax3.set_yticks(np.arange(3))

    ax3.set_xticklabels(labels)
    ax3.set_yticklabels(labels)

    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Ground Truth")

    ax3.set_title("Confusion Matrix")

    for i in range(3):
        for j in range(3):
            ax3.text(
                j,
                i,
                confusion.values[i, j],
                ha="center",
                va="center",
                fontsize=12,
                color="white"
                if confusion.values[i, j]
                > confusion.values.max() / 2
                else "black",
            )

    # ----------------------------------------------------------
    # Confidence
    # ----------------------------------------------------------

    ax4 = plt.subplot2grid((2, 3), (1, 0))

    correct = df[df["correct"]]["confidence"]

    incorrect = df[
        ~df["correct"]
    ]["confidence"]

    ax4.hist(
        correct,
        alpha=0.7,
        bins=10,
        label="Correct",
    )

    ax4.hist(
        incorrect,
        alpha=0.7,
        bins=10,
        label="Incorrect",
    )

    ax4.set_title("Confidence Distribution")
    ax4.set_xlabel("Confidence")
    ax4.legend()

    # ----------------------------------------------------------
    # Decision times
    # ----------------------------------------------------------

    ax5 = plt.subplot2grid((2, 3), (1, 1), colspan=2)

    colors = [
        "tab:green" if c else "tab:red"
        for c in df["correct"]
    ]

    ax5.bar(
        df["trial_number"],
        df["decision_time"],
        color=colors,
    )

    ax5.set_xlabel("Trial")
    ax5.set_ylabel("Decision Time (s)")
    ax5.set_title("Decision Time per Trial")

    plt.tight_layout()

    plt.show()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    if args.file is None:
        path = find_latest_log()

    else:
        path = Path(args.file)

    print(
        f"Loading {path}"
    )

    df = load_data(path)

    plot_dashboard(df)


if __name__ == "__main__":
    main()