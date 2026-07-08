"""
Visualize decoder metrics exported as JSON reports.

The plots deliberately separate incompatible units:
- rates in [0, 1]
- counts
- latency in milliseconds
- confidence summaries

This keeps the figures interpretable for a scientific report and avoids
putting accuracy, latency, and class counts on one misleading axis.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "mind_the_lasers" / "reports"
DEFAULT_OUT_DIR = DEFAULT_REPORT_DIR / "figures"
DEFAULT_PATTERN = "decoder_summary_*.json"
DEFAULT_CLASS_ORDER = ["left", "rest", "right"]


RATE_METRICS = [
    ("overall_accuracy", "Overall accuracy", True),
    ("balanced_accuracy", "Balanced accuracy", True),
    ("macro_f1_score", "Macro F1", True),
    ("rejection_rate", "Rejection rate", False),
    ("false_activation_rate", "Rest false activation", False),
]

CLASS_METRICS = [
    ("class_wise_accuracy", "One-vs-rest accuracy"),
    ("precision_per_class", "Precision"),
    ("recall_per_class", "Recall"),
    ("f1_score_per_class", "F1 score"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create scientifically separated figures from decoder summary "
            "JSON reports."
        )
    )
    parser.add_argument(
        "--json",
        nargs="*",
        type=Path,
        default=None,
        help=(
            "Specific JSON report(s) to visualize. If omitted, all "
            "decoder_summary_*.json files in --dir are used."
        ),
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory searched when --json is omitted.",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help="Glob pattern used inside --dir when --json is omitted.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for generated figures and flattened CSV output.",
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output figure format.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively after saving.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object.")

    return data


def infer_model_name(path: Path, report: dict[str, Any]) -> str:
    if "baseline" in report:
        return str(report["baseline"])

    name = path.stem
    name = re.sub(r"^decoder_(summary|metrics)_", "", name)
    name = re.sub(r"_(latest|\d+)$", "", name)
    return name


def discover_reports(args: argparse.Namespace) -> list[tuple[str, Path, dict[str, Any]]]:
    if args.json:
        paths = args.json
    else:
        paths = sorted(args.dir.glob(args.pattern))

    if not paths:
        raise FileNotFoundError(
            f"No JSON reports found. Searched {args.dir} with {args.pattern!r}."
        )

    reports = []
    for path in paths:
        report = load_json(path)
        label = infer_model_name(path, report)
        reports.append((label, path, report))

    return reports


def class_order(report: dict[str, Any]) -> list[str]:
    candidates = []
    for key in [
        "class_wise_accuracy",
        "precision_per_class",
        "recall_per_class",
        "f1_score_per_class",
        "predictions_per_class",
    ]:
        value = report.get(key)
        if isinstance(value, dict):
            candidates.extend(str(name) for name in value.keys())

    unique = list(dict.fromkeys(candidates))

    if set(DEFAULT_CLASS_ORDER).issubset(unique):
        return DEFAULT_CLASS_ORDER

    if unique:
        return unique

    matrix = report.get("confusion_matrix", [])
    if isinstance(matrix, list) and matrix:
        return [f"class_{idx}" for idx in range(len(matrix))]

    return []


def confusion_matrix(report: dict[str, Any]) -> np.ndarray:
    matrix = np.asarray(report.get("confusion_matrix", []), dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Expected a square confusion_matrix in the report.")

    return matrix


def get_false_activation_rate(report: dict[str, Any]) -> float | None:
    value = report.get("false_activations_during_rest")
    if isinstance(value, dict) and "rate" in value:
        return float(value["rate"])
    return None


def get_false_activation_count(report: dict[str, Any]) -> float | None:
    value = report.get("false_activations_during_rest")
    if isinstance(value, dict) and "count" in value:
        return float(value["count"])
    return None


def rate_value(report: dict[str, Any], metric: str) -> float | None:
    if metric == "false_activation_rate":
        return get_false_activation_rate(report)

    value = report.get(metric)
    if value is None:
        return None

    return float(value)


def safe_filename(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")


def save_figure(fig: plt.Figure, out_path: Path, show: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.25)

    if show:
        plt.show()

    plt.close(fig)
    print(f"[+] Saved {out_path}")


def set_plot_style() -> None:
    sns.set_theme(
        context="notebook",
        style="whitegrid",
        palette="colorblind",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
        },
    )


def flatten_reports(
    reports: list[tuple[str, Path, dict[str, Any]]]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for model, path, report in reports:
        labels = class_order(report)
        cm = confusion_matrix(report)
        total = float(cm.sum())

        for metric, display_name, higher_is_better in RATE_METRICS:
            value = rate_value(report, metric)
            if value is not None:
                rows.append(
                    {
                        "model": model,
                        "source_file": str(path),
                        "metric_group": "rate",
                        "metric": metric,
                        "display_name": display_name,
                        "class": "",
                        "value": value,
                        "unit": "proportion",
                        "higher_is_better": higher_is_better,
                    }
                )

        false_activation_count = get_false_activation_count(report)
        if false_activation_count is not None:
            rows.append(
                {
                    "model": model,
                    "source_file": str(path),
                    "metric_group": "count",
                    "metric": "false_activations_during_rest_count",
                    "display_name": "False activations during rest",
                    "class": "rest",
                    "value": false_activation_count,
                    "unit": "count",
                    "higher_is_better": False,
                }
            )

        rows.append(
            {
                "model": model,
                "source_file": str(path),
                "metric_group": "count",
                "metric": "n_windows",
                "display_name": "Evaluated windows",
                "class": "",
                "value": total,
                "unit": "count",
                "higher_is_better": "",
            }
        )

        if "mtl_suitability_score" in report:
            rows.append(
                {
                    "model": model,
                    "source_file": str(path),
                    "metric_group": "composite",
                    "metric": "mtl_suitability_score",
                    "display_name": "MTL suitability score",
                    "class": "",
                    "value": float(report["mtl_suitability_score"]),
                    "unit": "score",
                    "higher_is_better": True,
                }
            )

        for key, display_name in CLASS_METRICS:
            values = report.get(key, {})
            if not isinstance(values, dict):
                continue

            for cls in labels:
                if cls not in values:
                    continue

                rows.append(
                    {
                        "model": model,
                        "source_file": str(path),
                        "metric_group": "class_metric",
                        "metric": key,
                        "display_name": display_name,
                        "class": cls,
                        "value": float(values[cls]),
                        "unit": "proportion",
                        "higher_is_better": True,
                    }
                )

        predictions = report.get("predictions_per_class", {})
        if isinstance(predictions, dict):
            pred_total = float(sum(float(v) for v in predictions.values()))
            for cls in labels:
                if cls not in predictions:
                    continue

                count = float(predictions[cls])
                rows.append(
                    {
                        "model": model,
                        "source_file": str(path),
                        "metric_group": "prediction_count",
                        "metric": "predictions_per_class",
                        "display_name": "Predicted class count",
                        "class": cls,
                        "value": count,
                        "unit": "count",
                        "higher_is_better": "",
                    }
                )
                if pred_total > 0:
                    rows.append(
                        {
                            "model": model,
                            "source_file": str(path),
                            "metric_group": "prediction_fraction",
                            "metric": "predictions_per_class_fraction",
                            "display_name": "Predicted class fraction",
                            "class": cls,
                            "value": count / pred_total,
                            "unit": "proportion",
                            "higher_is_better": "",
                        }
                    )

        confidence = report.get("confidence_distribution", {})
        if isinstance(confidence, dict):
            for stat, value in confidence.items():
                rows.append(
                    {
                        "model": model,
                        "source_file": str(path),
                        "metric_group": "confidence",
                        "metric": f"confidence_{stat}",
                        "display_name": f"Confidence {stat}",
                        "class": "",
                        "value": float(value),
                        "unit": "proportion",
                        "higher_is_better": "",
                    }
                )

        latency = report.get("decision_latency", {})
        if isinstance(latency, dict):
            for stat, value in latency.items():
                rows.append(
                    {
                        "model": model,
                        "source_file": str(path),
                        "metric_group": "latency",
                        "metric": f"decision_latency_{stat}",
                        "display_name": stat.replace("_", " ").title(),
                        "class": "",
                        "value": float(value),
                        "unit": "ms",
                        "higher_is_better": False,
                    }
                )

        for true_idx, true_class in enumerate(labels):
            if true_idx >= cm.shape[0]:
                continue

            row_sum = float(cm[true_idx].sum())
            for pred_idx, pred_class in enumerate(labels):
                if pred_idx >= cm.shape[1]:
                    continue

                count = float(cm[true_idx, pred_idx])
                rows.append(
                    {
                        "model": model,
                        "source_file": str(path),
                        "metric_group": "confusion_matrix",
                        "metric": "confusion_matrix_count",
                        "class": true_class,
                        "predicted_class": pred_class,
                        "value": count,
                        "row_fraction": count / row_sum if row_sum else np.nan,
                        "unit": "count",
                        "higher_is_better": true_class == pred_class,
                    }
                )

    return pd.DataFrame(rows)


def plot_overview(
    reports: list[tuple[str, Path, dict[str, Any]]],
    out_dir: Path,
    output_format: str,
    show: bool,
) -> None:
    rows = []
    latency_rows = []
    confidence_rows = []
    prediction_rows = []
    suitability_rows = []
    model_totals = {}

    for model, _path, report in reports:
        labels = class_order(report)
        cm = confusion_matrix(report)
        model_totals[model] = int(cm.sum())

        for metric, display_name, higher_is_better in RATE_METRICS:
            value = rate_value(report, metric)
            if value is None:
                continue
            rows.append(
                {
                    "model": model,
                    "metric": display_name,
                    "value": value,
                    "higher_is_better": higher_is_better,
                }
            )

        latency = report.get("decision_latency", {})
        if isinstance(latency, dict):
            for key, display_name in [
                ("mean_ms", "Mean"),
                ("95th_percentile_ms", "95th percentile"),
            ]:
                if key in latency:
                    latency_rows.append(
                        {
                            "model": model,
                            "metric": display_name,
                            "value": float(latency[key]),
                        }
                    )

        confidence = report.get("confidence_distribution", {})
        if isinstance(confidence, dict):
            confidence_rows.append(
                {
                    "model": model,
                    "mean": float(confidence.get("mean", np.nan)),
                    "median": float(confidence.get("median", np.nan)),
                    "std": float(confidence.get("std", np.nan)),
                    "min": float(confidence.get("min", np.nan)),
                    "max": float(confidence.get("max", np.nan)),
                }
            )

        predictions = report.get("predictions_per_class", {})
        if isinstance(predictions, dict):
            total_predictions = float(sum(float(v) for v in predictions.values()))
            for cls in labels:
                if cls in predictions and total_predictions > 0:
                    prediction_rows.append(
                        {
                            "model": model,
                            "class": cls,
                            "fraction": float(predictions[cls]) / total_predictions,
                        }
                    )

        if "mtl_suitability_score" in report:
            suitability_rows.append(
                {
                    "model": model,
                    "score": float(report["mtl_suitability_score"]),
                }
            )

    rate_df = pd.DataFrame(rows)
    latency_df = pd.DataFrame(latency_rows)
    confidence_df = pd.DataFrame(confidence_rows)
    prediction_df = pd.DataFrame(prediction_rows)
    suitability_df = pd.DataFrame(suitability_rows)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    fig.suptitle(
        "Decoder Metrics Overview",
        fontsize=18,
        fontweight="bold",
    )

    ax = axes[0, 0]
    quality = rate_df[rate_df["higher_is_better"] == True]
    sns.barplot(data=quality, x="metric", y="value", hue="model", ax=ax)
    n_classes = max(len(class_order(reports[0][2])), 1)
    ax.axhline(
        1 / n_classes,
        color="0.35",
        linestyle="--",
        linewidth=1,
        label=f"{n_classes}-class chance",
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Proportion")
    ax.set_title("Classification quality")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Model", fontsize=8)

    ax = axes[0, 1]
    risk = rate_df[rate_df["higher_is_better"] == False]
    sns.barplot(data=risk, x="metric", y="value", hue="model", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Proportion")
    ax.set_title("Control risk metrics (lower is better)")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Model", fontsize=8)

    ax = axes[0, 2]
    if not latency_df.empty:
        sns.barplot(data=latency_df, x="metric", y="value", hue="model", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Milliseconds")
    ax.set_title("Decision latency")
    ax.legend(title="Model", fontsize=8)

    ax = axes[1, 0]
    if not prediction_df.empty:
        pivot = prediction_df.pivot(index="model", columns="class", values="fraction")
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Prediction fraction"},
            ax=ax,
        )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("")
    ax.set_title("Prediction distribution")

    ax = axes[1, 1]
    if not confidence_df.empty:
        y = np.arange(len(confidence_df))
        ax.hlines(
            y=y,
            xmin=confidence_df["min"],
            xmax=confidence_df["max"],
            color="0.75",
            linewidth=5,
            label="min-max",
        )
        ax.errorbar(
            confidence_df["mean"],
            y,
            xerr=confidence_df["std"],
            fmt="o",
            color="tab:blue",
            capsize=4,
            label="mean +- sd",
        )
        ax.scatter(
            confidence_df["median"],
            y,
            marker="|",
            s=180,
            color="black",
            label="median",
        )
        ax.set_yticks(y)
        ax.set_yticklabels(confidence_df["model"])
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)
    ax.set_xlabel("Confidence")
    ax.set_title("Confidence summary")

    ax = axes[1, 2]
    if not suitability_df.empty:
        sns.barplot(data=suitability_df, x="model", y="score", ax=ax)
        ax.set_ylim(0, 100)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_title("MTL suitability composite")
    ax.tick_params(axis="x", rotation=25)

    subtitle = ", ".join(
        f"{model}: n={total}" for model, total in model_totals.items()
    )
    fig.text(
        0.5,
        -0.035,
        (
            f"Evaluated windows: {subtitle}. Rates are proportions; latency "
            "is shown separately to preserve units."
        ),
        ha="center",
        fontsize=10,
    )

    save_figure(
        fig,
        out_dir / f"decoder_metrics_overview.{output_format}",
        show,
    )


def plot_confusion(
    model: str,
    report: dict[str, Any],
    out_dir: Path,
    output_format: str,
    show: bool,
) -> None:
    labels = class_order(report)
    cm = confusion_matrix(report)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_norm = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    annotations = np.empty(cm.shape, dtype=object)
    for row_idx in range(cm.shape[0]):
        for col_idx in range(cm.shape[1]):
            annotations[row_idx, col_idx] = (
                f"{int(cm[row_idx, col_idx])}\n{row_norm[row_idx, col_idx]:.1%}"
            )

    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    sns.heatmap(
        row_norm,
        annot=annotations,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Row-normalized fraction"},
        ax=ax,
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(
        f"{model}: confusion matrix\ncount and percentage within true class"
    )

    save_figure(
        fig,
        out_dir / f"{safe_filename(model)}_confusion_matrix.{output_format}",
        show,
    )


def plot_class_metrics(
    model: str,
    report: dict[str, Any],
    out_dir: Path,
    output_format: str,
    show: bool,
) -> None:
    labels = class_order(report)
    rows = []
    row_labels = []

    for key, display_name in CLASS_METRICS:
        values = report.get(key, {})
        if not isinstance(values, dict):
            continue

        rows.append(
            [
                float(values.get(cls, np.nan))
                for cls in labels
            ]
        )
        row_labels.append(display_name)

    if not rows:
        return

    data = pd.DataFrame(
        rows,
        index=row_labels,
        columns=labels,
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        cmap="mako",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Metric value"},
        ax=ax,
    )
    ax.set_xlabel("Class")
    ax.set_ylabel("")
    ax.set_title(f"{model}: class-wise decoder behavior")

    save_figure(
        fig,
        out_dir / f"{safe_filename(model)}_class_metrics.{output_format}",
        show,
    )


def plot_prediction_bias(
    model: str,
    report: dict[str, Any],
    out_dir: Path,
    output_format: str,
    show: bool,
) -> None:
    labels = class_order(report)
    cm = confusion_matrix(report)
    actual_counts = cm.sum(axis=1)
    predictions = report.get("predictions_per_class", {})

    if not isinstance(predictions, dict):
        return

    actual_total = float(actual_counts.sum())
    predicted_total = float(sum(float(value) for value in predictions.values()))

    rows = []
    for idx, cls in enumerate(labels):
        if actual_total > 0:
            rows.append(
                {
                    "class": cls,
                    "source": "True labels",
                    "fraction": float(actual_counts[idx]) / actual_total,
                }
            )
        if cls in predictions and predicted_total > 0:
            rows.append(
                {
                    "class": cls,
                    "source": "Predictions",
                    "fraction": float(predictions[cls]) / predicted_total,
                }
            )

    if not rows:
        return

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    sns.barplot(data=df, x="class", y="fraction", hue="source", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Class")
    ax.set_ylabel("Fraction")
    ax.set_title(f"{model}: class balance and prediction bias")
    ax.legend(title="")

    save_figure(
        fig,
        out_dir / f"{safe_filename(model)}_prediction_bias.{output_format}",
        show,
    )


def plot_single_report_dashboard(
    model: str,
    report: dict[str, Any],
    out_dir: Path,
    output_format: str,
    show: bool,
) -> None:
    labels = class_order(report)
    cm = confusion_matrix(report)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_norm = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    rate_rows = []
    for metric, display_name, higher_is_better in RATE_METRICS:
        value = rate_value(report, metric)
        if value is not None:
            rate_rows.append(
                {
                    "metric": display_name,
                    "value": value,
                    "kind": "Higher is better" if higher_is_better else "Lower is better",
                }
            )

    class_data = []
    class_row_labels = []
    for key, display_name in CLASS_METRICS:
        values = report.get(key, {})
        if isinstance(values, dict):
            class_data.append([float(values.get(cls, np.nan)) for cls in labels])
            class_row_labels.append(display_name)

    class_df = pd.DataFrame(
        class_data,
        index=class_row_labels,
        columns=labels,
    )

    predictions = report.get("predictions_per_class", {})
    actual_counts = cm.sum(axis=1)
    predicted_counts = np.array(
        [
            float(predictions.get(cls, np.nan))
            if isinstance(predictions, dict)
            else np.nan
            for cls in labels
        ]
    )

    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    grid = fig.add_gridspec(3, 3)
    fig.suptitle(
        f"{model}: decoder metrics dashboard",
        fontsize=18,
        fontweight="bold",
    )

    ax = fig.add_subplot(grid[0, 0])
    rate_df = pd.DataFrame(rate_rows)
    sns.barplot(data=rate_df, x="value", y="metric", hue="kind", dodge=False, ax=ax)
    ax.axvline(1 / max(len(labels), 1), color="0.35", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Proportion")
    ax.set_ylabel("")
    ax.set_title("Rate metrics")
    ax.legend(title="")

    ax = fig.add_subplot(grid[0, 1])
    annotations = np.empty(cm.shape, dtype=object)
    for row_idx in range(cm.shape[0]):
        for col_idx in range(cm.shape[1]):
            annotations[row_idx, col_idx] = (
                f"{int(cm[row_idx, col_idx])}\n{row_norm[row_idx, col_idx]:.1%}"
            )
    sns.heatmap(
        row_norm,
        annot=annotations,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")

    ax = fig.add_subplot(grid[0, 2])
    if not class_df.empty:
        sns.heatmap(
            class_df,
            annot=True,
            fmt=".2f",
            cmap="mako",
            vmin=0,
            vmax=1,
            cbar=False,
            ax=ax,
        )
    ax.set_xlabel("Class")
    ax.set_ylabel("")
    ax.set_title("Class-wise metrics")

    ax = fig.add_subplot(grid[1, 0])
    balance_rows = []
    actual_total = actual_counts.sum()
    predicted_total = np.nansum(predicted_counts)
    for idx, cls in enumerate(labels):
        if actual_total > 0:
            balance_rows.append(
                {
                    "class": cls,
                    "source": "True labels",
                    "fraction": actual_counts[idx] / actual_total,
                }
            )
        if predicted_total > 0:
            balance_rows.append(
                {
                    "class": cls,
                    "source": "Predictions",
                    "fraction": predicted_counts[idx] / predicted_total,
                }
            )
    sns.barplot(data=pd.DataFrame(balance_rows), x="class", y="fraction", hue="source", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Class")
    ax.set_ylabel("Fraction")
    ax.set_title("Prediction bias check")
    ax.legend(title="")

    ax = fig.add_subplot(grid[1, 1])
    confidence = report.get("confidence_distribution", {})
    if isinstance(confidence, dict) and confidence:
        stats = pd.DataFrame(
            {
                "stat": list(confidence.keys()),
                "value": [float(value) for value in confidence.values()],
            }
        )
        sns.barplot(data=stats, x="stat", y="value", ax=ax)
        ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Confidence")
    ax.set_title("Confidence summary")
    ax.tick_params(axis="x", rotation=25)

    ax = fig.add_subplot(grid[1, 2])
    latency = report.get("decision_latency", {})
    rows = []
    if isinstance(latency, dict):
        for key, label in [
            ("mean_ms", "Mean"),
            ("95th_percentile_ms", "95th percentile"),
        ]:
            if key in latency:
                rows.append({"stat": label, "value": float(latency[key])})
    if rows:
        latency_df = pd.DataFrame(rows)
        sns.barplot(data=latency_df, x="stat", y="value", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Milliseconds")
    ax.set_title("Decision latency")
    ax.tick_params(axis="x", rotation=25)

    ax = fig.add_subplot(grid[2, 0])
    count_rows = [
        {"metric": "Evaluated windows", "value": int(cm.sum())},
    ]
    false_count = get_false_activation_count(report)
    if false_count is not None:
        count_rows.append(
            {
                "metric": "False rest activations",
                "value": int(false_count),
            }
        )
    count_df = pd.DataFrame(count_rows)
    sns.barplot(data=count_df, x="value", y="metric", ax=ax)
    ax.set_xlabel("Count")
    ax.set_ylabel("")
    ax.set_title("Count metrics")

    ax = fig.add_subplot(grid[2, 1])
    if isinstance(predictions, dict):
        pred_df = pd.DataFrame(
            {
                "class": labels,
                "count": [float(predictions.get(cls, 0)) for cls in labels],
            }
        )
        sns.barplot(data=pred_df, x="class", y="count", ax=ax)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Count")
    ax.set_title("Predictions per class")

    ax = fig.add_subplot(grid[2, 2])
    if "mtl_suitability_score" in report:
        score_df = pd.DataFrame(
            {
                "metric": ["MTL suitability"],
                "value": [float(report["mtl_suitability_score"])],
            }
        )
        sns.barplot(data=score_df, x="metric", y="value", ax=ax)
        ax.set_ylim(0, 100)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_title("Composite score")

    total = int(cm.sum())
    footnote = f"n={total} evaluated windows"
    if false_count is not None:
        footnote += f"; false rest activations={int(false_count)}"

    fig.text(
        0.5,
        -0.025,
        (
            f"{footnote}. Confusion percentages are row-normalized by true "
            "class; confidence is summary-only because the JSON has no raw "
            "per-window probabilities."
        ),
        ha="center",
        fontsize=10,
    )

    save_figure(
        fig,
        out_dir / f"{safe_filename(model)}_decoder_metrics_dashboard.{output_format}",
        show,
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    reports = discover_reports(args)
    flat_df = flatten_reports(reports)
    csv_path = args.out_dir / "decoder_metrics_flattened.csv"
    flat_df.to_csv(csv_path, index=False)
    print(f"[+] Saved {csv_path}")

    plot_overview(
        reports=reports,
        out_dir=args.out_dir,
        output_format=args.format,
        show=args.show,
    )

    for model, _path, report in reports:
        plot_single_report_dashboard(
            model=model,
            report=report,
            out_dir=args.out_dir,
            output_format=args.format,
            show=args.show,
        )
        plot_confusion(
            model=model,
            report=report,
            out_dir=args.out_dir,
            output_format=args.format,
            show=args.show,
        )
        plot_class_metrics(
            model=model,
            report=report,
            out_dir=args.out_dir,
            output_format=args.format,
            show=args.show,
        )
        plot_prediction_bias(
            model=model,
            report=report,
            out_dir=args.out_dir,
            output_format=args.format,
            show=args.show,
        )


if __name__ == "__main__":
    main()
