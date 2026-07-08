"""
mind_the_lasers/src/tests/evaluate_pipeline_holdout.py

Basic holdout evaluation of a BCI pipeline.

This evaluates the raw model/pipeline output before:
- temporal smoothing
- confidence rejection
- online decision logic
- repeated sliding-window inference
- cross-validation
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from mind_the_lasers.src.pipeline.signal import (
    EEGDataLoaderOffline,
    EEGWindow,
)
from mind_the_lasers.src.pipeline.pipeline_config import build_pipeline
from mind_the_lasers.src.pipeline.decoder_metrics import DecoderEvaluator


# ============================================================
# Configuration
# ============================================================

DATASET_PATH = (
    r"D:/Programming/BCI_Practical/practical-ss26-team4/"
    r"data/sub-P999/sub-P999_ses-S009_task-Default_run-001_eeg.xdf"
)

BASELINE = "csp-lda"
WINDOW_LENGTH_SECONDS = 1
TEST_SIZE = 0.2

OUTPUT_PATH = Path(
    "mind_the_lasers/reports/"
    f"basic_holdout_evaluation_{BASELINE}.json"
)

# This must match the label encoding produced by EEGDataLoaderOffline.
LABEL_MAP = {
    0: "left",
    1: "right",
    2: "rest",
}

# Explicit ordering for the confusion matrix and report.
CLASS_ORDER = ["left", "right", "rest"]


def label_to_name(label: Any) -> str:
    """Convert numeric or string labels into the pipeline class names."""

    if isinstance(label, (str, np.str_)):
        return str(label)

    numeric_label = int(label)

    if numeric_label not in LABEL_MAP:
        raise ValueError(
            f"Unknown numeric label {numeric_label}. "
            f"Expected one of {list(LABEL_MAP.keys())}."
        )

    return LABEL_MAP[numeric_label]


def make_json_serializable(value: Any) -> Any:
    """Recursively convert NumPy values into JSON-compatible values."""

    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_json_serializable(item) for item in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    return value


def print_class_distribution(name: str, labels: np.ndarray) -> None:
    """Print the number of samples belonging to each class."""

    label_names = [label_to_name(label) for label in labels]
    classes, counts = np.unique(label_names, return_counts=True)

    print(f"\n{name} class distribution:")
    for class_name, count in zip(classes, counts):
        print(f"  {class_name}: {count}")


def main() -> None:
    # ========================================================
    # 1. Load the existing holdout split
    # ========================================================

    print(f"Loading dataset:\n{DATASET_PATH}")

    data_loader = EEGDataLoaderOffline(
        data_path=DATASET_PATH,
        window_length=WINDOW_LENGTH_SECONDS,
    )

    X_train, X_test, y_train, y_test = data_loader.load_data(
        test_size=TEST_SIZE
    )

    print("\nDataset shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_test:  {y_test.shape}")

    print_class_distribution("Training", y_train)
    print_class_distribution("Test", y_test)

    if X_train.ndim != 3 or X_test.ndim != 3:
        raise ValueError(
            "Expected EEG arrays with shape "
            "(n_windows, n_channels, n_samples)."
        )

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train have different lengths.")

    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test have different lengths.")

    sampling_rate = data_loader.sampling_rate
    window_samples = X_train.shape[-1]

    print(f"\nSampling rate: {sampling_rate} Hz")
    print(f"Samples per model input: {window_samples}")
    print(
        "Input duration: "
        f"{window_samples / sampling_rate:.3f} seconds"
    )

    # ========================================================
    # 2. Build and train the pipeline once
    # ========================================================

    pipeline = build_pipeline(
        BASELINE,
        window_samples,
        suffix="basic_holdout",
    )

    # Required because BCIPipeline.calibrate currently uses the
    # loader for sampling rate and channel information.
    pipeline.data_loader = data_loader

    pipeline.calibrate(X_train, y_train)

    # ========================================================
    # 3. Evaluate each held-out model input exactly once
    # ========================================================

    evaluator = DecoderEvaluator()

    true_labels = []
    predicted_labels = []
    confidences = []
    latencies = []
    all_probabilities = []

    print(f"\nEvaluating {len(X_test)} held-out windows...")

    for window_data, numeric_truth in zip(X_test, y_test):
        truth = label_to_name(numeric_truth)

        window = EEGWindow(
            data=window_data,
            sampling_rate=sampling_rate,
            timestamp=time.time(),
            ground_truth=truth,
        )

        start_time = time.perf_counter()
        raw_probabilities = pipeline.process_window(window)
        latency = time.perf_counter() - start_time

        if not isinstance(raw_probabilities, dict):
            raise TypeError(
                "pipeline.process_window() must return a dictionary "
                "such as {'left': 0.2, 'right': 0.7, 'rest': 0.1}. "
                f"Received {type(raw_probabilities).__name__}."
            )

        probabilities = {
            label_to_name(label): float(probability)
            for label, probability in raw_probabilities.items()
        }

        if not probabilities:
            raise ValueError("The pipeline returned an empty probability dictionary.")

        prediction = max(probabilities, key=probabilities.get)
        confidence = probabilities[prediction]

        true_labels.append(truth)
        predicted_labels.append(prediction)
        confidences.append(confidence)
        latencies.append(latency)
        all_probabilities.append(probabilities)

        # No smoothing and no rejection are applied.
        evaluator.log_step(
            truth=truth,
            pred=prediction,
            confidence=confidence,
            rejected=False,
            latency=latency,
        )

    # ========================================================
    # 4. Calculate transparent, standard ML metrics
    # ========================================================

    accuracy = accuracy_score(true_labels, predicted_labels)
    balanced_accuracy = balanced_accuracy_score(
        true_labels,
        predicted_labels,
    )
    macro_f1 = f1_score(
        true_labels,
        predicted_labels,
        labels=CLASS_ORDER,
        average="macro",
        zero_division=0,
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=CLASS_ORDER,
    )

    sklearn_report = classification_report(
        true_labels,
        predicted_labels,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )

    print("\n" + "=" * 60)
    print(f"RAW HOLDOUT RESULTS: {BASELINE.upper()}")
    print("=" * 60)

    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")
    print(f"Mean confidence:   {np.mean(confidences):.4f}")
    print(f"Mean latency:      {np.mean(latencies) * 1000:.2f} ms")

    print(f"\nConfusion-matrix class order: {CLASS_ORDER}")
    print(matrix)

    print("\nClassification report:")
    print(
        classification_report(
            true_labels,
            predicted_labels,
            labels=CLASS_ORDER,
            zero_division=0,
        )
    )

    # Keep the existing project evaluator as a secondary report.
    decoder_report = evaluator.generate_report(
        f"{BASELINE}_basic_holdout",
        save=False,
    )

    complete_report = {
        "evaluation_type": "single_holdout_without_smoothing",
        "baseline": BASELINE,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "sampling_rate": sampling_rate,
        "window_samples": window_samples,
        "window_duration_seconds": window_samples / sampling_rate,
        "class_order": CLASS_ORDER,
        "standard_metrics": {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "macro_f1": macro_f1,
            "confusion_matrix": matrix,
            "classification_report": sklearn_report,
            "mean_confidence": np.mean(confidences),
            "mean_latency_ms": np.mean(latencies) * 1000,
        },
        "decoder_evaluator_report": decoder_report,
        "true_labels": true_labels,
        "predicted_labels": predicted_labels,
        "probabilities": all_probabilities,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            make_json_serializable(complete_report),
            file,
            indent=4,
        )

    print(f"\nReport saved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()