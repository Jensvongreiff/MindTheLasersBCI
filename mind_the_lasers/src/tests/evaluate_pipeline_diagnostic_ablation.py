"""
Ablation test for the XDF -> windows -> CSP -> LDA path.

Runs three variants on the exact same epoch-level holdout split:
1. Current behavior: broadband raw windows -> CSP -> ordinary LDA
2. Diagnostic preprocessing: 8-30 Hz zero-phase band-pass + CAR -> CSP -> ordinary LDA
3. Same preprocessing -> regularized CSP -> shrinkage LDA

The zero-phase filter in variants 2/3 is intentionally an OFFLINE diagnostic.
It answers whether spectral/reference preprocessing is the cause of failure.
For production online use, replace it with a stateful causal SOS filter that is
applied continuously rather than restarting the filter on every window.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from mind_the_lasers.src.pipeline.feature_extraction import CSPWrapper
from mind_the_lasers.src.pipeline.model import LDAWrapper, LDAWrapperTest
from mind_the_lasers.src.pipeline.pipeline_constructor import BCIPipeline
from mind_the_lasers.src.pipeline.signal import EEGDataLoaderOffline, EEGWindow


DATASET_PATH = (
    r"D:/Programming/BCI_Practical/practical-ss26-team4/"
    r"data/sub-P999/sub-P999_ses-S009_task-Default_run-001_eeg.xdf"
)
WINDOW_LENGTH_SECONDS = 1.0
TEST_SIZE = 0.3
RANDOM_STATE = 42
OUTPUT_PATH = Path("mind_the_lasers/reports/csp_lda_ablation.json")

LABEL_NAMES = {0: "left", 1: "right", 2: "rest"}
CLASS_ORDER = [0, 1, 2]


def preprocess_offline_diagnostic(X: np.ndarray, fs: float) -> np.ndarray:
    """Offline-only 8-30 Hz zero-phase filtering followed by sample-wise CAR."""
    sos = butter(4, [8.0, 30.0], btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, np.asarray(X, dtype=float), axis=-1)
    return filtered - filtered.mean(axis=1, keepdims=True)


def summarize_signal(name: str, X: np.ndarray) -> None:
    finite = np.isfinite(X)
    channel_std = np.std(X, axis=(0, 2))
    print(f"\n{name} signal audit")
    print(f"  shape: {X.shape}")
    print(f"  finite: {finite.all()}")
    print(f"  absolute max: {np.nanmax(np.abs(X)):.3e}")
    print(f"  median channel std: {np.nanmedian(channel_std):.3e}")
    print(f"  min/max channel std: {np.nanmin(channel_std):.3e} / {np.nanmax(channel_std):.3e}")



def audit_event_timing(loader: EEGDataLoaderOffline) -> None:
    """Check whether the configured 0.5-2.5 s epochs cross the next task cue."""
    annotations = loader.raw_data.annotations
    task_markers = set(loader.events.values())
    task_events = sorted(
        (float(onset), str(description))
        for onset, description in zip(annotations.onset, annotations.description)
        if str(description) in task_markers
    )

    print("\nTask-marker timing audit")
    print(f"  task events: {len(task_events)}")
    if len(task_events) < 2:
        print("  Not enough task markers to inspect inter-event spacing.")
        return

    gaps_by_marker: dict[str, list[float]] = {marker: [] for marker in task_markers}
    overlaps = []
    epoch_end_offset = 2.5

    for (onset, marker), (next_onset, next_marker) in zip(task_events[:-1], task_events[1:]):
        gap = next_onset - onset
        gaps_by_marker[marker].append(gap)
        if gap < epoch_end_offset:
            overlaps.append((onset, marker, gap, next_marker))

    for marker in loader.events.values():
        gaps = np.asarray(gaps_by_marker[marker], dtype=float)
        if gaps.size:
            print(
                f"  {marker}: next-task-event gap "
                f"min/median={gaps.min():.3f}/{np.median(gaps):.3f} s"
            )

    print(
        "  epochs whose 2.5 s endpoint crosses the next task marker: "
        f"{len(overlaps)} / {len(task_events) - 1}"
    )
    for onset, marker, gap, next_marker in overlaps[:10]:
        print(
            f"    t={onset:.3f}: {marker} -> {next_marker} after {gap:.3f} s"
        )


def evaluate_variant(
    name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    loader: EEGDataLoaderOffline,
    *,
    shrinkage_lda: bool,
    csp_reg: str | None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"{name}_") as tmp:
        csp = CSPWrapper(str(Path(tmp) / "csp.pkl"), n_components=4)

        # CSPWrapper currently hard-codes reg=None. Set the desired implementation
        # by fitting an equivalent MNE CSP object directly when regularization is requested.
        if csp_reg is not None:
            from mne.decoding import CSP

            csp.csp = CSP(
                n_components=4,
                reg=csp_reg,
                log=True,
                norm_trace=False,
                cov_est="epoch",
            )
            csp.csp.fit(X_train, y_train)
            csp.fit = lambda *args, **kwargs: None  # BCIPipeline must not refit it.

        classifier = (
            LDAWrapperTest(str(Path(tmp) / "lda.pkl"))
            if shrinkage_lda
            else LDAWrapper(str(Path(tmp) / "lda.pkl"))
        )

        pipeline = BCIPipeline(
            feature_step=csp,
            classifier_step=classifier,
            data_loader=loader,
        )

        if csp_reg is None:
            pipeline.calibrate(X_train, y_train)
        else:
            features_train = csp.transform(X_train, sampling_rate=loader.sampling_rate)
            classifier.fit(features_train, y_train)

        train_features = csp.transform(X_train, sampling_rate=loader.sampling_rate)
        train_predictions = classifier.lda.predict(train_features)

        test_predictions: list[int] = []
        probability_rows: list[np.ndarray] = []

        start = time.perf_counter()
        for data in X_test:
            probabilities = pipeline.process_window(
                EEGWindow(
                    data=data,
                    sampling_rate=loader.sampling_rate,
                    timestamp=0.0,
                )
            )
            row = np.asarray(
                [probabilities[LABEL_NAMES[label]] for label in CLASS_ORDER],
                dtype=float,
            )
            probability_rows.append(row)
            test_predictions.append(int(np.argmax(row)))
        elapsed = time.perf_counter() - start

        test_predictions_array = np.asarray(test_predictions)
        probability_array = np.vstack(probability_rows)

        result: dict[str, Any] = {
            "name": name,
            "train_accuracy": float(accuracy_score(y_train, train_predictions)),
            "window_accuracy": float(accuracy_score(y_test, test_predictions_array)),
            "window_balanced_accuracy": float(
                balanced_accuracy_score(y_test, test_predictions_array)
            ),
            "window_macro_f1": float(
                f1_score(y_test, test_predictions_array, average="macro", zero_division=0)
            ),
            "window_confusion_matrix": confusion_matrix(
                y_test, test_predictions_array, labels=CLASS_ORDER
            ).tolist(),
            "mean_inference_ms": 1000.0 * elapsed / len(X_test),
        }

        # load_data() creates all windows from one source epoch contiguously.
        n_windows = int(loader.n_windows_per_epoch)
        if n_windows > 0 and len(y_test) % n_windows == 0:
            grouped_truth = y_test.reshape(-1, n_windows)
            if not np.all(grouped_truth == grouped_truth[:, :1]):
                raise RuntimeError("Window grouping no longer corresponds to source epochs.")

            epoch_truth = grouped_truth[:, 0]
            epoch_probabilities = probability_array.reshape(-1, n_windows, 3).mean(axis=1)
            epoch_predictions = epoch_probabilities.argmax(axis=1)

            result.update(
                {
                    "windows_per_source_epoch": n_windows,
                    "source_epoch_count": int(len(epoch_truth)),
                    "epoch_aggregated_accuracy": float(
                        accuracy_score(epoch_truth, epoch_predictions)
                    ),
                    "epoch_aggregated_balanced_accuracy": float(
                        balanced_accuracy_score(epoch_truth, epoch_predictions)
                    ),
                    "epoch_aggregated_macro_f1": float(
                        f1_score(
                            epoch_truth,
                            epoch_predictions,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    "epoch_aggregated_confusion_matrix": confusion_matrix(
                        epoch_truth, epoch_predictions, labels=CLASS_ORDER
                    ).tolist(),
                }
            )

        return result


def main() -> None:
    loader = EEGDataLoaderOffline(
        data_path=DATASET_PATH,
        window_length=WINDOW_LENGTH_SECONDS,
        stride=0.1,
    )
    X_train, X_test, y_train, y_test = loader.load_data(
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print(f"Sampling rate: {loader.sampling_rate} Hz")
    print(f"Channels ({len(loader.channel_labels)}): {loader.channel_labels}")
    print(f"Windows per source epoch: {loader.n_windows_per_epoch}")
    print(f"Train class counts: {np.bincount(y_train, minlength=3)}")
    print(f"Test class counts:  {np.bincount(y_test, minlength=3)}")
    audit_event_timing(loader)

    summarize_signal("Raw training", X_train)

    X_train_filtered = preprocess_offline_diagnostic(X_train, loader.sampling_rate)
    X_test_filtered = preprocess_offline_diagnostic(X_test, loader.sampling_rate)
    summarize_signal("Filtered + CAR training", X_train_filtered)

    results = [
        evaluate_variant(
            "current_raw_csp_lda",
            X_train,
            X_test,
            y_train,
            y_test,
            loader,
            shrinkage_lda=False,
            csp_reg=None,
        ),
        evaluate_variant(
            "filtered_car_csp_lda",
            X_train_filtered,
            X_test_filtered,
            y_train,
            y_test,
            loader,
            shrinkage_lda=False,
            csp_reg=None,
        ),
        evaluate_variant(
            "filtered_car_regularized_csp_shrinkage_lda",
            X_train_filtered,
            X_test_filtered,
            y_train,
            y_test,
            loader,
            shrinkage_lda=True,
            csp_reg="ledoit_wolf",
        ),
    ]

    print("\n" + "=" * 88)
    print("ABLATION RESULTS")
    print("=" * 88)
    for result in results:
        print(f"\n{result['name']}")
        print(f"  train accuracy:           {result['train_accuracy']:.4f}")
        print(f"  window accuracy:          {result['window_accuracy']:.4f}")
        print(f"  window balanced accuracy: {result['window_balanced_accuracy']:.4f}")
        if "epoch_aggregated_accuracy" in result:
            print(f"  epoch-aggregated accuracy:{result['epoch_aggregated_accuracy']:9.4f}")
            print(
                "  epoch balanced accuracy:  "
                f"{result['epoch_aggregated_balanced_accuracy']:.4f}"
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
