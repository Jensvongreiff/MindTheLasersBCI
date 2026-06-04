import argparse
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from collections import deque

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    brier_score_loss,
)

from week4.loading import load_bci2a_dataset
from week4.preprocessing import preprocess_bci2a_dataset
from week6.models import EEGNet, PyTorchClassifier

from week6.online_adaptation import (
    adabn_adapt_batch,
    predict_batch,
    set_bn_momentum,
    configure_tent,
    tent_adapt_and_predict,
)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
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


def get_target_data(data_cache, protocol, target_sub, subjects):
    if protocol == "cross-session":
        X_test = data_cache[target_sub]["E"]["X"]
        y_test = data_cache[target_sub]["E"]["y"]

    elif protocol == "cross-subject":
        X_test = data_cache[target_sub]["T"]["X"]
        y_test = data_cache[target_sub]["T"]["y"]

    else:
        raise ValueError(
            "Online adaptation is intended for cross-session or cross-subject."
        )

    return X_test, y_test


def load_model(weight_path, X_sample, nb_classes):
    clf = PyTorchClassifier(
        EEGNet,
        epochs=50,
        batch_size=16,
        lr=1e-3,
        mixup=False,
    )

    clf.load_weights(
        weight_path,
        X_sample,
        nb_classes=nb_classes,
    )

    return clf


def evaluate_online_adabn(
    model,
    X_test,
    y_test,
    batch_size,
    nb_classes,
    bn_momentum,
    device,
):
    set_bn_momentum(model, bn_momentum)

    all_probs = []
    all_preds = []
    all_true = []
    batch_metrics = []

    t0_total = time.perf_counter()

    for batch_idx, start in enumerate(range(0, len(X_test), batch_size)):
        end = start + batch_size

        X_batch = X_test[start:end]
        y_batch = y_test[start:end]

        X_t = torch.tensor(X_batch, dtype=torch.float32).unsqueeze(1).to(device)

        t0_batch = time.perf_counter()

        # AdaBN: update BN statistics on current target batch
        adabn_adapt_batch(model, X_t)

        # Predict using adapted BN statistics
        probs_t = predict_batch(model, X_t)

        batch_time = time.perf_counter() - t0_batch

        probs = probs_t.cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_probs.append(probs)
        all_preds.append(preds)
        all_true.append(y_batch)

        batch_metrics.append(
            {
                "batch_idx": batch_idx,
                "start_trial": start,
                "end_trial": min(end, len(X_test)),
                "batch_accuracy": accuracy_score(y_batch, preds),
                "batch_balanced_acc": balanced_accuracy_score(y_batch, preds),
                "batch_time_sec": batch_time,
            }
        )

    total_time = time.perf_counter() - t0_total

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_preds)
    probs = np.vstack(all_probs)

    if nb_classes == 2:
        y_prob = probs[:, 1]
        ece = expected_calibration_error(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)
        prob_data_y_true = y_true
        prob_data_y_prob = y_prob

    else:
        y_prob = np.max(probs, axis=1)
        correctness = (y_pred == y_true).astype(int)

        ece = expected_calibration_error(correctness, y_prob)

        y_true_onehot = np.eye(nb_classes)[y_true]
        brier = np.mean(np.sum((probs - y_true_onehot) ** 2, axis=1))

        prob_data_y_true = correctness
        prob_data_y_prob = y_prob

    cm = confusion_matrix(y_true, y_pred, labels=list(range(nb_classes)))

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "brier_score": brier,
        "ece": ece,
        "online_adaptation_time_sec": total_time,
        "online_time_per_trial_ms": 1000.0 * total_time / len(y_true),
    }

    for i in range(nb_classes):
        for j in range(nb_classes):
            metrics[f"cm_{i}{j}"] = int(cm[i, j])

    prob_data = {
        "y_true": prob_data_y_true.tolist(),
        "y_prob": prob_data_y_prob.tolist(),
    }

    return metrics, batch_metrics, prob_data

def evaluate_online_adabn_buffer(
    model,
    X_test,
    y_test,
    batch_size,
    nb_classes,
    bn_momentum,
    device,
    buffer_size=16,
):
    """
    Trial-level rolling-buffer AdaBN.

    One target trial arrives at a time.
    The buffer stores the latest `buffer_size` trials.
    AdaBN adapts using the buffer.
    Prediction is made only for the newest trial.
    """

    set_bn_momentum(model, bn_momentum)

    buffer_x = deque(maxlen=buffer_size)

    all_probs = []
    all_preds = []
    all_true = []
    step_metrics = []

    t0_total = time.perf_counter()

    for trial_idx in range(len(X_test)):
        X_trial = X_test[trial_idx : trial_idx + 1]
        y_trial = y_test[trial_idx : trial_idx + 1]

        X_current = torch.tensor(
            X_trial,
            dtype=torch.float32,
        ).unsqueeze(1).to(device)

        # Add newest trial, remove oldest automatically
        buffer_x.append(X_current)

        # Current rolling buffer: latest N trials
        X_buffer = torch.cat(list(buffer_x), dim=0)

        t0_step = time.perf_counter()

        # Adapt BN using latest N trials
        adabn_adapt_batch(model, X_buffer)

        # Predict only the newest trial
        probs_t = predict_batch(model, X_current)

        step_time = time.perf_counter() - t0_step

        probs = probs_t.cpu().numpy()
        pred = np.argmax(probs, axis=1)

        all_probs.append(probs)
        all_preds.append(pred)
        all_true.append(y_trial)

        step_metrics.append(
            {
                "trial_idx": trial_idx,
                "buffer_trials": X_buffer.shape[0],
                "correct": int(pred[0] == y_trial[0]),
                "step_time_sec": step_time,
            }
        )

    total_time = time.perf_counter() - t0_total

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_preds)
    probs = np.vstack(all_probs)

    if nb_classes == 2:
        y_prob = probs[:, 1]
        ece = expected_calibration_error(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)
        prob_data_y_true = y_true
        prob_data_y_prob = y_prob

    else:
        y_prob = np.max(probs, axis=1)
        correctness = (y_pred == y_true).astype(int)

        ece = expected_calibration_error(correctness, y_prob)

        y_true_onehot = np.eye(nb_classes)[y_true]
        brier = np.mean(np.sum((probs - y_true_onehot) ** 2, axis=1))

        prob_data_y_true = correctness
        prob_data_y_prob = y_prob

    cm = confusion_matrix(y_true, y_pred, labels=list(range(nb_classes)))

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "brier_score": brier,
        "ece": ece,
        "online_adaptation_time_sec": total_time,
        "online_time_per_trial_ms": 1000.0 * total_time / len(y_true),
        "buffer_size": buffer_size,
    }

    for i in range(nb_classes):
        for j in range(nb_classes):
            metrics[f"cm_{i}{j}"] = int(cm[i, j])

    prob_data = {
        "y_true": prob_data_y_true.tolist(),
        "y_prob": prob_data_y_prob.tolist(),
    }

    return metrics, step_metrics, prob_data


def evaluate_online_tent_buffer(
    model,
    X_test,
    y_test,
    nb_classes,
    device,
    tent_lr=1e-3,
    buffer_size=16,
):
    """
    Trial-level rolling-buffer TENT.

    One trial arrives at a time.
    A rolling buffer of recent trials is used for entropy minimization.
    Prediction is made only for the newest trial.
    """

    params = configure_tent(model)

    if len(params) == 0:
        raise RuntimeError("No BatchNorm affine parameters found for TENT.")

    optimizer = torch.optim.Adam(params, lr=tent_lr)

    buffer_x = deque(maxlen=buffer_size)

    all_probs = []
    all_preds = []
    all_true = []
    step_metrics = []

    t0_total = time.perf_counter()

    for trial_idx in range(len(X_test)):
        X_trial = X_test[trial_idx : trial_idx + 1]
        y_trial = y_test[trial_idx : trial_idx + 1]

        X_current = torch.tensor(
            X_trial,
            dtype=torch.float32,
        ).unsqueeze(1).to(device)

        buffer_x.append(X_current)
        X_buffer = torch.cat(list(buffer_x), dim=0)

        t0_step = time.perf_counter()

        # TENT adapts on rolling buffer
        _, loss_value = tent_adapt_and_predict(
            model=model,
            x_batch=X_buffer,
            optimizer=optimizer,
        )

        # Predict only newest trial after adaptation
        probs_t = predict_batch(model, X_current)

        step_time = time.perf_counter() - t0_step

        probs = probs_t.cpu().numpy()
        pred = np.argmax(probs, axis=1)

        all_probs.append(probs)
        all_preds.append(pred)
        all_true.append(y_trial)

        step_metrics.append(
            {
                "trial_idx": trial_idx,
                "buffer_trials": X_buffer.shape[0],
                "correct": int(pred[0] == y_trial[0]),
                "tent_entropy_loss": loss_value,
                "step_time_sec": step_time,
            }
        )

    total_time = time.perf_counter() - t0_total

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_preds)
    probs = np.vstack(all_probs)

    if nb_classes == 2:
        y_prob = probs[:, 1]
        ece = expected_calibration_error(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)
        prob_data_y_true = y_true
        prob_data_y_prob = y_prob

    else:
        y_prob = np.max(probs, axis=1)
        correctness = (y_pred == y_true).astype(int)

        ece = expected_calibration_error(correctness, y_prob)

        y_true_onehot = np.eye(nb_classes)[y_true]
        brier = np.mean(np.sum((probs - y_true_onehot) ** 2, axis=1))

        prob_data_y_true = correctness
        prob_data_y_prob = y_prob

    cm = confusion_matrix(y_true, y_pred, labels=list(range(nb_classes)))

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "brier_score": brier,
        "ece": ece,
        "online_adaptation_time_sec": total_time,
        "online_time_per_trial_ms": 1000.0 * total_time / len(y_true),
        "buffer_size": buffer_size,
        "tent_lr": tent_lr,
    }

    for i in range(nb_classes):
        for j in range(nb_classes):
            metrics[f"cm_{i}{j}"] = int(cm[i, j])

    prob_data = {
        "y_true": prob_data_y_true.tolist(),
        "y_prob": prob_data_y_prob.tolist(),
    }

    return metrics, step_metrics, prob_data


def main():
    parser = argparse.ArgumentParser(description="Online AdaBN Evaluator")

    parser.add_argument("--data", type=str, default="./data/bci2a_dataset")
    parser.add_argument(
        "--protocol",
        type=str,
        required=True,
        choices=["cross-session", "cross-subject"],
    )
    parser.add_argument("--weights_dir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)

    parser.add_argument("--classes", type=int, default=4, choices=[2, 4])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--bn_momentum", type=float, default=0.1)
    parser.add_argument("--ICA", action="store_true")

    parser.add_argument(
        "--buffer_size",
        type=int,
        default=16,
        help="Number of latest trials used for rolling-buffer AdaBN"
    )

    parser.add_argument(
        "--method",
        type=str,
        default="adabn",
        choices=["adabn", "tent"],
        help="Online adaptation method"
    )

    parser.add_argument(
        "--tent_lr",
        type=float,
        default=1e-3,
        help="Learning rate for TENT optimizer"
    )

    args = parser.parse_args()

    internal_protocol = args.protocol
    seed_pool = [42, 97, 123, 1337, 2024]
    run_seeds = seed_pool[: args.seeds] if args.seeds <= 5 else list(range(args.seeds))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[+] Loading data from {args.data} | ICA={args.ICA}")
    raw_dataset = load_bci2a_dataset(args.data, classes=args.classes)
    data_cache, _ = preprocess_bci2a_dataset(
        raw_dataset,
        apply_filter=True,
        apply_ica=args.ICA,
    )

    subjects = [f"A{i:02d}" for i in range(1, 10)]

    all_metrics = []
    all_batch_metrics = []
    all_prob_data = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("ONLINE ADABN EVALUATION")
    print("=" * 60)
    print(f"Protocol: {internal_protocol}")
    print(f"Classes: {args.classes}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"BN momentum: {args.bn_momentum}")
    print("=" * 60)

    for seed in run_seeds:
        for target_sub in subjects:
            weight_path = (
                Path(args.weights_dir)
                / f"eegnet_{internal_protocol}_{target_sub}_seed_{seed}.pt"
            )

            if not weight_path.exists():
                raise FileNotFoundError(f"Missing weights: {weight_path}")

            X_test, y_test = get_target_data(
                data_cache,
                internal_protocol,
                target_sub,
                subjects,
            )

            if args.classes == 2:
                test_mask = (y_test == 0) | (y_test == 1)
                X_test = X_test[test_mask]
                y_test = y_test[test_mask]

            y_test = y_test - np.min(y_test)

            clf = load_model(
                weight_path=weight_path,
                X_sample=X_test,
                nb_classes=args.classes,
            )

            model = clf.model
            assert model is not None

            if args.method == "adabn":
                metrics, batch_metrics, prob_data = evaluate_online_adabn_buffer(
                    model=model,
                    X_test=X_test,
                    y_test=y_test,
                    batch_size=args.batch_size,
                    nb_classes=args.classes,
                    bn_momentum=args.bn_momentum,
                    device=device,
                    buffer_size=args.buffer_size,
                )

            elif args.method == "tent":
                metrics, batch_metrics, prob_data = evaluate_online_tent_buffer(
                    model=model,
                    X_test=X_test,
                    y_test=y_test,
                    nb_classes=args.classes,
                    device=device,
                    tent_lr=args.tent_lr,
                    buffer_size=args.buffer_size,
                )

            metrics.update(
                {
                    "seed": seed,
                    "subject": target_sub,
                    "model": "EEGNET",
                    "adaptation": args.method.upper(),
                    "protocol": internal_protocol,
                    "buffer_size": args.buffer_size,
                    "bn_momentum": args.bn_momentum if args.method == "adabn" else None,
                    "tent_lr": args.tent_lr if args.method == "tent" else None,
                    "weights": str(weight_path),
                }
            )

            all_metrics.append(metrics)

            for bm in batch_metrics:
                bm.update(
                    {
                        "seed": seed,
                        "subject": target_sub,
                        "protocol": internal_protocol,
                        "adaptation": "AdaBN",
                    }
                )
                all_batch_metrics.append(bm)

            all_prob_data[f"{target_sub}_seed_{seed}"] = prob_data

            print(
                f"[{target_sub} | seed {seed}] "
                f"acc={metrics['accuracy']:.4f}, "
                f"bal_acc={metrics['balanced_acc']:.4f}, "
                f"ece={metrics['ece']:.4f}"
            )

    prefix = f"online_{args.method}"

    df = pd.DataFrame(all_metrics)
    df.to_csv(out_dir / f"{prefix}_metrics_per_seed.csv", index=False)

    batch_df = pd.DataFrame(all_batch_metrics)
    batch_df.to_csv(out_dir / f"{prefix}_step_metrics.csv", index=False)

    with open(out_dir / f"{prefix}_prob_data.json", "w") as f:
        json.dump(all_prob_data, f)

    num_cols = df.select_dtypes(include=[np.number]).columns
    summary = df[num_cols].agg(["mean", "std"]).T
    summary["formatted"] = summary.apply(
        lambda row: f"{row['mean']:.4f} ± {row['std']:.4f}",
        axis=1,
    )
    summary.to_csv(out_dir / f"{prefix}_summary.csv")

    print("\n" + "=" * 60)
    print("FINAL ONLINE ADABN REPORT")
    print("=" * 60)
    print(f"Protocol: {internal_protocol}")
    print(f"Total runs: {len(df)}")
    print("-" * 60)
    print(f"Accuracy:      {summary.loc['accuracy', 'formatted']}")
    print(f"Macro-F1:      {summary.loc['macro_f1', 'formatted']}")
    print(f"Balanced Acc:  {summary.loc['balanced_acc', 'formatted']}")
    print(f"ECE:           {summary.loc['ece', 'formatted']}")
    print(f"Brier Score:   {summary.loc['brier_score', 'formatted']}")
    print(f"Online ms/trial: {summary.loc['online_time_per_trial_ms', 'formatted']}")
    print("=" * 60)
    print(f"[+] Output saved to: {out_dir}/")


if __name__ == "__main__":
    main()