import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

from feature_extraction import load_features
import pickle

def make_classifier():
    """
    Classifier used consistently for all feature methods.

    Pipeline:
        StandardScaler
        → shrinkage LDA
    """
    clf = make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage="auto",
        )
    )
    return clf

################################ TASK 3.1 ################################
def within_session_cv_one(
    X,
    y,
    n_splits=5,
    random_state=42,
):
    """
    Run stratified 5-fold CV on one feature matrix.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_trials, n_features)

    y : np.ndarray
        Labels, shape (n_trials,)

    Returns
    -------
    fold_scores : np.ndarray
        Accuracy for each fold.
    """

    assert X.ndim == 2, f"Expected X shape (n_trials, n_features), got {X.shape}"
    assert y.ndim == 1, f"Expected y shape (n_trials,), got {y.shape}"
    assert X.shape[0] == y.shape[0], "X and y must have same number of trials"

    clf = make_classifier()

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    fold_scores = cross_val_score(
        clf,
        X,
        y,
        cv=cv,
        scoring="accuracy",
    )

    return fold_scores


def run_within_session_cv_all(
    features,
    methods=None,
    subjects=None,
    session="T",
    n_splits=5,
    random_state=42,
):
    """
    Run Part 2.1: 5-fold within-session CV for all subjects and methods.

    Parameters
    ----------
    features : dict
        Nested feature dictionary:
            features[subject][method][session]["X"]
            features[subject][method][session]["y"]

    methods : list or None
        Methods to evaluate. If None, inferred from first subject.

    subjects : list or None
        Subjects to evaluate. If None, all subjects in features.

    session : str
        Session to use for within-session CV. For Part 2.1 use "T".

    Returns
    -------
    fold_results_df : pd.DataFrame
        One row per subject/method/fold.

    summary_df : pd.DataFrame
        Mean ± SD per subject and method.

    group_summary_df : pd.DataFrame
        Mean ± SD across subjects per method.
    """

    if subjects is None:
        subjects = sorted(features.keys())

    if methods is None:
        first_subject = subjects[0]
        methods = sorted(features[first_subject].keys())

    fold_rows = []

    for subject in subjects:
        for method in methods:
            print(f"Running method {method} on subject {subject}")

            X = features[subject][method][session]["X"]
            y = features[subject][method][session]["y"]

            scores = within_session_cv_one(
                X,
                y,
                n_splits=n_splits,
                random_state=random_state,
            )

            for fold_idx, acc in enumerate(scores):
                fold_rows.append({
                    "subject": subject,
                    "method": method,
                    "session": session,
                    "fold": fold_idx,
                    "accuracy": acc,
                })

    fold_results_df = pd.DataFrame(fold_rows)

    # Per-subject summary across folds
    summary_df = (
        fold_results_df
        .groupby(["subject", "method"])["accuracy"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={
            "mean": "mean_cv_accuracy",
            "std": "std_cv_accuracy",
        })
    )

    # Group summary across subjects.
    # First average folds within each subject, then compute mean/std across subjects.
    group_summary_df = (
        summary_df
        .groupby("method")["mean_cv_accuracy"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={
            "mean": "group_mean_accuracy",
            "std": "group_std_across_subjects",
        })
    )

    return fold_results_df, summary_df, group_summary_df


def format_mean_std(mean, std):
    return f"{100 * mean:.1f} ± {100 * std:.1f}%"


def make_within_session_report_table(group_summary_df):
    table = group_summary_df.copy()

    table["accuracy_mean_sd"] = [
        format_mean_std(mean, std)
        for mean, std in zip(
            table["group_mean_accuracy"],
            table["group_std_across_subjects"],
        )
    ]

    return table[["method", "accuracy_mean_sd"]]


################################ TASK 3.2 ################################


def cross_session_transfer_one(
    X_train,
    y_train,
    X_test,
    y_test,
):
    """
    Train on session T and test on session E.

    Parameters
    ----------
    X_train : np.ndarray
        Session T feature matrix, shape (n_train_trials, n_features)

    y_train : np.ndarray
        Session T labels, shape (n_train_trials,)

    X_test : np.ndarray
        Session E feature matrix, shape (n_test_trials, n_features)

    y_test : np.ndarray
        Session E labels, shape (n_test_trials,)

    Returns
    -------
    accuracy : float
        Cross-session accuracy.

    y_pred : np.ndarray
        Predicted labels for session E.
    """

    assert X_train.ndim == 2, f"Expected X_train shape (n_trials, n_features), got {X_train.shape}"
    assert X_test.ndim == 2, f"Expected X_test shape (n_trials, n_features), got {X_test.shape}"
    assert y_train.ndim == 1, f"Expected y_train shape (n_trials,), got {y_train.shape}"
    assert y_test.ndim == 1, f"Expected y_test shape (n_trials,), got {y_test.shape}"

    assert X_train.shape[0] == y_train.shape[0], "X_train and y_train must have same number of trials"
    assert X_test.shape[0] == y_test.shape[0], "X_test and y_test must have same number of trials"
    assert X_train.shape[1] == X_test.shape[1], "Train and test feature dimensions must match"

    clf = make_classifier()

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    return accuracy, y_pred


def run_cross_session_transfer_all(
    features,
    methods=None,
    subjects=None,
    train_session="T",
    test_session="E",
):
    """
    Run Part 2.2: train on session T, test on session E.

    Parameters
    ----------
    features : dict
        Nested feature dictionary:
            features[subject][method]["T"]["X"]
            features[subject][method]["T"]["y"]
            features[subject][method]["E"]["X"]
            features[subject][method]["E"]["y"]

    methods : list or None
        Feature methods to evaluate. If None, inferred from first subject.

    subjects : list or None
        Subjects to evaluate. If None, all subjects in features.

    Returns
    -------
    transfer_df : pd.DataFrame
        One row per subject and method.

    predictions : dict
        predictions[subject][method]["y_true"]
        predictions[subject][method]["y_pred"]
    """

    if subjects is None:
        subjects = sorted(features.keys())

    if methods is None:
        first_subject = subjects[0]
        methods = sorted(features[first_subject].keys())

    rows = []
    predictions = {}

    for subject in subjects:
        predictions[subject] = {}

        for method in methods:
            print(f"Running method {method} on subject {subject}")

            X_train = features[subject][method][train_session]["X"]
            y_train = features[subject][method][train_session]["y"]

            X_test = features[subject][method][test_session]["X"]
            y_test = features[subject][method][test_session]["y"]

            accuracy, y_pred = cross_session_transfer_one(
                X_train,
                y_train,
                X_test,
                y_test,
            )

            rows.append({
                "subject": subject,
                "method": method,
                "train_session": train_session,
                "test_session": test_session,
                "cross_session_accuracy": accuracy,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "n_features": X_train.shape[1],
            })

            predictions[subject][method] = {
                "y_true": y_test,
                "y_pred": y_pred,
            }

    transfer_df = pd.DataFrame(rows)

    return transfer_df, predictions



def summarize_cross_session_transfer(transfer_df):
    """
    Summarize cross-session accuracy across subjects for each method.
    """

    group_summary = (
        transfer_df
        .groupby("method")["cross_session_accuracy"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={
            "mean": "group_mean_cross_session_accuracy",
            "std": "group_std_cross_session_accuracy",
        })
    )

    return group_summary


def compute_transfer_gap(summary_df, transfer_df):
    """
    Compute transfer gap:

        transfer gap = within-session CV accuracy - cross-session accuracy

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output from Part 2.1.
        Must contain:
            subject
            method
            mean_cv_accuracy

    transfer_df : pd.DataFrame
        Output from Part 2.2.
        Must contain:
            subject
            method
            cross_session_accuracy

    Returns
    -------
    gap_df : pd.DataFrame
        Per-subject, per-method transfer gap.

    gap_summary_df : pd.DataFrame
        Group mean ± SD transfer gap per method.
    """

    gap_df = summary_df.merge(
        transfer_df,
        on=["subject", "method"],
        how="inner",
    )

    gap_df["transfer_gap"] = (
        gap_df["mean_cv_accuracy"]
        - gap_df["cross_session_accuracy"]
    )

    gap_summary_df = (
        gap_df
        .groupby("method")["transfer_gap"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={
            "mean": "group_mean_transfer_gap",
            "std": "group_std_transfer_gap",
        })
    )

    return gap_df, gap_summary_df


def format_percentage(x):
    return f"{100 * x:.1f}%"


def make_transfer_report_table(gap_df):
    table = gap_df.copy()

    table["within_session_cv"] = table["mean_cv_accuracy"].apply(format_percentage)
    table["cross_session"] = table["cross_session_accuracy"].apply(format_percentage)
    table["transfer_gap_percent"] = table["transfer_gap"].apply(format_percentage)

    return table[
        [
            "subject",
            "method",
            "within_session_cv",
            "cross_session",
            "transfer_gap_percent",
        ]
    ]


################################ TASK 3.3 ################################

def sample_n_per_class(y, n_per_class, random_state=None):
    """
    Randomly sample n trials per class.

    Parameters
    ----------
    y : np.ndarray
        Labels, shape (n_trials,). Expected labels: 0 and 1.

    n_per_class : int
        Number of trials to sample per class.

    random_state : int or None
        Random seed.

    Returns
    -------
    selected_idx : np.ndarray
        Indices of selected trials.
    """

    rng = np.random.default_rng(random_state)

    selected = []

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]

        if n_per_class > len(cls_idx):
            raise ValueError(
                f"Requested {n_per_class} trials for class {cls}, "
                f"but only {len(cls_idx)} are available."
            )

        selected_cls = rng.choice(
            cls_idx,
            size=n_per_class,
            replace=False,
        )

        selected.extend(selected_cls)

    selected = np.asarray(selected)
    rng.shuffle(selected)

    return selected

def data_hunger_one(
    X_train,
    y_train,
    X_test,
    y_test,
    train_sizes=(5, 10, 20, 40, 72, 144),
    n_repeats=20,
    random_state=42,
):
    """
    Data hunger curve for one subject and one feature method.

    Train on subsets of session T and test on all of session E.
    """

    assert X_train.ndim == 2, f"Expected X_train shape (n_trials, n_features), got {X_train.shape}"
    assert X_test.ndim == 2, f"Expected X_test shape (n_trials, n_features), got {X_test.shape}"
    assert y_train.ndim == 1
    assert y_test.ndim == 1
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]

    rows = []

    rng = np.random.default_rng(random_state)

    for n_per_class in train_sizes:
        print(f"N_per_class: {n_per_class}")
        for repeat in range(n_repeats):
            seed = int(rng.integers(0, 1_000_000_000))

            selected_idx = sample_n_per_class(
                y_train,
                n_per_class=n_per_class,
                random_state=seed,
            )

            X_sub = X_train[selected_idx]
            y_sub = y_train[selected_idx]

            clf = make_classifier()
            clf.fit(X_sub, y_sub)

            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            rows.append({
                "n_per_class": n_per_class,
                "repeat": repeat,
                "accuracy": acc,
                "n_train_total": len(y_sub),
            })

    return rows


def run_data_hunger_all(
    features,
    methods=None,
    subjects=None,
    train_session="T",
    test_session="E",
    train_sizes=(5, 10, 20, 40, 72, 144),
    n_repeats=20,
    random_state=42,
    skip_too_large=True,
):
    """
    Run Part 2.3 data hunger curves for all subjects and methods.
    """

    if subjects is None:
        subjects = sorted(features.keys())

    if methods is None:
        first_subject = subjects[0]
        methods = sorted(features[first_subject].keys())

    all_rows = []

    for subject in subjects:
        for method in methods:
            print(f"Running method {method} on subject {subject}")

            X_train = features[subject][method][train_session]["X"]
            y_train = features[subject][method][train_session]["y"]

            X_test = features[subject][method][test_session]["X"]
            y_test = features[subject][method][test_session]["y"]

            class_counts = np.bincount(y_train)
            max_per_class = class_counts.min()

            valid_train_sizes = []
            for n in train_sizes:
                if n <= max_per_class:
                    valid_train_sizes.append(n)
                elif not skip_too_large:
                    raise ValueError(
                        f"{subject} {method}: requested n={n}, "
                        f"but max per class is {max_per_class}."
                    )

            rows = data_hunger_one(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                train_sizes=valid_train_sizes,
                n_repeats=n_repeats,
                random_state=random_state,
            )

            for row in rows:
                row["subject"] = subject
                row["method"] = method
                row["train_session"] = train_session
                row["test_session"] = test_session

            all_rows.extend(rows)

            print(
                f"Finished {subject} {method}: "
                f"train sizes={valid_train_sizes}"
            )

    data_hunger_df = pd.DataFrame(all_rows)

    return data_hunger_df


def summarize_data_hunger(data_hunger_df):
    """
    Create subject-level and group-level summaries.
    """

    subject_summary = (
        data_hunger_df
        .groupby(["subject", "method", "n_per_class"])["accuracy"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={
            "mean": "subject_mean_accuracy",
            "std": "subject_std_repeats",
        })
    )

    group_summary = (
        subject_summary
        .groupby(["method", "n_per_class"])["subject_mean_accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={
            "mean": "group_mean_accuracy",
            "std": "group_std_across_subjects",
            "count": "n_subjects",
        })
    )

    group_summary["group_sem_accuracy"] = (
        group_summary["group_std_across_subjects"]
        / np.sqrt(group_summary["n_subjects"])
    )

    return subject_summary, group_summary

import matplotlib.pyplot as plt


def plot_data_hunger_curves(
    group_summary,
    output_path="part_2_3_data_hunger_curves.png",
):
    """
    Plot average accuracy vs. number of training trials per class.
    One line per method, shaded SEM across subjects.
    """

    plt.figure(figsize=(10, 6))

    methods = sorted(group_summary["method"].unique())

    for method in methods:
        df_m = group_summary[group_summary["method"] == method]
        df_m = df_m.sort_values("n_per_class")

        x = df_m["n_per_class"].to_numpy()
        y = df_m["group_mean_accuracy"].to_numpy()
        sem = df_m["group_sem_accuracy"].to_numpy()

        plt.plot(
            x,
            y,
            marker="o",
            label=method,
        )

        plt.fill_between(
            x,
            y - sem,
            y + sem,
            alpha=0.2,
        )

    plt.axhline(
        0.75,
        linestyle="--",
        linewidth=1,
        label="75% accuracy",
    )

    plt.xlabel("Training trials per class")
    plt.ylabel("Cross-session accuracy")
    plt.title("Data hunger curves: train on session T, test on session E")
    plt.ylim(0.4, 1.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def compute_75_percent_of_max_threshold(group_summary):
    """
    For each method, find the smallest n_per_class where accuracy reaches
    75% of that method's own maximum accuracy.
    """

    rows = []

    for method in sorted(group_summary["method"].unique()):
        df_m = group_summary[group_summary["method"] == method].copy()
        df_m = df_m.sort_values("n_per_class")

        max_acc = df_m["group_mean_accuracy"].max()
        threshold = 0.75 * max_acc

        reached = df_m[df_m["group_mean_accuracy"] >= threshold]

        if len(reached) > 0:
            fastest_n = int(reached.iloc[0]["n_per_class"])
            acc_at_fastest = float(reached.iloc[0]["group_mean_accuracy"])
        else:
            fastest_n = None
            acc_at_fastest = np.nan

        rows.append({
            "method": method,
            "max_accuracy": max_acc,
            "threshold_75_percent_of_max": threshold,
            "fastest_n_per_class": fastest_n,
            "accuracy_at_fastest_n": acc_at_fastest,
        })

    threshold_df = pd.DataFrame(rows)

    return threshold_df


################################ main functions ################################
def main21():
    features = load_features()
    fold_results_df, summary_df, group_summary_df = run_within_session_cv_all(
        features,
        session="T",
        n_splits=5,
        random_state=42,
    )
    print(summary_df)

    report_table = make_within_session_report_table(group_summary_df)
    print(report_table)

    fold_results_df.to_csv("part_2_1_within_session_fold_results.csv", index=False)
    summary_df.to_csv("part_2_1_within_session_per_subject_summary.csv", index=False)
    group_summary_df.to_csv("part_2_1_within_session_group_summary.csv", index=False)


def main22():
    features = load_features()
    transfer_df, predictions = run_cross_session_transfer_all(features)

    print(transfer_df)
    cross_session_summary_df = summarize_cross_session_transfer(transfer_df)

    print(cross_session_summary_df)

    summary_df = pd.read_csv("part_2_1_within_session_per_subject_summary.csv")

    gap_df, gap_summary_df = compute_transfer_gap(
        summary_df,
        transfer_df,
    )

    print(gap_df)
    print(gap_summary_df)

    report_table = make_transfer_report_table(gap_df)
    print(report_table)

    transfer_df.to_csv("part_2_2_cross_session_transfer.csv", index=False)
    cross_session_summary_df.to_csv("part_2_2_cross_session_summary.csv", index=False)

    with open("cross_session_predictions.pkl", "wb") as f:
        pickle.dump(predictions, f)

    gap_df.to_csv("part_2_2_transfer_gap_per_subject.csv", index=False)
    gap_summary_df.to_csv("part_2_2_transfer_gap_summary.csv", index=False)


def main23():
    features = load_features()
    data_hunger_df = run_data_hunger_all(
        features,
        subjects=["A01"],
        train_sizes=(5, 10, 20, 40, 72, 144),
        n_repeats=20,
        random_state=42,
    )

    data_hunger_df.to_csv("part_2_3_data_hunger_results.csv", index=False)

    subject_hunger_summary, group_hunger_summary = summarize_data_hunger(
        data_hunger_df
    )

    subject_hunger_summary.to_csv(
        "part_2_3_data_hunger_subject_summary.csv",
        index=False,
    )

    group_hunger_summary.to_csv(
        "part_2_3_data_hunger_group_summary.csv",
        index=False,
    )

    plot_data_hunger_curves(
        group_hunger_summary,
        output_path="part_2_3_data_hunger_curves.png",
    )

    threshold_df = compute_75_percent_of_max_threshold(group_hunger_summary)

    print(threshold_df)

    threshold_df.to_csv(
        "part_2_3_75_percent_threshold.csv",
        index=False,
    )




if __name__ == "__main__":
    main23()