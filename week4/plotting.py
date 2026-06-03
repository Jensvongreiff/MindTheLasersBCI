import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import pandas as pd
from sklearn.metrics import confusion_matrix
from feature_extraction import load_features

import pickle


def plot_tsne_feature_spaces(
    features,
    subject="A01",
    methods=None,
    train_session="T",
    test_session="E",
    perplexity=30,
    random_state=42,
    output_path="figure_1_tsne_feature_spaces.png",
):
    """
    Plot t-SNE feature spaces for one representative subject.

    One panel per method.
    Color = class label.
    Marker = session.

    Parameters
    ----------
    features : dict
        features[subject][method][session]["X"]
        features[subject][method][session]["y"]

    subject : str
        Representative subject, e.g. "A01".

    methods : list or None
        Feature methods to plot. If None, inferred from features[subject].

    train_session : str
        Usually "T".

    test_session : str
        Usually "E".

    perplexity : float
        t-SNE perplexity. Must be smaller than number of samples.

    random_state : int
        Random seed.

    output_path : str
        Where to save the figure.
    """

    if methods is None:
        methods = sorted(features[subject].keys())

    n_methods = len(methods)

    n_cols = 2
    n_rows = int(np.ceil(n_methods / n_cols))

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, 5 * n_rows),
        squeeze=False,
    )

    class_names = {
        0: "left hand",
        1: "right hand",
    }

    session_markers = {
        train_session: "o",
        test_session: "^",
    }

    for method_idx, method in enumerate(methods):
        ax = axs[method_idx // n_cols, method_idx % n_cols]

        X_T = features[subject][method][train_session]["X"]
        y_T = features[subject][method][train_session]["y"]

        X_E = features[subject][method][test_session]["X"]
        y_E = features[subject][method][test_session]["y"]

        assert X_T.ndim == 2, f"{subject} {method} {train_session}: X must be 2D"
        assert X_E.ndim == 2, f"{subject} {method} {test_session}: X must be 2D"
        assert X_T.shape[1] == X_E.shape[1], (
            f"{subject} {method}: train/test feature dimensions differ: "
            f"{X_T.shape[1]} vs {X_E.shape[1]}"
        )

        X_all = np.vstack([X_T, X_E])
        y_all = np.concatenate([y_T, y_E])
        session_all = np.array(
            [train_session] * len(y_T) + [test_session] * len(y_E)
        )

        # Scale before t-SNE, fit scaler on combined data for visualization only.
        X_all_scaled = StandardScaler().fit_transform(X_all)

        # Perplexity must be less than n_samples.
        effective_perplexity = min(perplexity, max(5, (len(X_all_scaled) - 1) // 3))

        tsne = TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        )

        Z = tsne.fit_transform(X_all_scaled)

        for session in [train_session, test_session]:
            for cls in sorted(np.unique(y_all)):
                mask = (session_all == session) & (y_all == cls)

                ax.scatter(
                    Z[mask, 0],
                    Z[mask, 1],
                    marker=session_markers[session],
                    alpha=0.75,
                    s=35,
                    label=f"{session}, {class_names.get(cls, cls)}",
                )

        ax.set_title(f"{subject} — {method}")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(True)

    # Hide unused axes
    for j in range(n_methods, n_rows * n_cols):
        axs[j // n_cols, j % n_cols].axis("off")

    # Put one shared legend outside
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.suptitle(
        f"Figure 1: t-SNE feature spaces for representative subject {subject}",
        y=1.06,
        fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()





def plot_cross_session_grouped_bar(
    csv_path="part_2_2_cross_session_transfer.csv",
    output_path="figure_2_cross_session_grouped_bar.png",
):
    """
    Figure 2:
    Per-subject cross-session accuracy for all methods as a grouped bar chart.
    """

    df = pd.read_csv(csv_path)

    required_cols = {"subject", "method", "cross_session_accuracy"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    subjects = sorted(df["subject"].unique())
    methods = sorted(df["method"].unique())

    pivot = df.pivot(
        index="subject",
        columns="method",
        values="cross_session_accuracy",
    )

    pivot = pivot.loc[subjects, methods]

    x = np.arange(len(subjects))
    width = 0.8 / len(methods)

    plt.figure(figsize=(12, 6))

    for i, method in enumerate(methods):
        offset = (i - (len(methods) - 1) / 2) * width

        plt.bar(
            x + offset,
            pivot[method].values * 100,
            width=width,
            label=method,
        )

    plt.axhline(
        50,
        linestyle="--",
        linewidth=1,
        label="chance level",
    )

    plt.xticks(x, subjects)
    plt.xlabel("Subject")
    plt.ylabel("Cross-session accuracy [%]")
    plt.title("Figure 2: Per-subject cross-session accuracy")
    plt.ylim(0, 100)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return pivot


def plot_data_hunger_curves_from_csv(
    csv_path="part_2_3_data_hunger_results.csv",
    output_path="figure_3_data_hunger_curves.png",
):
    """
    Figure 3:
    Data hunger curves, one line per method, averaged across subjects.
    Shaded area = standard error across subjects.
    """

    df = pd.read_csv(csv_path)

    required_cols = {"subject", "method", "n_train_total", "accuracy"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # First average across repeats within each subject/method/training size
    subject_summary = (
        df
        .groupby(["subject", "method", "n_train_total"])["accuracy"]
        .mean()
        .reset_index()
        .rename(columns={"accuracy": "subject_mean_accuracy"})
    )

    # Then average across subjects
    group_summary = (
        subject_summary
        .groupby(["method", "n_train_total"])["subject_mean_accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={
            "mean": "group_mean_accuracy",
            "std": "group_std_accuracy",
            "count": "n_subjects",
        })
    )

    group_summary["sem"] = (
        group_summary["group_std_accuracy"] / np.sqrt(group_summary["n_subjects"])
    )

    plt.figure(figsize=(10, 6))

    methods = sorted(group_summary["method"].unique())

    for method in methods:
        df_m = group_summary[group_summary["method"] == method]
        df_m = df_m.sort_values("n_train_total")

        x = df_m["n_train_total"].to_numpy()
        y = df_m["group_mean_accuracy"].to_numpy() * 100
        sem = df_m["sem"].to_numpy() * 100

        plt.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            label=method,
        )

        plt.fill_between(
            x,
            y - sem,
            y + sem,
            alpha=0.2,
        )

    plt.axhline(
        50,
        linestyle="--",
        linewidth=1,
        label="chance level",
    )

    plt.xlabel("Number of training trials from session 1")
    plt.ylabel("Cross-session accuracy [%]")
    plt.title("Figure 3: Data hunger curves")
    plt.ylim(40, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return group_summary


def plot_average_confusion_matrices(
    predictions,
    methods=None,
    subjects=None,
    output_path="figure_4_average_confusion_matrices.png",
    normalize=True,
):
    """
    Figure 4:
    Confusion matrices for cross-session transfer, one per method,
    averaged across subjects.

    Parameters
    ----------
    predictions : dict
        predictions[subject][method]["y_true"]
        predictions[subject][method]["y_pred"]

    methods : list or None
        Methods to plot. If None, inferred from first subject.

    subjects : list or None
        Subjects to include. If None, all subjects.

    output_path : str
        Path to save figure.

    normalize : bool
        If True, each subject confusion matrix is row-normalized before averaging.
        This gives per-class accuracy proportions.
    """

    if subjects is None:
        subjects = sorted(predictions.keys())

    if methods is None:
        first_subject = subjects[0]
        methods = sorted(predictions[first_subject].keys())

    class_labels = [0, 1]
    class_names = ["Left hand", "Right hand"]

    n_methods = len(methods)
    n_cols = 2
    n_rows = int(np.ceil(n_methods / n_cols))

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4.5 * n_rows),
        squeeze=False,
    )

    for method_idx, method in enumerate(methods):
        ax = axs[method_idx // n_cols, method_idx % n_cols]

        cms = []

        for subject in subjects:
            y_true = predictions[subject][method]["y_true"]
            y_pred = predictions[subject][method]["y_pred"]

            cm = confusion_matrix(
                y_true,
                y_pred,
                labels=class_labels,
            ).astype(float)

            if normalize:
                row_sums = cm.sum(axis=1, keepdims=True)
                cm = np.divide(
                    cm,
                    row_sums,
                    out=np.zeros_like(cm),
                    where=row_sums != 0,
                )

            cms.append(cm)

        avg_cm = np.mean(cms, axis=0)

        im = ax.imshow(avg_cm, vmin=0, vmax=1 if normalize else None)

        ax.set_title(method)
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

        for i in range(avg_cm.shape[0]):
            for j in range(avg_cm.shape[1]):
                if normalize:
                    text = f"{avg_cm[i, j] * 100:.1f}%"
                else:
                    text = f"{avg_cm[i, j]:.1f}"

                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color="white" if avg_cm[i, j] > 0.5 else "black",
                )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused axes
    for j in range(n_methods, n_rows * n_cols):
        axs[j // n_cols, j % n_cols].axis("off")

    title = "Figure 4: Average cross-session confusion matrices"
    if normalize:
        title += " row-normalized"

    fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()



if __name__ == "__main__":
    features = load_features()

    # plot_tsne_feature_spaces(
    #     features,
    #     subject="A01",
    #     methods=["bandpower", "csp", "morlet", "riemannian_mdm"],
    #     output_path="figure_1_tsne_A01.png",
    # )

    # pivot = plot_cross_session_grouped_bar(
    #     csv_path="/home/dani/Documents/TUM/3.Semester/BCI/practical-ss26-team4/week4/features/part_2_2_cross_session_transfer.csv",
    #     output_path="figure_2_cross_session_grouped_bar.png",
    # )

    # print(pivot)


    # group_hunger_summary = plot_data_hunger_curves_from_csv(
    #     csv_path="/home/dani/Documents/TUM/3.Semester/BCI/practical-ss26-team4/week4/features/part_2_3_data_hunger_results.csv",
    #     output_path="figure_3_data_hunger_curves.png",
    # )

    # print(group_hunger_summary)


    with open("/home/dani/Documents/TUM/3.Semester/BCI/practical-ss26-team4/week4/features/cross_session_predictions.pkl", "rb") as f:
        predictions = pickle.load(f)

    plot_average_confusion_matrices(
        predictions,
        methods=["bandpower", "csp", "morlet", "riemannian_mdm"],
        output_path="figure_4_average_confusion_matrices.png",
        normalize=True,
    )

