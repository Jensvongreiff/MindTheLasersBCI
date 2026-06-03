import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_data_hunger_curves_with_method_thresholds(
    group_summary,
    output_path="part_2_3_data_hunger_curves_thresholds.png",
):
    """
    Plot data hunger curves and show where each method reaches
    75% of its own maximum accuracy.
    """

    plt.figure(figsize=(10, 6))

    method_names = {
        "bandpower": "Bandpower",
        "csp": "CSP",
        "morlet": "Morlet",
        "riemannian_mdm": "Riemannian MDM",
    }

    threshold_rows = []

    methods = sorted(group_summary["method"].unique())

    for method in methods:
        df_m = group_summary[group_summary["method"] == method].copy()
        df_m = df_m.sort_values("n_per_class")

        x = df_m["n_per_class"].to_numpy()
        y = df_m["group_mean_accuracy"].to_numpy()

        max_acc = y.max()
        threshold = 0.75 * max_acc

        reached = df_m[df_m["group_mean_accuracy"] >= threshold]

        if len(reached) > 0:
            fastest_n = int(reached.iloc[0]["n_per_class"])
            acc_at_fastest = float(reached.iloc[0]["group_mean_accuracy"])
        else:
            fastest_n = np.nan
            acc_at_fastest = np.nan

        label = method_names.get(method, method)

        plt.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            label=f"{label}",
        )

        # Horizontal method-specific 75%-of-max line
        plt.axhline(
            threshold,
            linestyle="--",
            linewidth=1,
            alpha=0.5,
        )

        # Vertical line at fastest n
        if not np.isnan(fastest_n):
            plt.axvline(
                fastest_n,
                linestyle=":",
                linewidth=1,
                alpha=0.6,
            )

            plt.scatter(
                fastest_n,
                acc_at_fastest,
                s=80,
                zorder=5,
            )

            plt.text(
                fastest_n,
                acc_at_fastest + 0.01,
                f"{label}: n={fastest_n}",
                fontsize=9,
                ha="center",
            )

        threshold_rows.append({
            "method": method,
            "max_accuracy": max_acc,
            "threshold_75_percent_of_max": threshold,
            "fastest_n_per_class": fastest_n,
            "accuracy_at_fastest_n": acc_at_fastest,
        })

    plt.xlabel("Training trials per class")
    plt.ylabel("Cross-session accuracy")
    plt.title("Data hunger curves: 75% of each method's own maximum accuracy")
    plt.ylim(0.35, 0.75)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    threshold_df = pd.DataFrame(threshold_rows)

    return threshold_df


group_summary = pd.read_csv("/home/dani/Documents/TUM/3.Semester/BCI/practical-ss26-team4/week4/features/part_2_3_data_hunger_group_summary.csv")

threshold_df = plot_data_hunger_curves_with_method_thresholds(
    group_summary,
    output_path="part_2_3_data_hunger_curves_method_thresholds.png",
)

print(threshold_df)

threshold_df.to_csv(
    "part_2_3_75_percent_of_own_max_threshold.csv",
    index=False,
)