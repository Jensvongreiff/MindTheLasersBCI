import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def find_latest_log():
    log_dir = (
        Path(__file__).resolve().parents[1]
        / "logs"
        / "game_logs"
    )

    csvs = sorted(
        log_dir.glob("*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not csvs:
        raise FileNotFoundError(
            f"No game logs found in {log_dir}"
        )

    return csvs[0]


def load_data(path):
    return pd.read_csv(path)


def plot_dashboard(df):

    row = df.iloc[0]

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(
        "Mind the Lasers - Play Summary",
        fontsize=18,
        fontweight="bold",
    )

    # ----------------------------------------------------------
    # Overall Score
    # ----------------------------------------------------------

    ax1 = plt.subplot2grid((2, 3), (0, 0))
    ax1.axis("off")

    score = row["overall_score"]

    ax1.text(
        0.5,
        0.65,
        f"{score:.1f}",
        fontsize=40,
        ha="center",
        va="center",
        fontweight="bold",
    )

    ax1.text(
        0.5,
        0.28,
        "Overall Score",
        fontsize=16,
        ha="center",
    )

    # ----------------------------------------------------------
    # Progress bars
    # ----------------------------------------------------------

    ax2 = plt.subplot2grid((2, 3), (0, 1))

    metrics = [
        "Level Progress",
        "Avoidance",
        "Movement",
    ]

    values = [
        row["levels_completed"] / row["total_levels"] * 100,
        row["avoidance_rate"] * 100,
        row["effective_movement_ratio"] * 100,
    ]

    ax2.barh(metrics, values)

    ax2.set_xlim(0, 100)
    ax2.set_xlabel("%")
    ax2.set_title("Performance")

    for i, v in enumerate(values):
        ax2.text(
            v + 2,
            i,
            f"{v:.0f}%",
            va="center",
        )

    # ----------------------------------------------------------
    # Movement pie chart
    # ----------------------------------------------------------

    ax3 = plt.subplot2grid((2, 3), (0, 2))

    movement = [
        row["time_moving_toward_goal"],
        row["time_moving_away_from_goal"],
        row["time_resting"],
    ]

    labels = [
        "Toward Goal",
        "Away",
        "Rest",
    ]

    ax3.pie(
        movement,
        labels=labels,
        autopct="%1.0f%%",
        startangle=90,
    )

    ax3.set_title("Movement Distribution")

    # ----------------------------------------------------------
    # Game statistics
    # ----------------------------------------------------------

    ax4 = plt.subplot2grid((2, 3), (1, 0))

    stats_names = [
        "Levels",
        "Collisions",
        "Boosts",
    ]

    stats_values = [
        row["levels_completed"],
        row["collisions"],
        row["boosts_used"],
    ]

    ax4.bar(
        stats_names,
        stats_values,
    )

    ax4.set_title("Game Statistics")

    # ----------------------------------------------------------
    # Commands
    # ----------------------------------------------------------

    ax5 = plt.subplot2grid((2, 3), (1, 1))

    cmd = [
        row["accepted_commands"],
        row["command_transitions"],
    ]

    ax5.bar(
        [
            "Accepted",
            "Transitions",
        ],
        cmd,
    )

    ax5.set_title("Command Usage")

    # ----------------------------------------------------------
    # Summary text
    # ----------------------------------------------------------

    ax6 = plt.subplot2grid((2, 3), (1, 2))
    ax6.axis("off")

    summary = (
        f"Highest level: {int(row['highest_level_reached'])}\n\n"
        f"Play time: {row['total_play_time']:.1f} s\n\n"
        f"Average level: {row['average_level_time']:.1f} s\n\n"
        f"Laser encounters: {int(row['laser_encounters'])}\n\n"
        f"Successful avoidances: {int(row['successful_avoidances'])}\n\n"
        f"Effective movement: {row['effective_movement_ratio']*100:.1f}%"
    )

    ax6.text(
        0.0,
        1.0,
        summary,
        fontsize=12,
        va="top",
    )

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

    print(f"Loading {path}")

    df = load_data(path)

    plot_dashboard(df)


if __name__ == "__main__":
    main()