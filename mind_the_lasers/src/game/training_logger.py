import csv
from datetime import datetime
from pathlib import Path


class TrainingLogger:
    FIELDNAMES = [
        "timestamp",
        "trial_number",
        "ground_truth_command",
        "predicted_command",
        "confidence",
        "accepted_command",
        "correct",
        "decision_time",
        "game_event",
    ]

    def __init__(self, output_dir=None):
        if output_dir is None:
            output_path = Path(__file__).resolve().parents[1] / "logs" / "training_logs"
        else:
            output_path = Path(output_dir)

        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.filepath = output_path / f"training_session_{timestamp}.csv"

        self.file = open(
            self.filepath,
            "w",
            newline="",
            encoding="utf-8",
        )

        self.writer = csv.DictWriter(
            self.file,
            fieldnames=self.FIELDNAMES,
        )

        self.writer.writeheader()
        self.file.flush()

        print(f"Training log: {self.filepath}")

    def log_decision(
        self,
        trial_number,
        ground_truth_command,
        predicted_command,
        confidence,
        accepted_command,
        correct,
        decision_time,
        game_event,
    ):
        self.writer.writerow(
            {
                "timestamp": datetime.now().isoformat(),
                "trial_number": trial_number,
                "ground_truth_command": ground_truth_command,
                "predicted_command": predicted_command,
                "confidence": confidence,
                "accepted_command": accepted_command,
                "correct": correct,
                "decision_time": round(decision_time, 4),
                "game_event": game_event,
            }
        )

        self.file.flush()

    def close(self):
        if not self.file.closed:
            self.file.close()