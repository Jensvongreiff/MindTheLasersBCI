import csv
from datetime import datetime
from pathlib import Path


class ErrPLogger:
    # Each CSV row represents ONE finalized trial decision, not one EEG sample.
    # Comments document the CSV schema but are NOT written into the CSV itself.
    # The first nine fields retain their original order for compatibility.
    FIELDNAMES = [
        "timestamp",          # Local wall-clock time when the decision is logged (ISO 8601); NOT an EEG-synchronized timestamp.
        "trial_type",         # Scheduled trial condition: "correct" (no flip) or "error" (LEFT/RIGHT swapped).
        "scenario",           # Target side for this trial: "left" or "right".
        "trial_number",       # 1-based number of the trial within the session (including calibration trials).
        "expected_command",   # Target-directed command: LEFT or RIGHT; the reference for correctness.
        "received_command",   # Command actually applied to the player, AFTER any injected LEFT/RIGHT flip.
        "confidence",         # Controller's confidence at logging time; keyboard and observation can report 1.0.
        "command_correct",    # True if received_command matches expected_command, False otherwise.
        "decision_time",      # Seconds from trial start until the decision is finalized (not necessarily first keypress).
        "input_command",      # Original controller command BEFORE error injection; blank for observation mode.
        "input_correct",      # True if input_command matches expected_command; False if not; blank without human input.
    ]

    def __init__(self, output_dir=None):
        if output_dir is None:
            output_path = Path(__file__).resolve().parents[1] / "logs" / "errp_logs"
        else:
            output_path = Path(output_dir)

        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = output_path / f"errp_session_{timestamp}.csv"
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

    def log_decision(
        self,
        trial_number,
        trial_type,
        scenario,
        expected_command,
        received_command,
        confidence,
        command_correct,
        decision_time,
        input_command=None,
    ):
        # input_command is None in observation mode: no participant/controller
        # response occurred, so neither original input nor its correctness
        # should be fabricated. The CSV writer writes None as an empty cell.
        input_correct = (
            input_command == expected_command
            if input_command is not None
            else None
        )

        self.writer.writerow(
            {
                "timestamp": datetime.now().isoformat(),
                "trial_type": trial_type,
                "scenario": scenario,
                "trial_number": trial_number,
                "expected_command": expected_command,
                "received_command": received_command,
                "confidence": confidence,
                "command_correct": command_correct,
                "decision_time": round(decision_time, 4),
                "input_command": input_command,
                "input_correct": input_correct,
            }
        )
        self.file.flush()

    def close(self):
        if not self.file.closed:
            self.file.close()
