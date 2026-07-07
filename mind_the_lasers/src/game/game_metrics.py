import csv
import time
from datetime import datetime
from pathlib import Path


class GameMetrics:
    FIELDNAMES = [
        "timestamp",
        "highest_level_reached",
        "levels_completed",
        "total_levels",
        "total_play_time",
        "average_level_time",
        "laser_encounters",
        "successful_avoidances",
        "avoidance_rate",
        "collisions",
        "accepted_commands",
        "command_transitions",
        "time_moving_toward_goal",
        "time_moving_away_from_goal",
        "time_resting",
        "effective_movement_ratio",
        "boosts_used",
        "boosted_collisions",
    ]

    def __init__(self, total_levels, output_dir="game_logs"):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.filepath = output_path / f"game_session_{timestamp}.csv"

        self.total_levels = total_levels

        self.session_start_time = time.perf_counter()
        self.level_start_time = time.perf_counter()

        self.highest_level_reached = 1
        self.levels_completed = 0

        self.completed_level_times = []

        self.laser_encounters = 0
        self.successful_avoidances = 0
        self.collisions = 0

        self.accepted_commands = 0
        self.command_transitions = 0

        self.time_moving_toward_goal = 0.0
        self.time_moving_away_from_goal = 0.0
        self.time_resting = 0.0

        self.boosts_used = 0
        self.boosted_collisions = 0

        self.last_command = None

        # Used to ensure one encounter/avoidance is counted
        # only once per laser per level.
        self.encountered_lasers = set()
        self.collided_lasers = set()
        self.avoided_lasers = set()

        self.saved = False

        self.final_play_time = None

        print(f"Game log: {self.filepath}")

    def record_command(self, command):
        self.accepted_commands += 1

        if self.last_command is not None and command != self.last_command:
            self.command_transitions += 1

        self.last_command = command

    def record_movement_time(self, command, goal_direction, dt):
        if command.value == 0:
            self.time_resting += dt

        elif command.value == goal_direction:
            self.time_moving_toward_goal += dt

        else:
            self.time_moving_away_from_goal += dt

    def record_laser_encounter(self, laser):
        laser_id = id(laser)

        if laser_id in self.encountered_lasers:
            return

        self.encountered_lasers.add(laser_id)
        self.laser_encounters += 1

    def record_collision(self, laser, boosted=False):
        laser_id = id(laser)

        if laser_id in self.collided_lasers:
            return

        self.collided_lasers.add(laser_id)
        self.collisions += 1

        if boosted:
            self.boosted_collisions += 1

    def record_avoidance(self, laser):
        laser_id = id(laser)

        if laser_id not in self.encountered_lasers:
            return

        if laser_id in self.collided_lasers:
            return

        if laser_id in self.avoided_lasers:
            return

        self.avoided_lasers.add(laser_id)
        self.successful_avoidances += 1

    def record_boost_used(self):
        self.boosts_used += 1

    def record_level_completed(self, next_level_number):
        level_time = time.perf_counter() - self.level_start_time

        self.completed_level_times.append(level_time)
        self.levels_completed += 1

        self.highest_level_reached = max(
            self.highest_level_reached,
            next_level_number,
        )

        self.level_start_time = time.perf_counter()

        self.reset_level_tracking()

    def reset_level_tracking(self):
        self.encountered_lasers.clear()
        self.collided_lasers.clear()
        self.avoided_lasers.clear()

    def total_play_time(self):
        if self.final_play_time is not None:
            return self.final_play_time

        return time.perf_counter() - self.session_start_time
    
    def stop_timer(self):
        if self.final_play_time is None:
            self.final_play_time = (
                time.perf_counter() - self.session_start_time
            )

    def average_level_time(self):
        if not self.completed_level_times:
            return 0.0

        return (
            sum(self.completed_level_times)
            / len(self.completed_level_times)
        )

    def avoidance_rate(self):
        if self.laser_encounters == 0:
            return 0.0

        return self.successful_avoidances / self.laser_encounters

    def effective_movement_ratio(self):
        total_movement = (
            self.time_moving_toward_goal
            + self.time_moving_away_from_goal
        )

        if total_movement == 0:
            return 0.0

        return self.time_moving_toward_goal / total_movement

    def get_summary(self):
        return {
            "highest_level_reached": self.highest_level_reached,
            "levels_completed": self.levels_completed,
            "total_levels": self.total_levels,
            "total_play_time": self.total_play_time(),
            "average_level_time": self.average_level_time(),
            "laser_encounters": self.laser_encounters,
            "successful_avoidances": self.successful_avoidances,
            "avoidance_rate": self.avoidance_rate(),
            "collisions": self.collisions,
            "accepted_commands": self.accepted_commands,
            "command_transitions": self.command_transitions,
            "time_moving_toward_goal": self.time_moving_toward_goal,
            "time_moving_away_from_goal": self.time_moving_away_from_goal,
            "time_resting": self.time_resting,
            "effective_movement_ratio": self.effective_movement_ratio(),
            "boosts_used": self.boosts_used,
            "boosted_collisions": self.boosted_collisions,
        }

    def save(self):
        if self.saved:
            return

        summary = self.get_summary()

        with open(
            self.filepath,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=self.FIELDNAMES,
            )

            writer.writeheader()

            writer.writerow(
                {
                    "timestamp": datetime.now().isoformat(),
                    **{
                        key: (
                            round(value, 4)
                            if isinstance(value, float)
                            else value
                        )
                        for key, value in summary.items()
                    },
                }
            )

        self.saved = True

        print(f"Saved game metrics: {self.filepath}")