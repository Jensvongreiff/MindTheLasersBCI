import random

import pygame

from mind_the_lasers.src.game.input_controller import Command
from mind_the_lasers.src.game.player import Player
from mind_the_lasers.src.game.settings import *
from mind_the_lasers.src.game.errp_logger import ErrPLogger
from mind_the_lasers.src.game.errp_trial import ErrPTrial

class ErrPMode:
    def __init__(
        self,
        screen,
        controller,
        marker_sender,
        sub_mode,
        trials_per_command=ERRP_TRIALS_PER_COMMAND,
    ):
        self.screen = screen
        self.controller = controller
        self.marker_sender = marker_sender

        # sub_mode can be "observation", "interactive", or "interactive_continuous"
        self.sub_mode = sub_mode

        self.player = Player()

        self.logger = ErrPLogger()

        self.font = pygame.font.Font(None, 48)
        self.small_font = pygame.font.Font(None, 30)

        self.trials = self._create_trials(trials_per_command)

        self.current_trial_index = 0
        self.current_trial = None

        self.session_complete = False
        self.waiting_for_space = False
        self.inter_trial_elapsed = 0.0

        self.feedback_timer = 0.0
        self.correct_delay_timer = 0.0
        self.incorrect_movement_timer = 0.0
        self.incorrect_event_active = False
        self.trial_input_marker_sent = False

        self.TRIAL_FEEDBACK_TIME = 0.5

        self.command_received = False

        self.command_delay_timer = 0.0
        self.command_delay_finished = False


        self.load_trial()

    def _create_trials(self, trials_per_command):
        trials = []

        def create_trial(trial_number, trial_type):
            scenario = (
                "left"
                if random.random() < LR_PROBABILITY
                else "right"
            )

            return ErrPTrial(
                trial_number=trial_number,
                trial_type=trial_type,
                scenario=scenario,
            )

        for trial_number in range(1, CALIBRATION_TRIALS + 1):
            trials.append(create_trial(trial_number, "correct"))

        first_randomized_trial = CALIBRATION_TRIALS + 1
        last_randomized_trial = CALIBRATION_TRIALS + trials_per_command * 2

        for trial_number in range(
            first_randomized_trial,
            last_randomized_trial + 1,
        ):
            trial_type = (
                "error"
                if random.random() < ERROR_PROBABILITY
                else "correct"
            )
            trials.append(create_trial(trial_number, trial_type))

        return trials

    def load_trial(self):
        if self.current_trial_index >= len(self.trials):
            self.session_complete = True
            self.current_trial = None
            return

        self.current_trial = self.trials[self.current_trial_index]

        self.player.reset_position(
            self.current_trial.start_x
        )

        self.player.direction = 0

        self.feedback_timer = 0.0
        self.correct_delay_timer = 0.0
        self.incorrect_movement_timer = 0.0
        self.incorrect_event_active = False
        self.trial_input_marker_sent = False

        self.command_received = False
        self.controller.reset()

        self.command_delay_timer = 0.0
        self.command_delay_finished = False

        self.current_trial.start()

        self.marker_sender.send(
            f"trial_start:{self.current_trial.trial_type}:"
            f"{self.current_trial.scenario}"
        )

    def handle_event(self, event):
        if (
            self.waiting_for_space
            and event.type == pygame.KEYDOWN
            and event.key == pygame.K_SPACE
            and not getattr(event, "repeat", False)
            and self.inter_trial_elapsed >= ERRP_INTER_TRIAL_MIN_TIME
        ):
            self.waiting_for_space = False
            self.load_trial()

    def update(self, dt):
        if self.session_complete:
            return

        if self.waiting_for_space:
            self.inter_trial_elapsed += dt
            return

        self.controller.update()

        trial = self.current_trial

        if not trial.decision_made:
            if self.sub_mode == "interactive_continuous":
                self._update_interactive_continuous(dt, trial)

            elif self.sub_mode == "observation" and not self.command_delay_finished:
                self.command_delay_timer += dt

                # Drain commands received during the preparation period.
                self.controller.get_new_prediction()

                if self.command_delay_timer >= ERRP_COMMAND_DELAY:
                    self.command_delay_finished = True

                    # Clear anything still buffered before opening
                    # the decision window.
                    self.controller.reset()

            elif self.sub_mode == "observation":
                command = self._apply_trial_error(
                    trial.ground_truth,
                    trial,
                )
                self._handle_decision(
                    command,
                    confidence=1.0,
                )

            elif not self.command_received:
                command = self.controller.get_new_prediction()

                if command in (Command.LEFT, Command.RIGHT):
                    self.command_received = True

                    confidence = self.controller.confidence
                    input_command = command  # Save raw input before the flip.
                    command = self._apply_trial_error(command, trial)

                    self._handle_decision(
                        command,
                        confidence,
                        input_command=input_command,
                    )

        # Decision has already been made.
        else:
            self.controller.get_new_prediction()
            self._update_after_decision(dt)

    def _update_interactive_continuous(self, dt, trial):
        self.controller.get_new_prediction()
        input_command = self.controller.get_command()

        if input_command in (Command.NONE, Command.REST):
            self.player.direction = 0
            return

        command = self._apply_trial_error(input_command, trial)
        self._send_trial_input_marker()
        self._move_player(command, dt)

        confidence = getattr(self.controller, "confidence", 1.0)

        if self._crossed_incorrect_threshold(trial):
            self._handle_decision(command, confidence, input_command=input_command)
            return

        if trial.ground_truth == Command.RIGHT:
            reached_target = self.player.x >= trial.goal_x
        else:
            reached_target = self.player.x <= trial.goal_x

        if reached_target:
            self.player.x = trial.goal_x
            self._handle_decision(command, confidence, input_command=input_command)

            self.correct_delay_timer = ERRP_CORRECT_TRIAL_DELAY
            if self.correct_delay_timer <= 0:
                self._finish_trial()

    def _crossed_incorrect_threshold(self, trial):
        fraction = INCORRECT_DISTANCE_THRESHOLD_FRACTION
        if fraction <= 0:
            raise ValueError(
                "INCORRECT_DISTANCE_THRESHOLD_FRACTION must be greater than zero"
            )

        wrong_direction = -trial.ground_truth.value
        barrier_x = WIDTH / 2 + wrong_direction * (WIDTH / 2) / fraction

        if wrong_direction < 0:
            return self.player.x <= barrier_x

        return self.player.x >= barrier_x

    @staticmethod
    def _apply_trial_error(command, trial):
        if trial.trial_type != "error":
            return command

        if command == Command.LEFT:
            return Command.RIGHT

        if command == Command.RIGHT:
            return Command.LEFT

        return command

    def _handle_decision(self, command, confidence, input_command=None):
        trial = self.current_trial

        decision_time = trial.decision_time()

        trial.accept_command(command)
        self._send_trial_input_marker()

        self.logger.log_decision(
            trial_number=trial.trial_number,
            trial_type=trial.trial_type,
            scenario=trial.scenario,
            expected_command=trial.ground_truth.name,
            received_command=command.name,
            confidence=confidence,
            command_correct=trial.correct,
            decision_time=decision_time,
            # Observation mode passes None; continuous mode passes the raw
            # direction at the moment the decision becomes final.
            input_command=input_command.name if input_command is not None else None,
        )

        # Let incorrect movement play before showing error feedback.
        if not trial.correct:
            movement_time = (
                ERRP_INCORRECT_FEEDBACK_TIME_CONTINUOUS
                if self.sub_mode == "interactive_continuous"
                else ERRP_INCORRECT_FEEDBACK_TIME
            )
            self.incorrect_movement_timer = movement_time
            self.incorrect_event_active = False

    def _send_trial_input_marker(self):
        if self.trial_input_marker_sent:
            return

        self.marker_sender.send(
            f"trial_input:{self.current_trial.trial_type}:"
            f"{self.current_trial.scenario}"
        )
        self.trial_input_marker_sent = True

    def _update_after_decision(self, dt):
        trial = self.current_trial

        # Incorrect decision:
        # play the movement, show feedback, then advance.
        if not trial.correct:
            if not self.incorrect_event_active:
                movement_dt = min(dt, self.incorrect_movement_timer)
                self._move_player(trial.accepted_command, movement_dt)
                self.incorrect_movement_timer -= movement_dt

                if self.incorrect_movement_timer <= 0:
                    self.incorrect_event_active = True
                    self.feedback_timer = self.TRIAL_FEEDBACK_TIME

                return

            self.feedback_timer -= dt

            if self.feedback_timer <= 0:
                self._finish_trial()

            return

        if self.correct_delay_timer > 0:
            self.correct_delay_timer -= dt

            if self.correct_delay_timer <= 0:
                self._finish_trial()

            return

        self._move_player(trial.accepted_command, dt)

        if trial.ground_truth == Command.RIGHT:
            reached_target = self.player.x >= trial.goal_x

        else:
            reached_target = self.player.x <= trial.goal_x

        if reached_target:
            self.player.x = trial.goal_x
            self.correct_delay_timer = ERRP_CORRECT_TRIAL_DELAY

            if self.correct_delay_timer <= 0:
                self._finish_trial()

    def _move_player(self, command, dt):
        self.player.direction = command.value
        self.player.x += self.player.direction * BALL_SPEED_TRAINING * dt
        self.player.x = max(
            BALL_RADIUS,
            min(WIDTH - BALL_RADIUS, self.player.x),
        )

    def draw(self):
        self.screen.fill(BG_COLOR)

        if self.session_complete:
            self._draw_session_complete()
            return

        if self.waiting_for_space:
            self._draw_inter_trial()
            return

        trial = self.current_trial

        self._draw_header(trial)
        self._draw_target(trial)
        self.player.draw(self.screen)

        if DEBUG_ERRP and trial.decision_made and (
            trial.correct or self.incorrect_event_active
        ):
            self._draw_decision_feedback(trial)

    def _draw_decision_feedback(self, trial):
        if trial.correct:
            text = "COMMAND CORRECT"
            color = (80, 220, 80)
        else:
            text = "COMMAND INCORRECT"
            color = (255, 80, 80)

        feedback = self.font.render(
            text,
            True,
            color,
        )

        rect = feedback.get_rect(
            center=(WIDTH // 2, HEIGHT - 100)
        )

        self.screen.blit(feedback, rect)



    def _draw_header(self, trial):
        title = self.font.render(
            "ERRP MODE",
            True,
            (255, 255, 255),
        )

        title_rect = title.get_rect(
            center=(WIDTH // 2, 40)
        )

        self.screen.blit(title, title_rect)

        scenario_text = ""
        if DEBUG_ERRP:
            scenario_text = f"SCENARIO: {trial.scenario.upper()}"
            scenario_text += f" | TYPE: {trial.trial_type.upper()}"
            if trial.trial_number <= CALIBRATION_TRIALS:
                scenario_text += " | CALIBRATION"

        cue = self.font.render(
            scenario_text,
            True,
            self._command_color(trial.ground_truth),
        )

        cue_rect = cue.get_rect(
            center=(WIDTH // 2, 100)
        )

        self.screen.blit(cue, cue_rect)

        progress = self.small_font.render(
            (
                f"Trial "
                f"{self.current_trial_index + 1}"
                f"/{len(self.trials)}"
            ),
            True,
            (220, 220, 220),
        )

        self.screen.blit(
            progress,
            (20, 20),
        )

    def _draw_target(self, trial):
        pygame.draw.circle(
            self.screen,
            self._command_color(trial.ground_truth),
            (
                int(trial.goal_x),
                HEIGHT // 2,
            ),
            BALL_RADIUS + 10,
            4,
        )

    def _command_color(self, command):
        if command == Command.RIGHT:
            return (80, 220, 80)

        if command == Command.LEFT:
            return (80, 140, 255)

        return (255, 255, 255)

    def _draw_session_complete(self):
        text = self.font.render(
            "ERRP SESSION COMPLETE",
            True,
            (255, 255, 255),
        )

        rect = text.get_rect(
            center=(WIDTH // 2, HEIGHT // 2)
        )

        self.screen.blit(text, rect)

    def _draw_inter_trial(self):
        title = self.font.render("TRIAL COMPLETE", True, TEXT_COLOR)
        progress = self.small_font.render(
            f"Trial {self.current_trial_index}/{len(self.trials)} finished",
            True, TEXT_COLOR,
        )
        prompt_text = (
            "Press SPACE to continue"
            if self.inter_trial_elapsed >= ERRP_INTER_TRIAL_MIN_TIME
            else "Please wait..."
        )
        prompt = self.small_font.render(prompt_text, True, TEXT_COLOR)

        for surface, y in (
            (title, HEIGHT // 2 - 65),
            (progress, HEIGHT // 2),
            (prompt, HEIGHT // 2 + 65),
        ):
            self.screen.blit(surface, surface.get_rect(center=(WIDTH // 2, y)))

    def _finish_trial(self):
        self.marker_sender.send(
            f"trial_end:{self.current_trial.trial_type}:"
            f"{self.current_trial.scenario}"
        )
        self.current_trial.completed = True
        self.current_trial_index += 1

        if self.current_trial_index >= len(self.trials):
            self.load_trial()  # Show the existing session-complete screen.
        else:
            self.waiting_for_space = True
            self.inter_trial_elapsed = 0.0
            self.player.direction = 0

    def is_complete(self):
        return self.session_complete

    def close(self):
        self.logger.close()

