import pygame

from mind_the_lasers.src.game.input_controller import Command
from mind_the_lasers.src.game.player import Player
from mind_the_lasers.src.game.settings import *

from mind_the_lasers.src.game.training_logger import TrainingLogger
from mind_the_lasers.src.game.training_trial import TrainingTrial

from mind_the_lasers.src.game.settings import *

class TrainingMode:
    def __init__(
        self,
        screen,
        controller,
        marker_sender,
        trials_per_command=TRAINING_TRIALS_PER_COMMAND,
    ):
        self.screen = screen
        self.controller = controller
        self.marker_sender = marker_sender

        self.player = Player()

        self.logger = TrainingLogger()

        self.font = pygame.font.Font(None, 48)
        self.small_font = pygame.font.Font(None, 30)

        self.trials = self._create_trials(trials_per_command)

        self.current_trial_index = 0
        self.current_trial = None

        self.session_complete = False

        self.rest_auto_direction = 1

        self.feedback_timer = 0.0

        self.REST_HOLD_TIME = 2.0
        self.rest_hold_timer = 0.0

        self.TRIAL_FEEDBACK_TIME = 0.5

        self.summary = None

        self.prediction_received = False

        self.command_delay_timer = 0.0
        self.command_delay_finished = False


        self.load_trial()

    def _create_trials(self, trials_per_command):
        trials = []

        trial_number = 1

        for _ in range(trials_per_command):
            trials.append(
                TrainingTrial(
                    trial_number=trial_number,
                    ground_truth=Command.RIGHT,
                    start_x=80,
                    goal_x=920,
                )
            )

            trial_number += 1

        for _ in range(trials_per_command):
            trials.append(
                TrainingTrial(
                    trial_number=trial_number,
                    ground_truth=Command.LEFT,
                    start_x=920,
                    goal_x=80,
                )
            )

            trial_number += 1

        for _ in range(trials_per_command):
            trials.append(
                TrainingTrial(
                    trial_number=trial_number,
                    ground_truth=Command.REST,
                    start_x=WIDTH // 2,
                    goal_x=WIDTH // 2,
                )
            )

            trial_number += 1

        return trials

    def load_trial(self):
        if self.current_trial_index >= len(self.trials):
            self.session_complete = True
            self.current_trial = None
            self.summary = self._compute_summary()
            return

        self.current_trial = self.trials[self.current_trial_index]

        self.player.reset_position(
            self.current_trial.start_x
        )

        self.player.direction = 0

        self.rest_auto_direction = 1

        self.feedback_timer = 0.0
        self.rest_hold_timer = 0.0

        self.prediction_received = False
        self.controller.reset()

        self.command_delay_timer = 0.0
        self.command_delay_finished = False

        self.current_trial.start()

        ground_truth = self.current_trial.ground_truth.name.lower()

        self.marker_sender.send(
            f"trial_start:{ground_truth}"
        )

    def update(self, dt):
        if self.session_complete:
            return

        self.controller.update()

        trial = self.current_trial

        if not trial.decision_made:
            if not self.command_delay_finished:
                self.command_delay_timer += dt

                # Drain predictions received during the preparation period.
                self.controller.get_new_prediction()

                if self.command_delay_timer >= TRAINING_COMMAND_DELAY:
                    self.command_delay_finished = True

                    # Clear anything still buffered before opening
                    # the decision window.
                    self.controller.reset()

            elif not self.prediction_received:
                prediction = self.controller.get_new_prediction()

                if prediction is not None:
                    self.prediction_received = True

                    confidence = self.controller.confidence

                    self._handle_decision(
                        prediction,
                        confidence,
                    )

        # Decision has already been made.
        else:
            self.controller.get_new_prediction()
            self._update_after_decision(dt)

        # Before a decision, REST trials move automatically.
        if not trial.decision_made:
            if trial.ground_truth == Command.REST:
                self._update_rest_trial(dt)

    def _handle_decision(self, command, confidence):
        trial = self.current_trial

        decision_time = trial.decision_time()

        trial.accept_command(command)

        if trial.correct:
            game_event = "correct_command"
        else:
            game_event = "incorrect_command"

        self.logger.log_decision(
            trial_number=trial.trial_number,
            ground_truth_command=trial.ground_truth.name,
            predicted_command=command.name,
            confidence=confidence,
            accepted_command=command.name,
            correct=trial.correct,
            decision_time=decision_time,
            game_event=game_event,
        )

        # Incorrect command:
        # show brief feedback, then advance.
        if not trial.correct:
            self.feedback_timer = self.TRIAL_FEEDBACK_TIME

    def _update_after_decision(self, dt):
        trial = self.current_trial

        # Incorrect decision:
        # wait briefly before next trial.
        if not trial.correct:
            self.feedback_timer -= dt

            if self.feedback_timer <= 0:
                self._finish_trial()

            return

        # Correct REST:
        # ball remains stopped for 2 seconds.
        if trial.ground_truth == Command.REST:
            self.player.direction = 0

            self.rest_hold_timer += dt

            if self.rest_hold_timer >= self.REST_HOLD_TIME:
                self._finish_trial()

            return

        # Correct LEFT / RIGHT:
        # visibly move ball toward target.
        self.player.direction = trial.ground_truth.value

        self.player.x += (
            self.player.direction
            * BALL_SPEED_TRAINING
            * dt
        )

        self.player.x = max(
            BALL_RADIUS,
            min(WIDTH - BALL_RADIUS, self.player.x),
        )

        if trial.ground_truth == Command.RIGHT:
            reached_target = self.player.x >= trial.goal_x

        else:
            reached_target = self.player.x <= trial.goal_x

        if reached_target:
            self.player.x = trial.goal_x
            self.feedback_timer = self.TRIAL_FEEDBACK_TIME

            trial.completed = True
            self._finish_trial()

    def _update_player(self, dt):
        trial = self.current_trial

        if trial is None:
            return

        if trial.ground_truth == Command.REST:
            self._update_rest_trial(dt)
            return

        self.player.update(dt)

    def _update_rest_trial(self, dt):
        self.player.x += (
            self.rest_auto_direction
            * BALL_SPEED_TRAINING
            * dt
        )

        if self.player.x >= WIDTH - BALL_RADIUS:
            self.player.x = WIDTH - BALL_RADIUS
            self.rest_auto_direction = -1

        elif self.player.x <= BALL_RADIUS:
            self.player.x = BALL_RADIUS
            self.rest_auto_direction = 1

        if self.player.hit_flash_timer > 0:
            self.player.hit_flash_timer -= dt

    def _compute_summary(self):
        summary = {}

        for command in (Command.RIGHT, Command.LEFT, Command.REST):
            trials = [
                trial for trial in self.trials
                if trial.ground_truth == command
            ]

            total = len(trials)

            correct = sum(
                1 for trial in trials
                if trial.correct is True
            )

            accuracy = correct / total if total > 0 else 0.0

            summary[command.name] = {
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
            }

        return summary

    def draw(self):
        self.screen.fill(BG_COLOR)

        if self.session_complete:
            self._draw_summary()
            return

        trial = self.current_trial

        self._draw_training_header(trial)
        self._draw_target(trial)
        self.player.draw(self.screen)

        if trial.decision_made:
            self._draw_decision_feedback(trial)

    def _draw_decision_feedback(self, trial):
        if trial.correct:
            text = "CORRECT"
            color = (80, 220, 80)
        else:
            text = "INCORRECT"
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



    def _draw_training_header(self, trial):
        title = self.font.render(
            "TRAINING MODE",
            True,
            (255, 255, 255),
        )

        title_rect = title.get_rect(
            center=(WIDTH // 2, 40)
        )

        self.screen.blit(title, title_rect)

        cue = self.font.render(
            f"COMMAND: {trial.ground_truth.name}",
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
        if trial.ground_truth == Command.REST:
            text = self.small_font.render(
                "Stop the ball",
                True,
                (255, 255, 255),
            )

            rect = text.get_rect(
                center=(WIDTH // 2, HEIGHT - 60)
            )

            self.screen.blit(text, rect)
            return

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

    def _draw_summary(self):
        title = self.font.render(
            "TRAINING SUMMARY",
            True,
            (255, 255, 255),
        )

        self.screen.blit(
            title,
            title.get_rect(center=(WIDTH // 2, 90)),
        )

        y = 180

        for command_name in ("RIGHT", "LEFT", "REST"):
            data = self.summary[command_name]

            text = (
                f"{command_name}: "
                f"{data['correct']}/{data['total']} correct "
                f"({data['accuracy'] * 100:.1f}%)"
            )

            line = self.small_font.render(
                text,
                True,
                (230, 230, 230),
            )

            self.screen.blit(
                line,
                line.get_rect(center=(WIDTH // 2, y)),
            )

            y += 45

        prompt = self.small_font.render(
            "Press ENTER to return to menu",
            True,
            (200, 200, 200),
        )

        self.screen.blit(
            prompt,
            prompt.get_rect(center=(WIDTH // 2, HEIGHT - 80)),
        )

    def _command_color(self, command):
        if command == Command.RIGHT:
            return (80, 220, 80)

        if command == Command.LEFT:
            return (80, 140, 255)

        if command == Command.REST:
            return (255, 220, 80)

        return (255, 255, 255)

    def _draw_session_complete(self):
        text = self.font.render(
            "TRAINING COMPLETE",
            True,
            (255, 255, 255),
        )

        rect = text.get_rect(
            center=(WIDTH // 2, HEIGHT // 2)
        )

        self.screen.blit(text, rect)

    def _finish_trial(self):
        self.current_trial.completed = True

        self.current_trial_index += 1

        self.load_trial()

    def is_complete(self):
        return self.session_complete

    def close(self):
        self.logger.close()

