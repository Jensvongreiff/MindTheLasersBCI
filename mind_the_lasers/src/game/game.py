import pygame

from settings import *
from player import Player
from level import make_levels
from game_metrics import GameMetrics
from input_controller import KeyboardController, UDPController, Command


class Game:
    def __init__(self, screen, controller):
        self.screen = screen
        self.controller = controller

        self.font = pygame.font.SysFont(None, 36)
        self.big_font = pygame.font.SysFont(None, 72)

        self.player = Player()

        self.levels = make_levels()

        self.metrics = GameMetrics(
            total_levels=len(self.levels)
        )

        self.summary_ready = False

        self.previous_command = Command.NONE

        self.level_index = 0
        self.game_over = False

        self.load_level()

    def load_level(self):
        self.level = self.levels[self.level_index]

        self.level.reset_player(self.player)

        self.player.direction = 0

        if RESET_LIVES_EVERY_LEVEL:
            self.player.reset_lives()

        self.controller.reset()

        self.previous_command = Command.NONE

    def restart_game(self):
        self.level_index = 0
        self.player.reset_lives()
        self.game_over = False
        self.load_level()

    def run(self):
        running = True

        while running:
            dt = self.clock.tick(FPS) / 1000

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if self.game_over and event.key == pygame.K_SPACE:
                        self.restart_game()

            if not self.game_over:
                self.update(dt)

            self.draw()

        pygame.quit()

    def update(self, dt):
        if self.game_over:
            return

        self.controller.update()

        command = self.controller.get_command()

        # Only count a command when the persistent game command changes.
        if command != self.previous_command:
            self.metrics.record_command(command)
            self.previous_command = command

        self.metrics.record_movement_time(
            command=command,
            goal_direction=self._goal_direction(),
            dt=dt,
        )

        # Detect a newly activated boost.
        was_boosted = self._player_is_boosted()

        self.player.process_command(command)

        is_boosted = self._player_is_boosted()

        if is_boosted and not was_boosted:
            self.metrics.record_boost_used()

        self.player.update(dt)

        self.level.update(dt)

        self._record_laser_metrics()

        for laser in self.level.lasers:
            if laser.collides(self.player):
                self.metrics.record_collision(
                    laser,
                    boosted=self._player_is_boosted(),
                )

                self.player.hit()

        if self.player.lives <= 0:
            self._finish_game()
            return

        if self.level.completed(self.player):
            next_level_number = self.level_index + 2

            self.metrics.record_level_completed(
                next_level_number=next_level_number,
            )

            self.level_index += 1

            if self.level_index >= len(self.levels):
                self._finish_game()
                return

            self.load_level()

    def draw_hearts(self):
        for i in range(MAX_LIVES):
            x = 25 + i * 34
            y = 25

            color = HEART_COLOR if i < self.player.lives else EMPTY_HEART_COLOR

            pygame.draw.circle(self.screen, color, (x, y), 10)
            pygame.draw.circle(self.screen, color, (x + 12, y), 10)

            points = [
                (x - 10, y + 4),
                (x + 22, y + 4),
                (x + 6, y + 24),
            ]

            pygame.draw.polygon(self.screen, color, points)

    def draw_boost(self):

        for i in range(MAX_BOOST_LEVEL):

            x = WIDTH - 170 + i * 30
            y = 25

            color = (255, 215, 0) if i < self.player.boost_level else (90, 90, 90)

            pygame.draw.polygon(
                self.screen,
                color,
                [
                    (x + 8, y),
                    (x + 16, y + 12),
                    (x + 10, y + 12),
                    (x + 18, y + 28),
                    (x + 6, y + 18),
                    (x + 12, y + 18),
                ],
            )

    def draw_goal(self):
        pygame.draw.circle(
            self.screen,
            GOAL_COLOR,
            (int(self.level.goal_x), HEIGHT // 2),
            26,
        )

    def draw_hud(self):
        self.draw_hearts()
        self.draw_boost()

        text = self.font.render(
            f"Level {self.level_index + 1}: {self.level.name}",
            True,
            TEXT_COLOR,
        )

        self.screen.blit(text, (20, 65))

        controls = self.font.render(
            "Left / Right = move   Down = rest",
            True,
            TEXT_COLOR,
        )

        self.screen.blit(controls, (20, HEIGHT - 40))

    def draw_game_over(self):
        title = self.big_font.render("GAME OVER", True, HIT_COLOR)
        prompt = self.font.render("Press SPACE to restart", True, TEXT_COLOR)

        self.screen.blit(
            title,
            (
                WIDTH // 2 - title.get_width() // 2,
                HEIGHT // 2 - 60,
            ),
        )

        self.screen.blit(
            prompt,
            (
                WIDTH // 2 - prompt.get_width() // 2,
                HEIGHT // 2 + 10,
            ),
        )

    def draw(self):
        self.screen.fill(BG_COLOR)

        self.draw_goal()

        for laser in self.level.lasers:
            laser.draw(self.screen)
            

        self.player.draw(self.screen)
        self.draw_hud()

        if self.game_over:
            self.draw_game_over()

        pygame.display.flip()


    def draw_summary(self):
        self.screen.fill(BG_COLOR)

        summary = self.metrics.get_summary()

        title = self.big_font.render(
            "GAME SUMMARY",
            True,
            (255, 255, 255),
        )

        self.screen.blit(
            title,
            title.get_rect(center=(WIDTH // 2, 70)),
        )

        lines = [
            (
                f"Highest Level Reached: "
                f"{summary['highest_level_reached']}"
                f"/{summary['total_levels']}"
            ),
            (
                f"Levels Completed: "
                f"{summary['levels_completed']}"
            ),
            (
                f"Play Time: "
                f"{summary['total_play_time']:.1f} s"
            ),
            (
                f"Average Level Time: "
                f"{summary['average_level_time']:.1f} s"
            ),
            (
                f"Laser Avoidances: "
                f"{summary['successful_avoidances']}"
                f"/{summary['laser_encounters']} "
                f"({summary['avoidance_rate'] * 100:.1f}%)"
            ),
            (
                f"Collisions: "
                f"{summary['collisions']}"
            ),
            (
                f"Command Transitions: "
                f"{summary['command_transitions']}"
            ),
            (
                f"Effective Movement: "
                f"{summary['effective_movement_ratio'] * 100:.1f}%"
            ),
            (
                f"Time Resting: "
                f"{summary['time_resting']:.1f} s"
            ),
            (
                f"Boosts Used: "
                f"{summary['boosts_used']}"
            ),
            (
                f"Boosted Collisions: "
                f"{summary['boosted_collisions']}"
            ),
        ]

        y = 140

        for text in lines:
            rendered = self.font.render(
                text,
                True,
                (230, 230, 230),
            )

            self.screen.blit(
                rendered,
                rendered.get_rect(center=(WIDTH // 2, y)),
            )

            y += 38

        prompt = self.font.render(
            "Press ENTER to return to menu",
            True,
            (200, 200, 200),
        )

        self.screen.blit(
            prompt,
            prompt.get_rect(
                center=(WIDTH // 2, HEIGHT - 40)
            ),
        )


    def _goal_direction(self):
        if self.level.goal_x > self.level.start_x:
            return 1

        return -1


    def _player_is_boosted(self):
        return self.player.speed > self.player.base_speed


    def _record_laser_metrics(self):
        goal_direction = self._goal_direction()

        for laser in self.level.lasers:
            laser_id = id(laser)

            # The player has reached/passed the laser position
            # while traveling toward the goal.
            if goal_direction == 1:
                reached_laser = self.player.x >= laser.x
            else:
                reached_laser = self.player.x <= laser.x

            if reached_laser:
                self.metrics.record_laser_encounter(laser)

            # Once the player has moved fully beyond the laser,
            # count it as an avoidance if no collision occurred.
            if goal_direction == 1:
                passed_laser = (
                    self.player.x - BALL_RADIUS
                    > laser.x + laser.beam_width // 2
                )
            else:
                passed_laser = (
                    self.player.x + BALL_RADIUS
                    < laser.x - laser.beam_width // 2
                )

            if passed_laser:
                self.metrics.record_avoidance(laser)


    def _finish_game(self):
        self.game_over = True

        self.player.direction = 0
        self.controller.reset()

        self.metrics.stop_timer()
        self.metrics.save()

        self.summary_ready = True