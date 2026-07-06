import pygame

from mind_the_lasers.src.game.settings import *
from mind_the_lasers.src.game.player import Player
from mind_the_lasers.src.game.level import make_levels
from mind_the_lasers.src.game.input_controller import KeyboardController


def main():
    game = Game()
    game.run()


class Game:
    def __init__(self, controller = KeyboardController()):
        pygame.init()

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Mind the Lasers")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)
        self.big_font = pygame.font.SysFont(None, 72)

        self.player = Player()
        self.controller = controller

        self.levels = make_levels()
        self.level_index = 0
        self.game_over = False

        self.load_level()

    def load_level(self):
        self.level = self.levels[self.level_index]
        self.level.reset_player(self.player)

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
        command = self.controller.get_command()

        self.player.process_command(command)
        self.player.update(dt)
        self.level.update(dt)

        for laser in self.level.lasers:
            if laser.collides(self.player):
                self.player.hit()

        if self.player.lives <= 0:
            self.game_over = True
            return

        if self.level.completed(self.player):
            self.level_index += 1

            if self.level_index >= len(self.levels):
                self.level_index = 0

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


if __name__ == "__main__":
    main()