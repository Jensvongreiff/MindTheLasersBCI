import pygame

from mind_the_lasers.src.game.settings import *


class ModeSelect:
    def __init__(self, screen):
        self.screen = screen

        self.title_font = pygame.font.Font(None, 120)
        self.option_font = pygame.font.Font(None, 44)
        self.hint_font = pygame.font.Font(None, 28)

    def handle_event(self, event):
        if event.type != pygame.KEYDOWN:
            return None

        if event.key == pygame.K_1:
            return "training"

        if event.key == pygame.K_2:
            return "play"

        return None

    def draw(self):
        self.screen.fill(BG_COLOR)

        center_x = WIDTH // 2

        mind_text = self.title_font.render(
            "MIND THE ",
            True,
            (255, 255, 255),
        )

        lasers_text = self.title_font.render(
            "LASERS",
            True,
            LASER_COLOR,
        )

        total_width = (
            mind_text.get_width()
            + lasers_text.get_width()
        )

        start_x = center_x - total_width // 2

        title_y = 120

        self.screen.blit(
            mind_text,
            (start_x, title_y),
        )

        self.screen.blit(
            lasers_text,
            (
                start_x + mind_text.get_width(),
                title_y,
            ),
        )

        training = self.option_font.render(
            "1 - TRAINING MODE",
            True,
            (230, 230, 230),
        )

        play = self.option_font.render(
            "2 - PLAY MODE",
            True,
            (230, 230, 230),
        )

        self.screen.blit(
            training,
            training.get_rect(
                center=(center_x, 350),
            ),
        )

        self.screen.blit(
            play,
            play.get_rect(
                center=(center_x, 415),
            ),
        )

        hint = self.hint_font.render(
            "ESC - Quit",
            True,
            (160, 160, 160),
        )

        self.screen.blit(
            hint,
            hint.get_rect(
                center=(center_x, HEIGHT - 45),
            ),
        )