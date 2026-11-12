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

        if event.key == pygame.K_3:
            return "errp"

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

        errp = self.option_font.render(
            "3 - ERRP EXPERIMENT MODE",
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

        self.screen.blit(
            errp,
            errp.get_rect(
                center=(center_x, 480),
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

class ErrPModeSelect:
    def __init__(self, screen):
        self.screen = screen

        self.title_font = pygame.font.Font(None, 120)
        self.option_font = pygame.font.Font(None, 44)
        self.hint_font = pygame.font.Font(None, 28)

    def handle_event(self, event):
        if event.type != pygame.KEYDOWN:
            return None

        if event.key == pygame.K_1:
            return "observation"

        if event.key == pygame.K_2:
            return "interactive"

        if event.key == pygame.K_3:
            return "interactive_continuous"

        return None

    def draw(self):
        self.screen.fill(BG_COLOR)

        center_x = WIDTH // 2

        mind_text = self.title_font.render(
            "ErrP ",
            True,
            (255, 255, 255),
        )

        lasers_text = self.title_font.render(
            "MODE",
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

        observation = self.option_font.render(
            "1 - OBSERVATION MODE",
            True,
            (230, 230, 230),
        )

        interactive = self.option_font.render(
            "2 - INTERACTIVE MODE",
            True,
            (230, 230, 230),
        )

        interactive_continuous = self.option_font.render(
            "3 - INTERACTIVE CONTINUOUS MODE",
            True,
            (230, 230, 230),
        )

        self.screen.blit(
            observation,
            observation.get_rect(
                center=(center_x, 350),
            ),
        )

        self.screen.blit(
            interactive,
            interactive.get_rect(
                center=(center_x, 415),
            ),
        )

        self.screen.blit(
            interactive_continuous,
            interactive_continuous.get_rect(
                center=(center_x, 480),
            ),
        )

        hint = self.hint_font.render(
            "ESC - Back",
            True,
            (160, 160, 160),
        )

        self.screen.blit(
            hint,
            hint.get_rect(
                center=(center_x, HEIGHT - 45),
            ),
        )