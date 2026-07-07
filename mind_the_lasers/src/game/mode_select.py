import pygame

from mind_the_lasers.src.game.settings import *


class ModeSelect:
    def __init__(self, screen):
        self.screen = screen

        self.title_font = pygame.font.Font(None, 64)
        self.font = pygame.font.Font(None, 40)

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

        title = self.title_font.render(
            "MIND THE LASERS",
            True,
            (255, 255, 255),
        )

        title_rect = title.get_rect(
            center=(WIDTH // 2, 150)
        )

        self.screen.blit(title, title_rect)

        training = self.font.render(
            "1 - TRAINING MODE",
            True,
            (255, 255, 255),
        )

        play = self.font.render(
            "2 - PLAY MODE",
            True,
            (255, 255, 255),
        )

        self.screen.blit(
            training,
            training.get_rect(
                center=(WIDTH // 2, 300)
            ),
        )

        self.screen.blit(
            play,
            play.get_rect(
                center=(WIDTH // 2, 380)
            ),
        )