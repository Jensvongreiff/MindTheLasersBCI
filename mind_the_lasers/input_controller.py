import pygame
from enum import Enum


class Command(Enum):
    LEFT = -1
    REST = 0
    RIGHT = 1


class KeyboardController:
    def get_command(self):
        keys = pygame.key.get_pressed()

        if keys[pygame.K_DOWN]:
            return Command.REST

        if keys[pygame.K_RIGHT]:
            return Command.RIGHT

        if keys[pygame.K_LEFT]:
            return Command.LEFT

        return Command.REST