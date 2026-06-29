import pygame
from settings import *


class Player:
    def __init__(self):
        self.x = 80
        self.y = HEIGHT // 2

        self.lives = MAX_LIVES

        self.hit_flash_timer = 0

        self.direction = 0  # -1 = left, 0 = stop, 1 = right

    @property
    def rect(self):
        return pygame.Rect(
            self.x - BALL_RADIUS,
            self.y - BALL_RADIUS,
            BALL_RADIUS * 2,
            BALL_RADIUS * 2,
        )

    def reset_position(self, x):
        self.x = x
        self.y = HEIGHT // 2
        self.direction = 0

    def reset_lives(self):
        self.lives = MAX_LIVES

    def process_command(self, command):
        if command.name == "NONE":
            return

        self.direction = command.value

    def hit(self):
        self.lives -= 1
        self.hit_flash_timer = HIT_FLASH_TIME

    def update(self, dt):
        self.x += self.direction * BALL_SPEED * dt

        self.x = max(BALL_RADIUS, min(WIDTH - BALL_RADIUS, self.x))

        if self.hit_flash_timer > 0:
            self.hit_flash_timer -= dt

    def draw(self, screen):
        color = HIT_COLOR if self.hit_flash_timer > 0 else BALL_COLOR

        pygame.draw.circle(
            screen,
            color,
            (int(self.x), int(self.y)),
            BALL_RADIUS,
        )