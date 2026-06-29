import pygame
from settings import *


class Player:
    def __init__(self):
        self.x = 80
        self.y = HEIGHT // 2
        self.lives = MAX_LIVES
        self.hit_flash_timer = 0
        self.invincible_timer = 0
        self.direction = 0

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

    def reset_lives(self):
        self.lives = MAX_LIVES

    def hit(self):
        if self.invincible_timer > 0:
            return

        self.lives -= 1
        self.hit_flash_timer = HIT_FLASH_TIME
        self.invincible_timer = INVINCIBLE_TIME

    def update(self, command, dt):
        self.x += command.value * BALL_SPEED * dt
        self.x = max(BALL_RADIUS, min(WIDTH - BALL_RADIUS, self.x))

        self.hit_flash_timer = max(0, self.hit_flash_timer - dt)
        self.invincible_timer = max(0, self.invincible_timer - dt)

    def draw(self, screen):
        color = HIT_COLOR if self.hit_flash_timer > 0 else BALL_COLOR

        pygame.draw.circle(
            screen,
            color,
            (int(self.x), int(self.y)),
            BALL_RADIUS,
        )