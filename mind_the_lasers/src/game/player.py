import pygame
from mind_the_lasers.src.game.settings import *


class Player:
    def __init__(self):
        self.x = 80
        self.y = HEIGHT // 2

        self.lives = MAX_LIVES

        self.hit_flash_timer = 0

        self.direction = 0  # -1 = left, 0 = stop, 1 = right

        self.base_speed = BALL_SPEED

        self.speed = BALL_SPEED

        self.boost_level = 0
        self.boost_timer = 0

        self.rest_timer = 0

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
        self.speed = self.base_speed
        self.boost_level = 0
        self.rest_timer = 0
        self.boost_timer = 0

    def reset_lives(self):
        self.lives = MAX_LIVES

    def process_command(self, command):

        if command.name == "NONE":
            return

        if command.value == 0:
            self.direction = 0
            return

        if self.direction == 0:
            self.speed = self.base_speed + self.boost_level * BOOST_PER_LEVEL

            if self.boost_level > 0:
                self.boost_timer = BOOST_DURATION

            self.boost_level = 0
            self.rest_timer = 0

        self.direction = command.value

    def hit(self):
        self.lives -= 1
        self.hit_flash_timer = HIT_FLASH_TIME

    def update(self, dt):

        if self.direction == 0:
            self.speed = self.base_speed
            self.boost_timer = 0

            self.rest_timer += dt

            while (
                self.rest_timer >= BOOST_CHARGE_TIME
                and self.boost_level < MAX_BOOST_LEVEL
            ):
                self.boost_level += 1
                self.rest_timer -= BOOST_CHARGE_TIME

        else:
            if self.boost_timer > 0:
                self.boost_timer -= dt

                if self.boost_timer <= 0:
                    self.speed = self.base_speed

        self.x += self.direction * self.speed * dt

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