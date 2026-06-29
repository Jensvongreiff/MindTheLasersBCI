import pygame
from settings import *


class SweepingLaser:
    def __init__(
        self,
        x,
        wait_time=1.5,
        grow_speed=500,
        beam_width=10,
        active_time=0.4,
    ):

        self.x = x

        self.wait_time = wait_time
        self.grow_speed = grow_speed
        self.beam_width = beam_width
        self.active_time = active_time

        self.state = "waiting"

        self.timer = wait_time

        self.current_height = 0

        self.has_hit_player = False

    def update(self, dt):

        if self.state == "waiting":

            self.timer -= dt

            if self.timer <= 0:
                self.state = "growing"
                self.current_height = 0

        elif self.state == "growing":

            self.current_height += self.grow_speed * dt

            if self.current_height >= HEIGHT:
                self.current_height = HEIGHT
                self.state = "active"
                self.timer = self.active_time
                self.has_hit_player = False

        elif self.state == "active":

            self.timer -= dt

            if self.timer <= 0:
                self.state = "waiting"
                self.timer = self.wait_time

    @property
    def rect(self):

        return pygame.Rect(
            self.x - self.beam_width // 2,
            0,
            self.beam_width,
            self.current_height,
        )

    def collides(self, player):

        if self.state not in ("growing", "active"):
            return False

        return self.rect.colliderect(player.rect)

    def draw(self, screen):

        if self.state == "waiting":
            return

        pygame.draw.rect(
            screen,
            LASER_COLOR,
            self.rect,
        )

        # brighter center
        pygame.draw.line(
            screen,
            (255, 180, 180),
            (self.x, 0),
            (self.x, self.current_height),
            2,
        )