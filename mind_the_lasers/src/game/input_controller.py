import pygame
from enum import Enum
import collections
from typing import Dict
import queue
import multiprocessing
from mind_the_lasers.src.pipeline.smoothing import SmoothingController


class Command(Enum):
    LEFT = -1
    REST = 0
    RIGHT = 1
    NONE = 99

class KeyboardController:
    def get_command(self):
        keys = pygame.key.get_pressed()

        if keys[pygame.K_DOWN]:
            return Command.REST

        if keys[pygame.K_RIGHT]:
            return Command.RIGHT

        if keys[pygame.K_LEFT]:
            return Command.LEFT

        return Command.NONE
    
class BCIController:
    """Reads non-blocking from the BCI worker queue and applies smoothing."""
    def __init__(self, output_queue: multiprocessing.Queue, smoothing_controller: SmoothingController):
        self.output_queue = output_queue
        self.smoothing = smoothing_controller
        self.current_command = Command.REST
        
        self._map = {
            "left": Command.LEFT,
            "right": Command.RIGHT,
            "rest": Command.REST
        }

    def get_command(self) -> Command:
        # Drain the queue to fetch the most recent pipeline execution
        latest_data = None
        while True:
            try:
                latest_data = self.output_queue.get_nowait()
            except queue.Empty:
                break
                
        if latest_data is not None:
            raw_probs = latest_data["probabilities"]
            smoothed_str = self.smoothing.process(raw_probs)
            self.current_command = self._map.get(smoothed_str, Command.REST)

        return self.current_command

class UnifiedController:
    """Prioritizes manual keyboard overrides during active BCI play."""
    def __init__(self, bci_controller: BCIController, keyboard_controller: KeyboardController):
        self.bci = bci_controller
        self.keyboard = keyboard_controller

    def get_command(self) -> Command:
        kb_cmd = self.keyboard.get_command()
        if kb_cmd != Command.NONE:
            return kb_cmd
        return self.bci.get_command()