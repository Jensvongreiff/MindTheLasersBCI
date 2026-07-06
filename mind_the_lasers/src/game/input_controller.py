import pygame
import queue
import multiprocessing
from enum import Enum
from mind_the_lasers.src.pipeline.smoothing import SmoothingController
from mind_the_lasers.src.pipeline.decoder_metrics import DecoderEvaluator

class Command(Enum):
    LEFT = -1
    REST = 0
    RIGHT = 1
    NONE = 99

class KeyboardController:
    def get_command(self) -> Command:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_DOWN]: return Command.REST
        if keys[pygame.K_RIGHT]: return Command.RIGHT
        if keys[pygame.K_LEFT]: return Command.LEFT
        return Command.NONE

class BCIController:
    """Interacts with the BCI Pipeline and performs automated offline evaluations."""
    def __init__(self, output_queue: multiprocessing.Queue, smoothing_controller: SmoothingController, baseline_name: str = "pipeline"):
        self.output_queue = output_queue
        self.smoothing = smoothing_controller
        self.evaluator = DecoderEvaluator()
        self.baseline_name = baseline_name
        self.current_command = Command.REST
        self._map = {"left": Command.LEFT, "right": Command.RIGHT, "rest": Command.REST}

    def get_command(self) -> Command:
        while True:
            try:
                msg = self.output_queue.get_nowait()
                if msg is None: # Catch the EOF sentinel
                    self.evaluator.generate_report(self.baseline_name)
                    pygame.event.post(pygame.event.Event(pygame.QUIT))
                    return Command.NONE
                
                # PROCESS AND LOG EVERY BCI WINDOW IN THE QUEUE
                smoothed_str, max_conf, was_rejected = self.smoothing.process(msg["probabilities"])
                
                # Continuously update the current command; the loop will exit with the absolute latest.
                self.current_command = self._map.get(smoothed_str, Command.REST)

                # Log the metric for this specific window
                truth = msg.get("ground_truth")
                if truth:
                    latency = msg.get("latency", 0.0)
                    self.evaluator.log_step(
                        truth=truth,
                        pred=smoothed_str,
                        confidence=max_conf,
                        rejected=was_rejected,
                        latency=latency
                    )

            except queue.Empty:
                break # Queue is fully processed
                
        return self.current_command

class UnifiedController:
    def __init__(self, bci_controller: BCIController, keyboard_controller: KeyboardController):
        self.bci = bci_controller
        self.keyboard = keyboard_controller

    def get_command(self) -> Command:
        kb_cmd = self.keyboard.get_command()
        return kb_cmd if kb_cmd != Command.NONE else self.bci.get_command()