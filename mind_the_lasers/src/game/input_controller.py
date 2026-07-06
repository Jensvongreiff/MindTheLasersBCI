import pygame
import queue
import multiprocessing
from enum import Enum
from sklearn.metrics import confusion_matrix, classification_report
from mind_the_lasers.src.pipeline.smoothing import SmoothingController

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
    def __init__(self, output_queue: multiprocessing.Queue, smoothing_controller: SmoothingController):
        self.output_queue = output_queue
        self.smoothing = smoothing_controller
        self.current_command = Command.REST
        self.y_true, self.y_pred = [], []
        self._map = {"left": Command.LEFT, "right": Command.RIGHT, "rest": Command.REST}

    def get_command(self) -> Command:
        latest_data = False
        while True:
            try:
                msg = self.output_queue.get_nowait()
                if msg is None: # Catch the EOF sentinel
                    self._generate_report()
                    pygame.event.post(pygame.event.Event(pygame.QUIT))
                    return Command.NONE
                latest_data = msg
            except queue.Empty:
                break
                
        if latest_data:
            smoothed_str = self.smoothing.process(latest_data["probabilities"])
            self.current_command = self._map.get(smoothed_str, Command.REST)

            # Automated evaluation logic
            truth = latest_data.get("ground_truth")
            if truth:
                self.y_true.append(truth)
                self.y_pred.append(smoothed_str)

        return self.current_command

    def _generate_report(self):
        if not self.y_true: return
        print("\n" + "="*40 + "\nOFFLINE SIMULATION METRICS REPORT\n" + "="*40)
        print(classification_report(self.y_true, self.y_pred, target_names=["left", "rest", "right"]))
        print("Confusion Matrix:")
        print(confusion_matrix(self.y_true, self.y_pred, labels=["left", "rest", "right"]))
        print("="*40 + "\n")

class UnifiedController:
    def __init__(self, bci_controller: BCIController, keyboard_controller: KeyboardController):
        self.bci = bci_controller
        self.keyboard = keyboard_controller

    def get_command(self) -> Command:
        kb_cmd = self.keyboard.get_command()
        return kb_cmd if kb_cmd != Command.NONE else self.bci.get_command()