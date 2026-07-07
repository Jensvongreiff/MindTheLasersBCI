import pygame
from enum import Enum

import socket
import json


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



class UDPController:
    def __init__(self, ip="127.0.0.1", port=5005):
        # Persistent command used by Play Mode.
        self.command = Command.NONE

        # Most recently received raw prediction.
        self.latest_prediction = Command.NONE

        # Confidence belonging to the most recent prediction.
        self.confidence = 0.0

        # True when a new UDP packet has arrived and has not yet
        # been consumed by get_new_prediction().
        self.new_prediction_available = False

        self.sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )

        self.sock.bind((ip, port))

        self.sock.setblocking(False)

        print(f"Listening for UDP predictions on {ip}:{port}")

    def update(self):
        try:
            while True:
                data, _ = self.sock.recvfrom(1024)

                msg = json.loads(
                    data.decode("utf-8")
                )

                pred = msg["prediction"]

                self.confidence = msg.get(
                    "confidence",
                    0.0,
                )

                if pred == 0:
                    received_command = Command.REST

                elif pred == 1:
                    received_command = Command.LEFT

                elif pred == 2:
                    received_command = Command.RIGHT

                else:
                    continue

                # Store the raw classifier prediction.
                self.latest_prediction = received_command

                # Update the persistent game command.
                self.command = received_command

                # Signal that Training Mode has a new prediction to process.
                self.new_prediction_available = True

        except BlockingIOError:
            pass

    def get_command(self):
        """
        Used by Play Mode.

        Returns the persistent current command.

        Example:
            Receive RIGHT once.

            get_command() -> RIGHT
            get_command() -> RIGHT
            get_command() -> RIGHT
            ...
        """

        return self.command

    def get_new_prediction(self):
        """
        Used by Training Mode.

        Returns a prediction only once per newly received UDP packet.

        Example:
            Receive RIGHT packet.

            get_new_prediction() -> RIGHT
            get_new_prediction() -> None
            get_new_prediction() -> None

            Receive another RIGHT packet.

            get_new_prediction() -> RIGHT
        """

        if not self.new_prediction_available:
            return None

        self.new_prediction_available = False

        return self.latest_prediction
    
    def reset(self):
        self.command = Command.NONE
        self.latest_prediction = Command.NONE
        self.confidence = 0.0
        self.new_prediction_available = False

        # Clear old UDP packets still waiting in the socket
        try:
            while True:
                self.sock.recvfrom(1024)
        except BlockingIOError:
            pass
