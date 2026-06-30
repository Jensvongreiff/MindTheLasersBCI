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

        self.command = Command.NONE
        self.confidence = 0.0

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((ip, port))
        self.sock.setblocking(False)

        print(f"Listening for UDP predictions on {ip}:{port}")

    def update(self):

        try:
            while True:
                data, _ = self.sock.recvfrom(1024)

                msg = json.loads(data.decode("utf-8"))

                pred = msg["prediction"]
                self.confidence = msg.get("confidence", 0.0)

                if pred == 0:
                    self.command = Command.REST
                elif pred == 1:
                    self.command = Command.LEFT
                elif pred == 2:
                    self.command = Command.RIGHT

        except BlockingIOError:
            pass

    def get_command(self):

        return self.command