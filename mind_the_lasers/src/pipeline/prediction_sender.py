import json
import socket


class PredictionSender:
    LABEL_TO_GAME_IDX = {
        "rest": 0,
        "left": 1,
        "right": 2,
    }

    def __init__(self, ip="127.0.0.1", port=5005):
        self.address = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, prediction, confidence=1.0, extra=None):
        """
        prediction can be either

            "left"
            "right"
            "rest"

        or

            0
            1
            2
        """

        if isinstance(prediction, str):
            label = prediction.lower()
            prediction = self.LABEL_TO_GAME_IDX[label]

        else:
            prediction = int(prediction)

            idx_to_label = {
                0: "rest",
                1: "left",
                2: "right",
            }

            label = idx_to_label[prediction]

        msg = {
            "prediction": prediction,
            "label": label,
            "confidence": float(confidence),
        }

        if extra is not None:
            msg["extra"] = extra

        self.sock.sendto(
            json.dumps(msg).encode("utf-8"),
            self.address,
        )