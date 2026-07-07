import socket
import json


class PredictionSender:
    """
    UDP sender for Mind the Lasers classifier predictions.

    Usage
    -----
    Create a sender once:

        sender = PredictionSender()

    Whenever the classifier produces a prediction:

        prediction = classifier.predict(features)
        confidence = classifier.predict_proba(features).max()

        sender.send(
            prediction=prediction,
            confidence=confidence,
        )

    The game will automatically receive the prediction over UDP and
    update its internal controller. No additional networking code is
    required in the classifier.


    Game UDP Interface
    ------------------
    Destination
        IP:   127.0.0.1
        Port: 5005

    Packet format
    -------------
    {
        "prediction": int,
        "label": str,
        "confidence": float
    }

    Prediction mapping
    ------------------
        0 = REST
        1 = LEFT
        2 = RIGHT

    Frequency
    ---------
    Send one packet for every decoded prediction.

    Typical update rate:
        5–10 Hz

    Confidence
    ----------
    Float in the range [0.0, 1.0].

    """

    def __init__(self,
                 ip="127.0.0.1",
                 port=5005):

        self.sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )

        self.address = (ip, port)

        self.labels = {
            0: "rest",
            1: "left",
            2: "right",
        }

    def send(self,
             prediction,
             confidence,
             extra=None):

        msg = {
            "prediction": prediction,
            "label": self.labels[prediction],
            "confidence": float(confidence),
        }

        if extra is not None:
            msg["extra"] = extra

        self.sock.sendto(
            json.dumps(msg).encode(),
            self.address,
        )