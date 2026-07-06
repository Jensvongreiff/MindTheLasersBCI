#Logic for Daniels output command smoothing

import collections
from typing import Dict

class SmoothingController:
    """
    Stabilizes raw BCI probabilities into discrete game commands.
    Mitigates jerky movements using confidence thresholding and a rolling majority vote.
    """
    def __init__(self, window_size: int = 5, confidence_threshold: float = 0.60):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.buffer = collections.deque(maxlen=window_size)
        
        # Pre-fill buffer to prevent index errors or erratic initial state
        for _ in range(window_size):
            self.buffer.append("rest")

    def process(self, probabilities: Dict[str, float]) -> str:
        # Identify the class with the maximum probability
        max_class = max(probabilities, key=probabilities.get)
        max_prob = probabilities[max_class]

        # Apply confidence thresholding: fallback to 'rest' if uncertain
        if max_prob < self.confidence_threshold:
            prediction = "rest"
        else:
            prediction = max_class

        # Update rolling buffer
        self.buffer.append(prediction)

        # Compute majority vote
        vote_counts = collections.Counter(self.buffer)
        majority_class = vote_counts.most_common(1)[0][0]

        return majority_class