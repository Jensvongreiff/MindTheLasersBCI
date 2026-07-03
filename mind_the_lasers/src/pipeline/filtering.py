from abc import ABC, abstractmethod
import numpy as np
from scipy.signal import butter, lfilter
from signal import EEGWindow

class BaseFilter(ABC):
    """Base class for all filtering steps."""
    @abstractmethod
    def process(self, window: EEGWindow) -> EEGWindow:
        pass

class IIRBandpassFilter(BaseFilter):
    """
    Example: 8-30Hz Bandpass filter for Motor Imagery.
    Uses lfilter (causal) instead of filtfilt (non-causal) for online safe processing.

    note: Look into maintaining filter states between windows to reduce edge artifacts.
    """
    def __init__(self, lowcut: float, highcut: float, fs: int, order: int = 4):
        self.fs = fs
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        # Calculate filter coefficients once during initialization
        self.b, self.a = butter(order, [low, high], btype='band')

    def process(self, window: EEGWindow) -> EEGWindow:
        # Apply the filter along the time axis (axis=1)
        filtered_data = lfilter(self.b, self.a, window.data, axis=1)
        
        # Return a new window object to prevent mutability bugs
        return EEGWindow(
            data=filtered_data,
            sampling_rate=window.sampling_rate,
            timestamp=window.timestamp
        )