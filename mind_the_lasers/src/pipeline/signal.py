import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class EEGWindow:
    """
    Standardized container for a window of EEG data.
    This ensures all pipeline steps receive the data and sampling rate consistently.
    """
    data: np.ndarray  # Shape: (n_channels, n_samples)
    sampling_rate: int
    timestamp: float  # Useful for the game controller to align inputs
    
    # Optional metadata that might be added during the pipeline
    is_artifact_free: Optional[bool] = None