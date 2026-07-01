from abc import ABC, abstractmethod
import numpy as np
from signal import EEGWindow

class BaseArtifactRemoval(ABC):
    """Base class for artifact removal."""
    
    @abstractmethod
    def fit(self, offline_data: np.ndarray):
        """Called only during the calibration phase."""
        pass

    @abstractmethod
    def transform(self, window: EEGWindow) -> EEGWindow:
        """Called online during the game loop."""
        pass

class SpatialFilterICA(BaseArtifactRemoval):
    """
    Example: Applies a pre-calculated ICA unmixing matrix online.
    """
    def __init__(self):
        self.unmixing_matrix = None
        self.components_to_keep = None

    def fit(self, offline_data: np.ndarray, bad_components: list):
        # In reality, you'd use MNE's ICA here on the offline_data
        # self.unmixing_matrix = mne_ica.unmixing_matrix_
        # Here we mock it:
        n_channels = offline_data.shape[0]
        self.unmixing_matrix = np.eye(n_channels) # Mock matrix
        self.components_to_keep = [i for i in range(n_channels) if i not in bad_components]

    def transform(self, window: EEGWindow) -> EEGWindow:
        if self.unmixing_matrix is None:
            raise ValueError("ICA has not been fitted with offline calibration data yet.")
        
        # 1. Map to source space: Sources = Unmixing * Data
        sources = np.dot(self.unmixing_matrix, window.data)
        
        # 2. Zero out bad components
        # (Alternatively, only reconstruct using components_to_keep)
        clean_sources = sources[self.components_to_keep, :]
        mixing_matrix = np.linalg.pinv(self.unmixing_matrix)
        
        # 3. Map back to sensor space
        clean_data = np.dot(mixing_matrix[:, self.components_to_keep], clean_sources)
        
        return EEGWindow(
            data=clean_data,
            sampling_rate=window.sampling_rate,
            timestamp=window.timestamp,
            is_artifact_free=True
        )