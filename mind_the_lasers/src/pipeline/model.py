from abc import ABC, abstractmethod
import numpy as np
from signal import EEGWindow

# --- Base Classes ---
class BaseFeatureExtractor(ABC):
    @abstractmethod
    def extract(self, window: EEGWindow) -> np.ndarray:
        """Takes an EEG window and returns a 1D feature vector."""
        pass

class BaseClassifier(ABC):
    @abstractmethod
    def predict_proba(self, features: np.ndarray) -> dict:
        """Takes a feature vector and returns class probabilities."""
        pass

class BaseEndToEndModel(ABC):
    @abstractmethod
    def predict_proba(self, window: EEGWindow) -> dict:
        """Takes an EEG window directly and returns class probabilities."""
        pass

# --- Example Implementations ---
class CSPFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, pre_trained_csp):
        # Expects an already fitted mne.decoding.CSP object
        self.csp = pre_trained_csp

    def extract(self, window: EEGWindow) -> np.ndarray:
        # CSP expects shape (n_epochs, n_channels, n_times)
        # We add a dummy epoch dimension for online single-window processing
        data_expanded = np.expand_dims(window.data, axis=0)
        features = self.csp.transform(data_expanded)
        return features[0] # Return the 1D vector

class LDAClassifier(BaseClassifier):
    def __init__(self, pre_trained_lda):
        # Expects an already fitted sklearn.discriminant_analysis.LinearDiscriminantAnalysis
        self.lda = pre_trained_lda

    def predict_proba(self, features: np.ndarray) -> dict:
        features_expanded = np.expand_dims(features, axis=0)
        probs = self.lda.predict_proba(features_expanded)[0]
        
        # Return structured probabilities for the game controller
        return {
            "left": probs[0],
            "right": probs[1],
            "rest": probs[2]
        }

class EEGNetWrapper(BaseEndToEndModel):
    def __init__(self, pytorch_model):
        self.model = pytorch_model
        self.model.eval() # Ensure it's in evaluation mode for online inference

    def predict_proba(self, window: EEGWindow) -> dict:
        import torch # Imported here to keep base dependencies light
        
        # Prepare tensor (batch, channels, samples)
        tensor_data = torch.tensor(window.data, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(tensor_data)
            probs = torch.softmax(output, dim=1).numpy()[0]
            
        return {
            "left": probs[0],
            "right": probs[1],
            "rest": probs[2]
        }