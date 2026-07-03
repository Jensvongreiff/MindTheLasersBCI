from abc import ABC, abstractmethod
import numpy as np
from signal import EEGWindow

# --- Base Classes ---
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