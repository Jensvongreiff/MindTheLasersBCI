import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from abc import ABC, abstractmethod
from .signal import EEGWindow

CLASS_LABELS = {
    0: "left",
    1: "right",
    2: "rest",
}


def _probability_dict_from_classifier(classifier, probs: np.ndarray) -> dict:
    classes = getattr(classifier, "classes_", None)
    if classes is None:
        classes = np.arange(len(probs))

    result = {label: 0.0 for label in CLASS_LABELS.values()}

    for class_id, probability in zip(classes, probs):
        label = CLASS_LABELS.get(int(class_id))
        if label is not None:
            result[label] = float(probability)

    return result

# --- Base Classes ---
class BaseClassifier(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray): pass

    @abstractmethod
    def predict_proba(self, features: np.ndarray) -> dict: pass

class BaseEndToEndModel(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray): pass

    @abstractmethod
    def predict_proba(self, window: EEGWindow) -> dict: pass


# --- EEGNet Architecture (Standalone) ---
class EEGNet(nn.Module):
    def __init__(self, nb_classes=3, Chans=16, Samples=250, dropoutRate=0.5, F1=8, D=2, F2=16):
        super(EEGNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding='same', bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding='same', groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropoutRate)
        )
        with torch.no_grad():
            dummy_out = self.block2(self.block1(torch.zeros(1, 1, Chans, Samples)))
            n_flat = dummy_out.view(1, -1).size(1)

        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(n_flat, nb_classes)

    def forward(self, x):
        return self.classifier(self.flatten(self.block2(self.block1(x))))


# --- End-To-End Model Wrapper ---
class LDAWrapper(BaseClassifier):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.lda = None
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.lda = pickle.load(f)
            print(f"Loaded fitted LDA from {self.model_path}.")

    def fit(self, X: np.ndarray, y: np.ndarray):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        self.lda = LinearDiscriminantAnalysis()
        self.lda.fit(X, y)
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.lda, f)

    def predict_proba(self, features: np.ndarray) -> dict:
        if self.lda is None:
            raise RuntimeError(
                f"No fitted LDA model is available at {self.model_path}."
            )
        probs = self.lda.predict_proba(np.expand_dims(features, axis=0))[0]
        return _probability_dict_from_classifier(self.lda, probs)

# --- End-To-End Model Wrapper ---
class LDAWrapperTest(BaseClassifier):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.lda = None
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.lda = pickle.load(f)

    @staticmethod
    def make_classifier():
        """
        Classifier used consistently for all feature methods.

        Pipeline:
            StandardScaler
            → shrinkage LDA
        """
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        
        clf = make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(
                solver="lsqr",
                shrinkage="auto",
            )
        )
        return clf

    def fit(self, X: np.ndarray, y: np.ndarray):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        self.lda = self.make_classifier()
        self.lda.fit(X, y)
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.lda, f)

    def predict_proba(self, features: np.ndarray) -> dict:
        if self.lda is None:
            raise RuntimeError(
                f"No fitted LDA model is available at {self.model_path}."
            )
        probs = self.lda.predict_proba(features.reshape(1, -1))[0]
        return _probability_dict_from_classifier(self.lda, probs)


class EEGNetBCIWrapper(BaseEndToEndModel):
    def __init__(self, weights_path: str, n_classes: int = 3):
        self.model = None
        self.weights_path = weights_path
        self.n_classes = n_classes

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Executes offline stochastic gradient descent to formulate subject-specific weights."""
        # Dynamically infer dimensions from training data
        n_epochs, n_channels, n_samples = X.shape
        self.model = EEGNet(nb_classes=self.n_classes, Chans=n_channels, Samples=n_samples)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1) 
        y_t = torch.tensor(y, dtype=torch.long)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=16, shuffle=True)

        self.model.train()
        for epoch in range(50):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(batch_x), batch_y)
                loss.backward()
                optimizer.step()

        os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.weights_path)
        self.model.eval()

    def predict_proba(self, window: EEGWindow) -> dict:

        if self.model is None:
            n_channels = window.data.shape[0]
            n_samples = window.data.shape[1]

            self.model = EEGNet(nb_classes=self.n_classes, Chans=n_channels, Samples=n_samples)

            if not os.path.exists(self.weights_path):
                raise FileNotFoundError(
                    "No EEGNet weights found. Run the Mind the Lasers "
                    "calibration/evaluation script first or pass the matching "
                    f"suffix. Expected: {self.weights_path}"
                )

            self.model.load_state_dict(torch.load(self.weights_path, map_location='cpu', weights_only=True))
            
            self.model.eval()
            
        tensor_data = torch.tensor(window.data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor_data), dim=1).numpy()[0]
        return {
            CLASS_LABELS[index]: float(probability)
            for index, probability in enumerate(probs)
            if index in CLASS_LABELS
        }
