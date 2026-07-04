import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from abc import ABC, abstractmethod
from .signal import EEGWindow

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
    def __init__(self, nb_classes=3, Chans=22, Samples=250, dropoutRate=0.5, F1=8, D=2, F2=16):
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
class EEGNetBCIWrapper(BaseEndToEndModel):
    def __init__(self, weights_path: str, n_channels: int, n_samples: int, n_classes: int = 3):
        self.model = EEGNet(nb_classes=n_classes, Chans=n_channels, Samples=n_samples)
        self.weights_path = weights_path
        
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
        
        self.model.eval()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Standalone training execution."""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1) 
        y_t = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)

        self.model.train()
        for epoch in range(40):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                out = self.model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

        torch.save(self.model.state_dict(), self.weights_path)
        self.model.eval()

    def predict_proba(self, window: EEGWindow) -> dict:
        tensor_data = torch.tensor(window.data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            output = self.model(tensor_data)
            probs = torch.softmax(output, dim=1).numpy()[0]

        return {"left": float(probs[0]), "right": float(probs[1]), "rest": float(probs[2])}


# --- Traditional Classifier Wrapper ---
class LDAWrapper(BaseClassifier):
    def __init__(self, lda_model_path: str):
        """
        Loads the fitted scikit-learn LDA object. 
        Scikit-learn is self-contained via pickle.
        """
        with open(lda_model_path, 'rb') as f:
            self.lda = pickle.load(f)

    def predict_proba(self, features: np.ndarray) -> dict:
        # sklearn expects a 2D array: (n_samples, n_features)
        features_expanded = np.expand_dims(features, axis=0)
        
        probs = self.lda.predict_proba(features_expanded)[0]

        return {
            "left": float(probs[0]),
            "right": float(probs[1]),
            "rest": float(probs[2])
        }