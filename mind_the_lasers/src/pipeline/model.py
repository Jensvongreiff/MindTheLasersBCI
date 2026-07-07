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

    def fit(self, X: np.ndarray, y: np.ndarray):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        self.lda = LinearDiscriminantAnalysis()
        self.lda.fit(X, y)
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.lda, f)

    def predict_proba(self, features: np.ndarray) -> dict:
        probs = self.lda.predict_proba(np.expand_dims(features, axis=0))[0]
        return {"left": float(probs[0]), "right": float(probs[1]), "rest": float(probs[2])}


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

            if os.path.exists(self.weights_path):
                self.model.load_state_dict(torch.load(self.weights_path, map_location='cpu', weights_only=True))
            else:
                print("Warning: No weights found. Running with untrained model.")
            
            self.model.eval()
            
        tensor_data = torch.tensor(window.data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor_data), dim=1).numpy()[0]
        return {"left": float(probs[0]), "right": float(probs[1]), "rest": float(probs[2])}