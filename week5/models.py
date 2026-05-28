import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from week5.data_augmentation import mixup_batch, mixup_criterion

class EEGNet(nn.Module):
    """
    Native PyTorch Implementation of EEGNet (Lawhern et al., 2018).
    Compact CNN for BCI robust to spatial non-stationarity.
    """
    def __init__(self, nb_classes=2, Chans=22, Samples=1000, 
                 dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16):
        super(EEGNet, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False),
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
            
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, nb_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x


class PyTorchClassifier:
    """
    Scikit-Learn style wrapper for PyTorch Models.
    Allows for training (.fit), evaluating (.predict_proba), and loading pre-trained weights.
    """
    def __init__(self, model_class, epochs=50, batch_size=16, lr=1e-3, mixup=False):
        self.model_class = model_class
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.mixup = mixup # NEW: Store the mixup flag
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def fit(self, X, y):
        # Convert numpy arrays to PyTorch tensors
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
        y_t = torch.tensor(y, dtype=torch.long).to(self.device)
        
        # Initialize model and optimizer
        self.model = self.model_class(nb_classes=2, Chans=X.shape[1], Samples=X.shape[2]).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        
        # --- Training Loop Branching ---
        if self.mixup:
            print("    [Mixup] Applying Feature-Space Mixup (alpha=0.2) during training.")
            for epoch in range(self.epochs):
                for batch_x, batch_y in loader:
                    optimizer.zero_grad()
                    # Colleague's Mixup Logic
                    batch_x, y_a, y_b, lam = mixup_batch(batch_x, batch_y, alpha=0.2)
                    outputs = self.model(batch_x)
                    loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
                    loss.backward()
                    optimizer.step()
        else:
            # Standard Training Loop
            for epoch in range(self.epochs):
                for batch_x, batch_y in loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
        
        return self

    def load_weights(self, path, X_sample):
        """Instantiates the architecture and loads pre-trained weights from a .pt file."""
        Chans, Samples = X_sample.shape[1], X_sample.shape[2]
        self.model = self.model_class(Chans=Chans, Samples=Samples, **self.kwargs).to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.model.eval()

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("Model is uninitialized. Call .fit() or .load_weights() first.")
            
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_t)
            probs = torch.softmax(outputs, dim=1)
            
        return probs.cpu().numpy()

    def get_info(self):
        """Exposes the internal hardware and parameter settings for the Latency/Device Report."""
        import platform # Built-in python module to get CPU info
        
        # 1. Get exact hardware name
        if self.device.type == "cuda":
            # Grabs the actual string from your NVIDIA driver (e.g., "NVIDIA GeForce RTX 4090")
            hardware_name = torch.cuda.get_device_name(self.device)
        else:
            # If running on CPU, grab the processor name (e.g., "Intel64 Family 6...")
            processor = platform.processor()
            hardware_name = f"CPU ({processor})" if processor else "CPU"

        # 2. Get parameter count
        params = sum(p.numel() for p in self.model.parameters()) if self.model else "N/A"
        
        return {
            "device": hardware_name,
            "parameters": params,
            "batch_size": self.batch_size
        }