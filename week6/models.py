import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from week5.data_augmentation import mixup_batch, mixup_criterion


def coral_loss(features_a, features_b):
    """
    CORAL loss between two feature groups.

    features_a: Tensor of shape [n_a, feature_dim]
    features_b: Tensor of shape [n_b, feature_dim]

    Matches second-order feature statistics by penalizing covariance differences.
    """
    if features_a.size(0) < 2 or features_b.size(0) < 2:
        return torch.tensor(0.0, device=features_a.device)

    d = features_a.size(1)

    a = features_a - features_a.mean(dim=0, keepdim=True)
    b = features_b - features_b.mean(dim=0, keepdim=True)

    cov_a = (a.T @ a) / (features_a.size(0) - 1)
    cov_b = (b.T @ b) / (features_b.size(0) - 1)

    return ((cov_a - cov_b) ** 2).sum() / (4 * d * d)


def multi_domain_coral_loss(features, domains):
    """
    Computes the average pairwise CORAL loss over all domains present in a batch.

    features: Tensor of shape [batch_size, feature_dim]
    domains: Tensor of shape [batch_size], e.g. subject IDs or session IDs
    """
    unique_domains = domains.unique()

    if unique_domains.numel() < 2:
        return torch.tensor(0.0, device=features.device)

    total_loss = torch.tensor(0.0, device=features.device)
    n_pairs = 0

    for i in range(len(unique_domains)):
        for j in range(i + 1, len(unique_domains)):
            f_i = features[domains == unique_domains[i]]
            f_j = features[domains == unique_domains[j]]

            if f_i.size(0) >= 2 and f_j.size(0) >= 2:
                total_loss = total_loss + coral_loss(f_i, f_j)
                n_pairs += 1

    if n_pairs == 0:
        return torch.tensor(0.0, device=features.device)

    return total_loss / n_pairs


class EEGNet(nn.Module):
    """
    Native PyTorch Implementation of EEGNet (Lawhern et al., 2018).
    Compact CNN for BCI robust to spatial non-stationarity.
    """
    def __init__(self, nb_classes=4, Chans=22, Samples=1000,
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

        # CHANGED: Split flattening and linear classification so latent features
        # can be returned cleanly for CORAL regularization.
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(n_flat, nb_classes)

    def forward(self, x, return_features=False):
        x = self.block1(x)
        x = self.block2(x)
        features = self.flatten(x)
        logits = self.classifier(features)

        if return_features:
            return logits, features

        return logits


class PyTorchClassifier:
    """
    Scikit-Learn style wrapper for PyTorch Models.
    Allows for training (.fit), evaluating (.predict_proba), and loading pre-trained weights.

    CORAL-DG usage:
        clf = PyTorchClassifier(EEGNet, coral=True, lambda_coral=0.05, batch_size=64)
        clf.fit(X_train, y_train, domains=train_subject_ids)
    """
    def __init__(self, model_class, epochs=50, batch_size=16, lr=1e-3,
                 mixup=False, coral=False, lambda_coral=0.05):
        self.model_class = model_class
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.mixup = mixup
        self.coral = coral
        self.lambda_coral = lambda_coral
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def fit(self, X, y, domains=None):
        # CORAL needs domain IDs, e.g. subject IDs or session IDs.
        if self.coral and domains is None:
            raise ValueError("CORAL-DG requires domains. Call fit(X, y, domains=subject_or_session_ids).")

        # Mixing samples also mixes labels; domain IDs become ambiguous.
        # Keep these methods separate for clean assignment comparisons.
        if self.mixup and self.coral:
            raise ValueError("mixup=True and coral=True are intentionally not combined here. Run them as separate methods.")

        # Convert numpy arrays to PyTorch tensors
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
        y_t = torch.tensor(y, dtype=torch.long).to(self.device)

        # Dynamically detect class count from the labels
        n_classes = len(np.unique(y))

        # Initialize model and optimizer
        self.model = self.model_class(nb_classes=n_classes, Chans=X.shape[1], Samples=X.shape[2]).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if self.coral:
            domains_t = torch.tensor(domains, dtype=torch.long).to(self.device)
            dataset = TensorDataset(X_t, y_t, domains_t)
        else:
            dataset = TensorDataset(X_t, y_t)

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()

        # --- Training Loop Branching ---
        if self.mixup:
            print("    [Mixup] Applying Feature-Space Mixup (alpha=0.2) during training.")
            for epoch in range(self.epochs):
                for batch_x, batch_y in loader:
                    optimizer.zero_grad()
                    batch_x, y_a, y_b, lam = mixup_batch(batch_x, batch_y, alpha=0.2)
                    outputs = self.model(batch_x)
                    loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
                    loss.backward()
                    optimizer.step()

        elif self.coral:
            print(f"    [CORAL-DG] Training with source-domain CORAL regularization (lambda={self.lambda_coral}).")
            for epoch in range(self.epochs):
                for batch_x, batch_y, batch_domains in loader:
                    optimizer.zero_grad()

                    outputs, features = self.model(batch_x, return_features=True)
                    loss_cls = criterion(outputs, batch_y)
                    loss_coral = multi_domain_coral_loss(features, batch_domains)
                    loss = loss_cls + self.lambda_coral * loss_coral

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

    def load_weights(self, path, X_sample, nb_classes=4):
        """
        Instantiates the architecture and loads pre-trained weights.
        """
        Chans, Samples = X_sample.shape[1], X_sample.shape[2]

        self.model = self.model_class(
            nb_classes=nb_classes,
            Chans=Chans,
            Samples=Samples
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(path, map_location=self.device)
        )

        self.model.eval()

    def save_weights(self, path):
        """
        Saves trained model weights to disk.
        """
        if self.model is None:
            raise RuntimeError("No trained model to save.")

        torch.save(self.model.state_dict(), path)

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
        import platform

        if self.device.type == "cuda":
            hardware_name = torch.cuda.get_device_name(self.device)
        else:
            processor = platform.processor()
            hardware_name = f"CPU ({processor})" if processor else "CPU"

        params = sum(p.numel() for p in self.model.parameters()) if self.model else "N/A"

        return {
            "device": hardware_name,
            "parameters": params,
            "batch_size": self.batch_size,
            "coral": self.coral,
            "lambda_coral": self.lambda_coral if self.coral else None
        }
