import numpy as np
import torch


def add_gaussian_noise(X, noise_std=0.05, rng=None):
    """
    Add Gaussian noise to EEG trials.

    X shape: (trials, channels, timepoints)
    """
    if rng is None:
        rng = np.random.default_rng()

    trial_std = np.std(X, axis=(1, 2), keepdims=True)
    noise = rng.normal(
        loc=0.0,
        scale=noise_std * trial_std,
        size=X.shape,
    )

    return X + noise


def amplitude_scaling(X, scale_std=0.1, rng=None):
    """
    Randomly scale EEG trial amplitudes.

    X shape: (trials, channels, timepoints)
    """
    if rng is None:
        rng = np.random.default_rng()

    scales = rng.normal(
        loc=1.0,
        scale=scale_std,
        size=(X.shape[0], 1, 1),
    )

    return X * scales


def augment_eeg_training_data(
    X,
    y,
    noise_std=0.05,
    scale_std=0.1,
    augment_factor=1,
    seed=None,
):
    """
    Creates augmented copies of the training set and appends them to the original data.

    X shape: (trials, channels, timepoints)
    y shape: (trials,)
    """
    rng = np.random.default_rng(seed)

    X_aug_list = [X]
    y_aug_list = [y]

    for _ in range(augment_factor):
        X_aug = amplitude_scaling(X, scale_std=scale_std, rng=rng)
        X_aug = add_gaussian_noise(X_aug, noise_std=noise_std, rng=rng)

        X_aug_list.append(X_aug)
        y_aug_list.append(y)

    X_out = np.concatenate(X_aug_list, axis=0)
    y_out = np.concatenate(y_aug_list, axis=0)

    indices = rng.permutation(len(X_out))
    return X_out[indices], y_out[indices]


def mixup_batch(batch_x, batch_y, alpha=0.2):
    """
    Applies Mixup to one mini-batch.

    batch_x shape: (batch, 1, channels, timepoints)
    batch_y shape: (batch,)
    """
    if alpha <= 0:
        return batch_x, batch_y, batch_y, 1.0

    lam = np.random.beta(alpha, alpha)

    batch_size = batch_x.size(0)
    index = torch.randperm(batch_size, device=batch_x.device)

    mixed_x = lam * batch_x + (1.0 - lam) * batch_x[index]

    y_a = batch_y
    y_b = batch_y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, outputs, y_a, y_b, lam):
    """
    Computes Mixup loss using normal CrossEntropyLoss.
    """
    return lam * criterion(outputs, y_a) + (1.0 - lam) * criterion(outputs, y_b)