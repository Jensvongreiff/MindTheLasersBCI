import torch
import torch.nn as nn


def reset_bn_stats(model: nn.Module):
    """Reset BatchNorm running statistics."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.reset_running_stats()


def set_bn_momentum(model: nn.Module, momentum: float):
    """Set BatchNorm momentum for online adaptation."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = momentum


def adabn_adapt_batch(model: nn.Module, x_batch: torch.Tensor):
    """
    AdaBN adaptation step.

    Updates only BatchNorm running_mean/running_var using target batch.
    No gradients, no optimizer, no weight updates.
    """
    model.train()

    with torch.no_grad():
        _ = model(x_batch)

    model.eval()


def predict_batch(model: nn.Module, x_batch: torch.Tensor):
    """Predict probabilities for one batch."""
    model.eval()

    with torch.no_grad():
        logits = model(x_batch)
        probs = torch.softmax(logits, dim=1)

    return probs


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax predictions."""
    probs = torch.softmax(logits, dim=1)
    log_probs = torch.log_softmax(logits, dim=1)
    return -(probs * log_probs).sum(dim=1)


def configure_tent(model: nn.Module):
    """
    Configure model for TENT.

    Freeze all parameters except BatchNorm affine parameters:
    gamma/weight and beta/bias.
    """
    model.train()
    model.requires_grad_(False)

    params = []

    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.requires_grad_(True)

            if m.weight is not None:
                m.weight.requires_grad_(True)
                params.append(m.weight)

            if m.bias is not None:
                m.bias.requires_grad_(True)
                params.append(m.bias)

    return params


def tent_adapt_and_predict(
    model: nn.Module,
    x_batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
):
    """
    One TENT online adaptation step.

    Uses target batch only.
    Minimizes entropy.
    Updates only BN affine params.
    Returns adapted probabilities.
    """
    model.train()

    optimizer.zero_grad()

    logits = model(x_batch)
    loss = softmax_entropy(logits).mean()

    loss.backward()
    optimizer.step()

    model.eval()

    with torch.no_grad():
        logits = model(x_batch)
        probs = torch.softmax(logits, dim=1)

    return probs, loss.item()