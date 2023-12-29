"""
Neural collapse metrics

This module contains functions computing the metrics (NC1-NC4) introduced in Papyan et al. (2020).
The code was adapted from Zhu et al. (2021): https://github.com/tding1/Neural-Collapse
"""
import numpy as np
import torch
from torch import nn


def _get_feature_means(features: torch.Tensor, targets: torch.Tensor, num_classes: int) -> tuple[
    torch.Tensor, dict[torch.Tensor]]:
    # returns global mean and dict of class means

    mu_c_dict = dict()

    samples_per_class = [max(1, sum(targets == i).cpu().item()) for i in range(num_classes)]

    # global mean
    mu_G = torch.sum(features, dim=0)

    # class means
    for b in range(len(targets)):
        y = targets[b].item()
        if y not in mu_c_dict:
            mu_c_dict[y] = features[b, :]
        else:
            mu_c_dict[y] = mu_c_dict[y] + features[b, :]

    # in case some target was not present in batch assign global mean
    # TODO: check if that makes sense, or if we should assign e.g. zero vector
    for y in range(num_classes):
        if y not in mu_c_dict:
            mu_c_dict[y] = mu_G

    mu_G = mu_G / len(targets)
    for i in range(num_classes):
        mu_c_dict[i] = mu_c_dict[i] / samples_per_class[i]

    return mu_G, mu_c_dict


def _get_classifier_weights(model: torch.nn.Module) -> torch.Tensor:
    # returns weights of last linear layer

    linear_layers = [layer for layer in model.children() if isinstance(layer, nn.Linear)]
    return linear_layers[-1].state_dict()['weight']


def NC1(model: torch.nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int) -> float:
    """
    Compute NC1 (cross-example within-class variability).

    :param model: model with 'embed' function
    :param inputs: batch of data
    :param targets: ground truth labels
    :param num_classes: number of distinct classes
    :return: NC1
    """

    features = model.embed(inputs)
    mu_G, mu_c_dict = _get_feature_means(features, targets, num_classes)

    # within-class covariance
    Sigma_W = 0
    for b in range(len(targets)):
        y = targets[b].item()
        Sigma_W = Sigma_W + (features[b, :] - mu_c_dict[y]).unsqueeze(1) @ (features[b, :] - mu_c_dict[y]).unsqueeze(0)
    Sigma_W = Sigma_W / len(targets)

    # between-class covariance
    Sigma_B = 0
    K = len(mu_c_dict)
    for i in range(K):
        Sigma_B = Sigma_B + (mu_c_dict[i] - mu_G).unsqueeze(1) @ (mu_c_dict[i] - mu_G).unsqueeze(0)
    Sigma_B = Sigma_B / K

    return torch.trace(Sigma_W @ torch.linalg.pinv(Sigma_B)) / len(mu_c_dict)


def NC2(model: torch.nn.Module) -> float:
    """
    Compute NC2 (distance of last-layer classifier to a Simplex ETF).

    :param model: model with fully connected last layer
    :return: NC2
    """

    device = next(model.parameters()).device

    W = _get_classifier_weights(model)

    # compute ETF metric (see Zhu et al. (2021))
    K = W.shape[0]
    WWT = torch.mm(W, W.T)
    WWT /= torch.norm(WWT, p='fro')

    sub = (torch.eye(K) - 1 / K * torch.ones((K, K))).to(device) / pow(K - 1, 0.5)

    return torch.norm(WWT - sub, p='fro').item()


def NC3(model: torch.nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int) -> float:
    """
    Compute NC3 (distance of learned features to dual classifier)

    :param model: model with 'embed' function
    :param inputs: batch of data
    :param targets: ground truth labels
    :param num_classes: number of distinct classes
    :return: NC3
    """

    device = next(model.parameters()).device

    features = model.embed(inputs)
    mu_G, mu_c_dict = _get_feature_means(features, targets, num_classes)
    W = _get_classifier_weights(model)

    K = len(mu_c_dict)
    H = torch.empty(mu_c_dict[0].shape[0], K)
    for i in range(K):
        H[:, i] = mu_c_dict[i] - mu_G

    WH = torch.mm(W, H.to(device))
    WH /= torch.norm(WH, p='fro')
    sub = 1 / pow(K - 1, 0.5) * (torch.eye(K) - 1 / K * torch.ones((K, K))).to(device)

    res = torch.norm(WH - sub, p='fro')
    return res.detach().cpu().numpy().item()


def NC4(model: torch.nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int) -> float:
    """
    Compute NC4 (proportion of classifications agreeing with nearest center classifier)

    :param model: model with 'embed' function
    :param inputs: batch of data
    :param targets: ground truth labels
    :param num_classes: number of distinct classes
    :return: NC4
    """

    features = model.embed(inputs)
    _, mu_c_dict = _get_feature_means(features, targets, num_classes)

    class_centers = [value for (_, value) in sorted(mu_c_dict.items())]

    nearest_centers = [np.argmin(list(map(lambda x: torch.norm(x - feature), class_centers))) for feature in features]

    preds = torch.argmax(model(inputs), dim=1)

    return (nearest_centers == preds.numpy()).sum() / len(inputs)
