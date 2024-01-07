"""
Neural collapse metrics

This module contains functions computing the metrics (NC1-NC4) introduced in Papyan et al. (2020).
The code was adapted from Zhu et al. (2021): https://github.com/tding1/Neural-Collapse
"""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def _get_feature_means(model: nn.Module,
                       data_loader: DataLoader,
                       num_classes: int) -> tuple[torch.Tensor, dict[torch.Tensor]]:
    # returns global mean and dict of class means
    mu_G = 0
    mu_c_dict = dict()
    samples_per_class = np.zeros(num_classes)
    for inputs, targets in data_loader:
        features = model.embed(inputs)

        samples_per_class += [sum(targets == i).cpu().item() for i in range(num_classes)]

        # global mean
        mu_G = mu_G + torch.sum(features, dim=0)

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

    mu_G = mu_G / len(data_loader.dataset)
    for i in range(num_classes):
        if samples_per_class[i] > 0:
            mu_c_dict[i] = mu_c_dict[i] / samples_per_class[i]

    return mu_G, mu_c_dict


def _get_classifier_weights(model: torch.nn.Module) -> torch.Tensor:
    # returns weights of last linear layer

    linear_layers = [layer for layer in model.children() if isinstance(layer, nn.Linear)]
    return linear_layers[-1].state_dict()['weight']


def NC1(model: nn.Module,
        num_classes: int,
        inputs: torch.Tensor = None,
        targets: torch.Tensor = None,
        data_loader: DataLoader = None) -> float:
    """
    Compute NC1 (cross-example within-class variability).

    :param model: model with 'embed' function
    :param num_classes: number of distinct classes
    :param inputs: batch of data
    :param targets: ground truth labels
    :param data_loader: data loader (if provided, 'inputs' and 'targets' are ignored)
    :return: NC1
    """
    if data_loader is None:
        assert inputs is not None and targets is not None, "no data provided"
        data_loader = DataLoader(list(zip(inputs, targets)), batch_size=len(targets))

    # global mean and dict of class means
    mu_G, mu_c_dict = _get_feature_means(model=model, data_loader=data_loader, num_classes=num_classes)

    # within-class covariance
    Sigma_W = 0
    for inputs, targets in data_loader:
        features = model.embed(inputs)

        for b in range(len(targets)):
            y = targets[b].item()
            Sigma_W = Sigma_W + (features[b, :] - mu_c_dict[y]).unsqueeze(1) @ (features[b, :] - mu_c_dict[y]).unsqueeze(0)

    Sigma_W = Sigma_W / len(data_loader.dataset)

    # between-class covariance
    Sigma_B = 0
    for i in range(num_classes):
        Sigma_B = Sigma_B + (mu_c_dict[i] - mu_G).unsqueeze(1) @ (mu_c_dict[i] - mu_G).unsqueeze(0)

    Sigma_B = Sigma_B / num_classes

    return torch.trace(Sigma_W @ torch.linalg.pinv(Sigma_B)) / num_classes


def NC2(model: nn.Module) -> float:
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


def NC3(model: nn.Module,
        num_classes: int,
        inputs: torch.Tensor = None,
        targets: torch.Tensor = None,
        data_loader: DataLoader = None) -> float:
    """
    Compute NC3 (distance of learned features to dual classifier)

    :param model: model with 'embed' function
    :param num_classes: number of distinct classes
    :param inputs: batch of data
    :param targets: ground truth labels
    :param data_loader: data loader (if provided, 'inputs' and 'targets' are ignored)
    :return: NC3
    """
    if data_loader is None:
        assert inputs is not None and targets is not None, "no data provided"
        data_loader = DataLoader(list(zip(inputs, targets)), batch_size=len(targets))

    device = next(model.parameters()).device

    mu_G, mu_c_dict = _get_feature_means(model=model, data_loader=data_loader, num_classes=num_classes)
    W = _get_classifier_weights(model)

    K = num_classes
    H = torch.empty(mu_c_dict[0].shape[0], K)
    for i in range(K):
        H[:, i] = mu_c_dict[i] - mu_G

    WH = torch.mm(W, H.to(device))
    WH /= torch.norm(WH, p='fro')
    sub = 1 / pow(K - 1, 0.5) * (torch.eye(K) - 1 / K * torch.ones((K, K))).to(device)

    res = torch.norm(WH - sub, p='fro')
    return res.detach().cpu().numpy().item()


def NC4(model: nn.Module,
        num_classes: int,
        inputs: torch.Tensor = None,
        targets: torch.Tensor = None,
        data_loader: DataLoader= None) -> float:
    """
    Compute NC4 (proportion of classifications agreeing with nearest center classifier)

    :param model: model with 'embed' function
    :param num_classes: number of distinct classes
    :param inputs: batch of data
    :param targets: ground truth labels
    :param data_loader: data loader (if provided, 'inputs' and 'targets' are ignored)
    :return: NC4
    """
    if data_loader is None:
        assert inputs is not None and targets is not None, "no data provided"
        data_loader = DataLoader(list(zip(inputs, targets)), batch_size=len(targets))

    _, mu_c_dict = _get_feature_means(model=model, data_loader=data_loader, num_classes=num_classes)

    class_centers = [value for (_, value) in sorted(mu_c_dict.items())]

    nearest_centers, preds = [], []
    for inputs, _ in data_loader:
        features = model.embed(inputs)
        nearest_centers.extend([np.argmin(list(map(lambda x: torch.norm(x - ft), class_centers))) for ft in features])
        preds.extend(torch.argmax(model(inputs), dim=1).numpy())

    nearest_centers, preds = np.array(nearest_centers), np.array(preds)

    return (nearest_centers == preds).mean()
