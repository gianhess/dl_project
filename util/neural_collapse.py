"""
Neural collapse metrics

This file contains functions computing the metrics (NC1-NC4) introduced in Papyan et al. (2020).
The code was adapted from Zhu et al. (2021): https://github.com/tding1/Neural-Collapse
"""
import torch


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

    mu_G = 0
    mu_c_dict = dict()

    samples_per_class = [max(1, sum(targets == i).cpu().item()) for i in range(num_classes)]

    features = model.embed(inputs)

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

    mu_G = mu_G / len(targets)
    for i in range(num_classes):
        mu_c_dict[i] = mu_c_dict[i] / samples_per_class[i]

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
