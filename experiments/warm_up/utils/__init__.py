import os
from functools import partial

import numpy as np
import torch
from torch.utils.data import random_split
from torchvision import transforms, datasets
import time



def get_accuracy(logit, true_y):
    pred_y = torch.argmax(logit, dim=1)
    return (pred_y == true_y).float().mean()

def eval_on_dataloader(model,dataloader):
    device = next(model.parameters()).device
    accuracies = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (data_x, data_y) in enumerate(dataloader):
            data_x = data_x.to(device)
            data_y = data_y.to(device)

            model_y = model(data_x)
            batch_accuracy = get_accuracy(model_y, data_y)
            accuracies.append(batch_accuracy.item())

        accuracy = np.mean(accuracies)
    return accuracy



def get_cifar10_loaders(use_half_train=False, data_aug=False, batch_size=128, dataset_portion=None, drop_last=False, seed = 42):
    normalize_transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    torch.manual_seed(seed)

    if not data_aug:
        train_transform = transforms.Compose([transforms.ToTensor(), normalize_transform])
    else:
        train_transform = transforms.Compose([transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.25,.25,.25),
                                            transforms.RandomRotation(2),
                                            normalize_transform])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize_transform])
    original_train_dataset = datasets.CIFAR10(root=os.path.join('data', 'cifar10_data'),
                                              train=True, transform=train_transform, download=True)
    original_test_dataset = datasets.CIFAR10(root=os.path.join('data', 'cifar10_data'),
                                             train=False, transform=test_transform, download=True)

    if use_half_train:
        dataset_portion = 0.5
    if dataset_portion:
        dataset_size = len(original_train_dataset)
        split = int(np.floor((1 - dataset_portion) * dataset_size))
        original_train_dataset, _ = random_split(original_train_dataset, [dataset_size - split, split], )

    loader_args = {
        "batch_size": batch_size,
    }

    train_loader = torch.utils.data.DataLoader(
        dataset=original_train_dataset,
        shuffle=True,
        drop_last=drop_last,
        **loader_args)

    test_loader = torch.utils.data.DataLoader(
        dataset=original_test_dataset,
        shuffle=False,
        **loader_args)

    return {"train_loader": train_loader,
            "test_loader": test_loader}



def train_step(model: torch.nn.Module, train_loader, criterion, optimizer, reg = False):
    '''
    : param model: torch.nn.Module
    : param train_loader: torch.utils.data.DataLoader
    : param criterion: torch.nn.Module
    : param optimizer: torch.optim.Optimizer
    : return: float

    Trains the model for one epoch on the training set.
    Returns the average accuracy of the epoch.
    '''

    device = next(model.parameters()).device
    y_preds = torch.tensor([]).to(device)
    y_trues = torch.tensor([]).to(device)
    model.train()
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        if reg:  loss = criterion(y_pred, y, x, model)
        else: loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        y_preds = torch.cat((y_preds, y_pred), 0)
        y_trues = torch.cat((y_trues, y), 0)
    return (y_preds.argmax(1) == y_trues).float().mean().item()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_and_test(model, train_loader, test_loader, criterion, optimizer, max_epochs = 100, stop_acc = 0.99, seed = 42):
    seed_everything(seed)
    converged = False
    model.train()
    start_time = time.time()
    for epoch in range(max_epochs):
        train_acc = train_step(model, train_loader, criterion, optimizer)
        if stop_acc is not None and train_acc > stop_acc:
            converged = True
            break
    if not converged: print('Convergence not reached, increase epochs')
    end_time = time.time()
    test_acc = eval_on_dataloader(model, test_loader)
    return {'test_acc': test_acc, 'train_time': end_time - start_time}