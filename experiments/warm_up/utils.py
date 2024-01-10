import os
from functools import partial

import numpy as np
import torch
from torch.utils.data import random_split, ConcatDataset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt


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




def plot_pairwise(df, metric1, metric2, interp = 200):    
    x = df.index 
    y1 = df[metric1]
    y2 = df[metric2]

    fig, ax1 = plt.subplots()
    ax1.plot(x, y1, '-', color = 'red', label = metric1)
    ax1.set_xlabel('Epochs of Warm Up')
    ax1.set_ylabel(metric1, color='red')
    ax1.tick_params('y', colors='red')


    ax2 = ax1.twinx()
    ax2.plot(x, y2, '-', color = 'blue', label = metric2)
    ax2.set_ylabel(metric2, color='blue')
    ax2.tick_params('y', colors='blue')

    fig.tight_layout()  
    plt.title(f'{metric1} and {metric2} on CIFAR10')
    ax1.grid(True, linestyle='--', alpha=0.7)

    if interp: ax1.axvline(x=interp, color='grey', linestyle='--', linewidth=1, label='Interpolation')

    ax1.set_facecolor('#f0f0f0')  # Background color
    fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.85))  # Legend



    plt.show()



def train_step(model, train_loader, criterion, optimizer, device):
    accuracies = []
    model = model.to(device)
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        accuracies.append((y_pred.argmax(1) == y).float().mean().item())
    return np.mean(accuracies)


def train_and_test(model, train_loader, test_loader, criterion, optimizer, max_epochs = 100, stop_acc = 0.99, seed = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    converged = False
    model.train()
    for epoch in range(max_epochs):
        train_acc = train_step(model, train_loader, criterion, optimizer)
        if stop_acc is not None and train_acc > stop_acc:
            converged = True
            break
    if not converged: print('Convergence not reached, increase epochs')
    test_acc = eval_on_dataloader(model, test_loader)
    return {'test_acc': test_acc}


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