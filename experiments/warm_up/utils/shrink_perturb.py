from utils import get_cifar10_loaders, seed_everything, train_and_test
from utils.resnet import resnet18
from tqdm import trange
import torch
import pickle
from utils.neural_collapse import NC
import numpy as np
import copy
import wandb


def SP_NC_PL(seed: int,
            batch_size= 128,
            dir: str = 'results',
            shrink: float = 0.6,
            perturb: float = 0.01,
            max_epoch_full=100,
            stop_acc=0.99,
            max_epochs_warm_up = 350):
    '''
    : param seed: int random seed
    : param batch_size: int
    : param dir: directory to save results
    : param shrink: float
    : param perturb: float
    : param max_epoch_full: int maximum number of epochs to train the model on the full dataset
    : param stop_acc: float accuracy on the training set to stop training
    : param max_epochs_warm_up: int maximum number of epochs to train the model on the half dataset
    : return: None

    Measures the Neural Collapse and the Performance Loss of the model trained on the full dataset after Shrink and Perturb.
    Saves the results in a pickle file.
    '''

    seed_path = f'{dir}/seed{seed}'
    checkpoints_path = f'{seed_path}/checkpoints'
    # load results
    with open(f'{seed_path}/results.pkl', 'rb') as f:
        results = pickle.load(f)

     # getting the loaders
    loaders_half = get_cifar10_loaders(0.5, seed = seed, batch_size= batch_size) # half dataset
    train_loader_half, test_loader_half = loaders_half['train_loader'], loaders_half['test_loader']
    loaders_full = get_cifar10_loaders(seed = seed, batch_size= batch_size) # complete dataset
    train_loader_full, test_loader_full = loaders_full['train_loader'], loaders_full['test_loader']


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr = 0.001
    seed_everything(seed)

    progress = trange(5, max_epochs_warm_up+1, 5)
    for epoch in progress:
        # load model
        model = resnet18(num_classes = 10)
        model.load_state_dict(torch.load(f'{checkpoints_path}/warm_up_{epoch}.pt'))
        dummy_model = resnet18(num_classes = 10)
        # shrink and perturb the model
        with torch.no_grad():
            for real_parameter, random_parameter in zip(model.parameters(), dummy_model.parameters()):
                real_parameter.mul_(shrink).add_(random_parameter, alpha=perturb)
        Optimizer = torch.optim.SGD
        # compute NC
        model = model.to(device) # you should change this into nc function
        progress.set_description(f"Epoch {epoch} (measuring NC)")
        nc = NC(model, train_loader_half)
        new_key_mapping = {key : f'{key}_SP' for key in nc.keys()}
        SP_NC = {new_key_mapping[old_key]: value for old_key, value in nc.items()}
        # compute PL
        progress.set_description(f"Epoch {epoch} (computing PL)")
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = Optimizer(model.parameters(), lr = lr)
        
        PL = train_and_test(model, train_loader_full, test_loader_full, criterion= criterion, optimizer= optimizer, seed = seed)
        new_key_mapping = {key : f'{key}_SP' for key in PL.keys()}
        SP_PL = {new_key_mapping[old_key]: value for old_key, value in PL.items()}

        pos_res = int((epoch / 5) -1)
        results[pos_res].update(SP_NC)
        results[pos_res].update(SP_PL)

        if epoch == 5: print(results[pos_res])
        
        with open(f'{seed_path}/results.pkl', 'wb') as f:
            pickle.dump(results, f)