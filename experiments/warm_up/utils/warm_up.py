import torch
from torch import nn
from torch.nn import functional as F
import os
from utils.resnet import resnet18
from tqdm import trange
from utils import get_cifar10_loaders, train_step, seed_everything, train_and_test, eval_on_dataloader
from utils.neural_collapse import NC, NC1
import numpy as np
import copy
import pickle
import wandb




def warm_up (seed : int,
            batch_size : int = 128,
            max_epoch_warm_up : int = 350,
            log_interval : int = 5,
            Optimizer : torch.optim.Optimizer = torch.optim.SGD,
            Criterion : torch.nn.Module = torch.nn.CrossEntropyLoss(),
            lr : float = 0.001,
            dir : str = 'results'):
        
    '''
    : param seed: int random seed
    : param batch_size: int
    : param max_epoch_warm_up: int
    : param log_interval: int 
    : param Optimizer: torch.optim.Optimizer
    : param lr: float
    : param dir: directory to save results
    : return: None

    Trains the model on half of the dataset for max_epoch_warm_up epochs.
    Saves the model every log_interval epochs.
    '''

    seed_path = f'{dir}/seed{seed}'
    checkpoints_path = f'{seed_path}/checkpoints'
    # creating directories
    os.makedirs(seed_path, exist_ok = True)
    os.makedirs(checkpoints_path, exist_ok = True)

    # setting seeds
    seed_everything(seed)

    # getting data loaders
    loaders_half = get_cifar10_loaders(0.5, seed = seed, batch_size= batch_size) # half dataset
    train_loader_half, test_loader_half = loaders_half['train_loader'], loaders_half['test_loader']

    # setting up model training
    model = resnet18(num_classes = 10)
    criterion = Criterion()
    optimizer = Optimizer(model.parameters(), lr = lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    progress = trange(max_epoch_warm_up, position=0)
    for epoch in progress:
        # training step on half dataset
        progress.set_description(f"Epoch {epoch+1} of {max_epoch_warm_up} (warm up)")
        train_step(model, train_loader_half, criterion, optimizer)
        if (epoch+1) % log_interval == 0:
            # saving the model
            progress.set_description(f"Epoch {epoch+1} of {max_epoch_warm_up} (saving the model)")
            torch.save(model.state_dict(), f'{checkpoints_path}/warm_up_{epoch+1}.pt')


class Regularized_Loss(nn.Module):
    def __init__(self, weight_factor=1.0, clip=200):
        super(Regularized_Loss, self).__init__()
        self.weight_factor = weight_factor
        self.clip = clip

    def forward(self, predicted, target, input, model):
        # Compute the cross-entropy loss
        ce_loss = F.cross_entropy(predicted, target)

        # NC1 regularization
        nc1 = NC1(model,10, input, target)
        if nc1 > self.clip:
            nc1 = self.clip

        # Combine the cross-entropy loss with the nc1 regularizer
        total_loss = ce_loss - self.weight_factor * nc1

        return total_loss
    
    

def warm_up_reg (seed : int,
            batch_size : int = 128,
            max_epoch_warm_up : int = 350,
            log_interval : int = 5,
            max_epoch_full: int = 100,
            Optimizer : torch.optim.Optimizer = torch.optim.SGD,
            Criterion : torch.nn.Module = Regularized_Loss,
            lr : float = 0.001,
            weight_factor: float = 0.075,
            clip: float = 200,
            dir : str = 'results_reg'):
        
    '''
    : param seed: int random seed
    : param batch_size: int
    : param max_epoch_warm_up: int
    : param log_interval: int 
    : param Optimizer: torch.optim.Optimizer
    : param lr: float
    : param dir: directory to save results
    : return: None

    Trains the model on half of the dataset for max_epoch_warm_up epochs.
    Saves the model every log_interval epochs.
    '''

    os.makedirs(dir, exist_ok = True)
    seed_path = f'{dir}/seed{seed}_{weight_factor}'
    checkpoints_path = f'{seed_path}/checkpoints'
    # creating directories
    os.makedirs(seed_path, exist_ok = True)
    os.makedirs(checkpoints_path, exist_ok = True)

    # setting seeds
    seed_everything(seed)

    # getting data loaders
    loaders_half = get_cifar10_loaders(0.5, seed = seed, batch_size= batch_size) # half dataset
    train_loader_half, test_loader_half = loaders_half['train_loader'], loaders_half['test_loader']
    loader_full = get_cifar10_loaders(seed = seed, batch_size= batch_size) # full dataset
    train_loader_full, test_loader_full = loader_full['train_loader'], loader_full['test_loader']

    # setting up model training on half dataset
    model = resnet18(num_classes = 10)
    criterion = Criterion(weight_factor = weight_factor, clip = clip)
    optimizer = Optimizer(model.parameters(), lr = lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    progress = trange(max_epoch_warm_up, position=0)
    results = []
    wandb.init(project="warm-up", name = f'regularized_loss_{weight_factor}_{seed}_test')
    for epoch in progress:
        # training step on half dataset
        progress.set_description(f"Epoch {epoch+1} of {max_epoch_warm_up} (warm up)")
        train_acc_half = train_step(model, train_loader_half, criterion, optimizer, reg=True)
        if (epoch+1) % log_interval == 0:
            results.append({'train_acc_half': train_acc_half})
            # saving the model
            test_acc_half = eval_on_dataloader(model, test_loader_half)
            results[-1].update({'test_acc_half': test_acc_half})
            progress.set_description(f"Epoch {epoch+1} of {max_epoch_warm_up} (saving the model)")
            torch.save(model.state_dict(), f'{checkpoints_path}/warm_up_{epoch+1}.pt')
            # measuring the neural collapse
            progress.set_description(f"Epoch {epoch+1} of {max_epoch_warm_up} (measuring NC)")
            nc = NC(model, train_loader_half)
            results[-1].update(nc)
            # measuring PL
            model_copy = copy.deepcopy(model)
            # compute PL
            progress.set_description(f"Epoch {epoch+1} (computing PL)")
            criterion2 = torch.nn.CrossEntropyLoss()
            optimizer2 = Optimizer(model_copy.parameters(), lr = lr)
            PL = train_and_test(model_copy, train_loader_full, test_loader_full, criterion= criterion2, optimizer= optimizer2, seed = seed, max_epochs = max_epoch_full)
            results[-1].update(PL)
            if epoch == 5: print(results[-1])
            wandb.log(results[-1])
            # saving results
            with open(f'{seed_path}/results.pkl', 'wb') as f:
                pickle.dump(results, f)
    wandb.finish()
    

def warm_up_weight_decay(seed : int,
            batch_size : int = 128,
            max_epoch_warm_up : int = 350,
            log_interval : int = 5,
            max_epoch_full: int = 100,
            Optimizer : torch.optim.Optimizer = torch.optim.SGD,
            Criterion : torch.nn.Module = torch.nn.CrossEntropyLoss,
            lr : float = 0.001,
            weight_decay: float = 0.075,
            dir : str = 'results_reg_weight_decay'):
        
    '''
    : param seed: int random seed
    : param batch_size: int
    : param max_epoch_warm_up: int
    : param log_interval: int 
    : param Optimizer: torch.optim.Optimizer
    : param lr: float
    : param dir: directory to save results
    : return: None

    Trains the model on half of the dataset for max_epoch_warm_up epochs.
    Saves the model every log_interval epochs.
    '''

    os.makedirs(dir, exist_ok = True)
    seed_path = f'{dir}/seed{seed}_{weight_decay}'
    checkpoints_path = f'{seed_path}/checkpoints'
    # creating directories
    os.makedirs(seed_path, exist_ok = True)
    os.makedirs(checkpoints_path, exist_ok = True)

    # setting seeds
    seed_everything(seed)

    # getting data loaders
    loaders_half = get_cifar10_loaders(0.5, seed = seed, batch_size= batch_size) # half dataset
    train_loader_half, test_loader_half = loaders_half['train_loader'], loaders_half['test_loader']
    loader_full = get_cifar10_loaders(seed = seed, batch_size= batch_size) # full dataset
    train_loader_full, test_loader_full = loader_full['train_loader'], loader_full['test_loader']

    # setting up model training on half dataset
    model = resnet18(num_classes = 10)
    criterion = Criterion()
    optimizer = Optimizer(model.parameters(), lr = lr, weight_decay = weight_decay)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    progress = trange(max_epoch_warm_up, position=0)
    results = []
    wandb.init(project="warm-up", name = f'regularized_L2_{weight_decay}_{seed}')
    for epoch in progress:
        # training step on half dataset
        progress.set_description(f"Epoch {epoch+1} of {max_epoch_warm_up} (warm up)")
        train_acc_half = train_step(model, train_loader_half, criterion, optimizer)
        if (epoch+1) % log_interval == 0:
            results.append({'train_acc_half': train_acc_half})
            # saving the model
            test_acc_half = eval_on_dataloader(model, test_loader_half)
            results[-1].update({'test_acc_half': test_acc_half})
            progress.set_description(f"Epoch {epoch+1} of {max_epoch_warm_up} (saving the model)")
            #torch.save(model.state_dict(), f'{checkpoints_path}/warm_up_{epoch+1}.pt')
            # measuring the neural collapse
            progress.set_description(f"Epoch {epoch+1} of {max_epoch_warm_up} (measuring NC)")
            nc = NC(model, train_loader_half)
            results[-1].update(nc)
            # measuring PL
            model_copy = copy.deepcopy(model)
            # compute PL
            progress.set_description(f"Epoch {epoch+1} (computing PL)")
            criterion2 = torch.nn.CrossEntropyLoss()
            optimizer2 = Optimizer(model_copy.parameters(), lr = lr)
            PL = train_and_test(model_copy, train_loader_full, test_loader_full, criterion= criterion2, optimizer= optimizer2, seed = seed, max_epochs = max_epoch_full)
            results[-1].update(PL)
            if epoch == 5: print(results[-1])
            wandb.log(results[-1])
            # saving results
            with open(f'{seed_path}/results.pkl', 'wb') as f:
                pickle.dump(results, f)
    wandb.finish()