from tqdm import trange
import torch
import pickle
from utils import get_cifar10_loaders, seed_everything, train_and_test
from utils.resnet import resnet18



def measure_PL(seed: int,
               batch_size: int = 128,
               max_epoch_full: int = 100,
               stop_acc: float = 0.99,
               dir : str = 'results'):
    '''
    : param seed: int random seed
    : param batch_size: int
    : param max_epoch_full: int maximum number of epochs to train the model on the full dataset
    : param stop_acc: float accuracy on the training set to stop training
    : param dir: directory to save results
    : return: None

    Measures the Performance Loss of the model trained on the full dataset.
    Saves the results in a pickle file.
    '''

    seed_path = f'{dir}/seed{seed}'
    checkpoints_path = f'{seed_path}/checkpoints'

    # load results
    with open(f'{seed_path}/results.pkl', 'rb') as f:
        results = pickle.load(f)

    # setting seeds
    seed_everything(seed)

    # getting data loaders
    loaders_full = get_cifar10_loaders(seed = seed, batch_size= batch_size) # full dataset
    train_loader_full, test_loader_full = loaders_full['train_loader'], loaders_full['test_loader']

    lr = 0.001
    log_interval = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_epoch_warm_up = len(results) * log_interval

    for epoch in trange(log_interval, max_epoch_warm_up + 1, log_interval):
        pos_res = int((epoch / 5) -1)

        # setting up model
        model = resnet18(num_classes = 10)

        # loading the model
        model.load_state_dict(torch.load(f'{checkpoints_path}/warm_up_{epoch}.pt'))
        model = model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr = lr)
        
        # measuring the performance loss
        results[pos_res].update(train_and_test(model, train_loader_full, test_loader_full, criterion= criterion, optimizer= optimizer, seed = seed))

        if epoch == 5: print(results[pos_res])

        # saving results
        with open(f'{seed_path}/results.pkl', 'wb') as f:
            pickle.dump(results, f)