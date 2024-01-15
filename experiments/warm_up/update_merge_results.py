from utils import get_cifar10_loaders, eval_on_dataloader, seed_everything
from utils.resnet import resnet18
from tqdm import tqdm
import torch
import pickle
import pandas as pd
from utils.viz import get_results


max_epoch = 115


for seed in range(7, 12):
    seed_everything(seed)
    results_list = get_results(seed)[: max_epoch//5]
    checkpoints_path = f'results/seed{seed}/checkpoints/'
    final_path = f'results/seed{seed}/results_updated.pkl'

    loaders_half = get_cifar10_loaders(0.5, seed = seed)
    train_loader_half, test_loader_half = loaders_half['train_loader'], loaders_half['test_loader']
    index = range(5, (len(results_list)+1)* 5, 5)
    i = 0
    for epoch in tqdm(index):
        model = resnet18()
        checkpoint_path = f'{checkpoints_path}warm_up_{epoch}.pt'
        model.load_state_dict(torch.load(checkpoint_path))
        model.to('cuda')
        train_acc_half = eval_on_dataloader(model, train_loader_half)
        test_acc_half = eval_on_dataloader(model, test_loader_half)
        with torch.no_grad():
            for real_parameter, random_parameter in zip(model.parameters(), resnet18(num_classes = 10).to('cuda').parameters()):
                real_parameter.mul_(0.6).add_(random_parameter, alpha=0.01)
        train_acc_half_SP = eval_on_dataloader(model, train_loader_half)
        test_acc_half_SP = eval_on_dataloader(model, test_loader_half)
        results_list[i].update({'train_acc_half': train_acc_half, 'test_acc_half': test_acc_half, 'train_acc_half_SP': train_acc_half_SP, 'test_acc_half_SP': test_acc_half_SP})
        i+=1
    pickle.dump(results_list, open(final_path, 'wb'))

for seed in range(7,12):
    index = index = range(5, (24)* 5, 5)
    results_path_reg = f'results_reg/seed{seed}_0.05/results.pkl'
    results_path = f'results/seed{seed}/results_updated.pkl'
    results_reg = pickle.load(open(results_path_reg, 'rb'))[:23]
    results = pickle.load(open(results_path, 'rb'))[:23]
    results_reg_df = pd.DataFrame(results_reg, index = range(5, (len(results_reg)+1)* 5, 5))
    results_df = pd.DataFrame(results, index = range(5, (len(results)+1)* 5, 5))


    # concatenating everything:
    # first we rename all columns of results_reg_df to add _reg
    results_reg_df.columns = [f'{col}_reg' for col in results_reg_df.columns]
    final_df = pd.concat([results_reg_df, results_df], axis = 1)
    final_df.to_csv(f'results/seed{seed}/final_df.csv')



'''seed = 7
checkpoints_path = f'results/seed{seed}/checkpoints/'
checkpoints_path_reg = f'results_reg/seed{seed}_0.05/checkpoints/'
results_path = f'results_reg/seed{seed}_0.05/results.pkl'
results_reg = pickle.load(open(results_path, 'rb'))

loaders_half = get_cifar10_loaders(0.5, seed = 7)
test_loader = loaders_half['test_loader']

i = 0
for epoch in tqdm(index):
    checkpoint_path_reg = f'{checkpoints_path_reg}warm_up_{epoch}.pt'
    model = resnet18()
    model.load_state_dict(torch.load(checkpoint_path_reg))
    model.to('cuda')
    test_acc_half = eval_on_dataloader(model, test_loader)
    results_reg[i].update({'test_acc_half': test_acc_half})
    i+=1

results_reg_df = pd.DataFrame(results_reg, index = index)
with open('results_reg/seed7_0.05/results.pkl', 'wb') as f:
    pickle.dump(results_reg, f)'''