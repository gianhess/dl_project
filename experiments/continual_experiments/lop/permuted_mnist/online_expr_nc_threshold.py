import sys
import json
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from lop.algos.bp import Backprop
from lop.algos.cbp import ContinualBackprop
from lop.nets.linear import MyLinear
from torch.nn.functional import softmax
from lop.nets.deep_ffnn import DeepFFNN
from lop.utils.miscellaneous import nll_accuracy, compute_matrix_rank_summaries
from lop.utils.neural_collapse import NC1, NC2, NC3, NC4
import wandb
import random


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

SEED = 42

def online_expr(params: {}):

    if params["wandb"]:
        wandb.init(
            # set the wandb project where this run will be logged
            project="permuted_mnist_heavy_first",

            name=f"nc_thresh_{params['nc_value']}",
            # name='test',
            # track hyperparameters and run metadata
            config=params
        )

    agent_type = params['agent']
    num_tasks = 200
    if 'num_tasks' in params.keys():
        num_tasks = params['num_tasks']
    if 'num_examples' in params.keys() and "change_after" in params.keys():
        num_tasks = int(params["num_examples"]/params["change_after"])

    step_size = params['step_size']
    opt = params['opt']
    weight_decay = 0
    use_gpu = 0
    dev = 'cpu'
    to_log = False
    num_features = 2000
    change_after = 10 * 6000
    to_perturb = False
    perturb_scale = 0.1
    num_hidden_layers = 1
    mini_batch_size = 1
    replacement_rate = 0.0001
    decay_rate = 0.99
    maturity_threshold = 100
    util_type = 'adaptable_contribution'
    nc_threshold = 0.25

    seed = SEED
    if "seed" in params.keys():
        seed = params["seed"]
    set_seed(seed)

    if 'to_log' in params.keys():
        to_log = params['to_log']
    if 'weight_decay' in params.keys():
        weight_decay = params['weight_decay']
    if 'num_features' in params.keys():
        num_features = params['num_features']
    if 'change_after' in params.keys():
        change_after = params['change_after']
    if 'use_gpu' in params.keys():
        if params['use_gpu'] == 1:
            use_gpu = 1
            dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            if dev == torch.device("cuda"):    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if 'to_perturb' in params.keys():
        to_perturb = params['to_perturb']
    if 'perturb_scale' in params.keys():
        perturb_scale = params['perturb_scale']
    if 'num_hidden_layers' in params.keys():
        num_hidden_layers = params['num_hidden_layers']
    if 'mini_batch_size' in params.keys():
        mini_batch_size = params['mini_batch_size']
    if 'replacement_rate' in params.keys():
        replacement_rate = params['replacement_rate']
    if 'decay_rate' in params.keys():
        decay_rate = params['decay_rate']
    if 'maturity_threshold' in params.keys():
        maturity_threshold = params['mt']
    if 'util_type' in params.keys():
        util_type = params['util_type']
    if 'nc_value' in params.keys():
        nc_threshold = params['nc_value']

    classes_per_task = 10
    images_per_class = 6000
    input_size = 784
    num_hidden_layers = num_hidden_layers
    net = DeepFFNN(input_size=input_size, num_features=num_features, num_outputs=classes_per_task,
                   num_hidden_layers=num_hidden_layers)

    if agent_type == 'linear':
        net = MyLinear(
            input_size=input_size, num_outputs=classes_per_task
        )
        net.layers_to_log = []

    if agent_type in ['bp', 'linear', "l2"]:
        learner = Backprop(
            net=net,
            step_size=step_size,
            opt=opt,
            loss='nll',
            weight_decay=weight_decay,
            device=dev,
            to_perturb=to_perturb,
            perturb_scale=perturb_scale,
        )
    elif agent_type in ['cbp']:
        learner = ContinualBackprop(
            net=net,
            step_size=step_size,
            opt=opt,
            loss='nll',
            replacement_rate=replacement_rate,
            maturity_threshold=maturity_threshold,
            decay_rate=decay_rate,
            util_type=util_type,
            accumulate=True,
            device=dev,
        )

    accuracy = nll_accuracy
    examples_per_task = images_per_class * classes_per_task
    total_examples = int(num_tasks * change_after* 10)
    total_iters = int(total_examples/mini_batch_size)
    save_after_every_n_tasks = 1
    if num_tasks >= 10:
        save_after_every_n_tasks = int(num_tasks/10)

    accuracies = torch.zeros(total_iters, dtype=torch.float)
    # weight_mag_sum = torch.zeros((total_iters, num_hidden_layers+1), dtype=torch.float)

    # rank_measure_period = change_after
    # effective_ranks = torch.zeros((int(total_examples/rank_measure_period), num_hidden_layers), dtype=torch.float)
    # approximate_ranks = torch.zeros((int(total_examples/rank_measure_period), num_hidden_layers), dtype=torch.float)
    # approximate_ranks_abs = torch.zeros((int(total_examples/rank_measure_period), num_hidden_layers), dtype=torch.float)
    # ranks = torch.zeros((int(total_examples/rank_measure_period), num_hidden_layers), dtype=torch.float)
    # dead_neurons = torch.zeros((int(total_examples/rank_measure_period), num_hidden_layers), dtype=torch.float)

    #initialize NC metrics per task
    nc1 = torch.zeros((num_tasks+1), dtype=torch.float)
    # nc2 = torch.zeros((num_tasks+1), dtype=torch.float)
    # nc3 = torch.zeros((num_tasks+1), dtype=torch.float)
    # nc4 = torch.zeros((num_tasks+1), dtype=torch.float)

    iter = 0
    with open('data/mnist_', 'rb+') as f:
        x, y, _, _ = pickle.load(f)
        if use_gpu == 1:
            x = x.to(dev)
            y = y.to(dev)

    for task_idx in (range(num_tasks)):
        is_first = True if task_idx < 1 else False

        new_iter_start = iter

        pixel_permutation = np.random.permutation(input_size)
        x = x[:, pixel_permutation]
        data_permutation = np.random.permutation(examples_per_task)
        x, y = x[data_permutation], y[data_permutation]

        counter = 0

        while ((nc1[task_idx + 1] == 0 or nc1[task_idx + 1] >= nc_threshold) and is_first) or not is_first:
           
            for start_idx in tqdm(range(0, change_after, mini_batch_size)):
                start_idx = start_idx % examples_per_task
                batch_x = x[start_idx: start_idx+mini_batch_size]
                batch_y = y[start_idx: start_idx+mini_batch_size]

                # train the network
                loss, network_output = learner.learn(x=batch_x, target=batch_y)

                # if to_log and agent_type != 'linear':
                #     for idx, layer_idx in enumerate(learner.net.layers_to_log):
                #         weight_mag_sum[iter][idx] = learner.net.layers[layer_idx].weight.data.abs().sum()
                # log accuracy
                with torch.no_grad():
                    accuracies[iter] = accuracy(softmax(network_output, dim=1), batch_y).cpu()
                iter += 1

            with torch.no_grad():
                nc1[task_idx+1] = NC1(model=net, inputs=x, targets=y, num_classes=10)
                #nc2[task_idx+1] = NC2(model=net)
                #nc3[task_idx+1] = NC3(model=net, inputs=x, targets=y, num_classes=10)
                #nc4[task_idx+1] = NC4(model=net, inputs=x, targets=y, num_classes=10)
            counter += 1

            if not is_first:
                break
        
        print("NC threshold reached after ", counter, " iterations")

        if params["wandb"]:
            wandb.log({"accuracies": accuracies[new_iter_start:iter - 1].mean(), "nc1": nc1[task_idx],
                        #"nc2": nc2[task_idx],"nc3": nc3[task_idx], "nc4": nc4[task_idx],
                        # 'approximate_ranks_layer1': approximate_ranks[task_idx][0].cpu(),
                        # 'approximate_ranks_layer2': approximate_ranks[task_idx][1].cpu(),
                        # 'approximate_ranks_layer3': approximate_ranks[task_idx][2].cpu(),
                        # 'dead_neurons_layer1': dead_neurons[task_idx][0].cpu(),
                        # 'dead_neurons_layer2': dead_neurons[task_idx][1].cpu(),
                        # 'dead_neurons_layer3': dead_neurons[task_idx][2].cpu()
                        })


        print('recent accuracy', accuracies[new_iter_start:iter - 1].mean())
        #print(' accuracy', accuracies[iter - 1])
        if task_idx % save_after_every_n_tasks == 0:
            data = {
                'accuracies': accuracies.cpu(),
                # 'weight_mag_sum': weight_mag_sum.cpu(),
                # 'ranks': ranks.cpu(),
                # 'effective_ranks': effective_ranks.cpu(),
                # 'approximate_ranks': approximate_ranks.cpu(),
                # 'abs_approximate_ranks': approximate_ranks_abs.cpu(),
                # 'dead_neurons': dead_neurons.cpu(),
                'nc1': nc1,
                #'nc2': nc2,
                #'nc3': nc3,
                #'nc4': nc4,
            }
            save_data(file=params['data_file'], data=data)


    data = {
        'accuracies': accuracies.cpu(),
        # 'weight_mag_sum': weight_mag_sum.cpu(),
        # 'ranks': ranks.cpu(),
        # 'effective_ranks': effective_ranks.cpu(),
        # 'approximate_ranks': approximate_ranks.cpu(),
        # 'abs_approximate_ranks': approximate_ranks_abs.cpu(),
        # 'dead_neurons': dead_neurons.cpu(),
        'nc1': nc1,
        #'nc2': nc2,
        #'nc3': nc3,
        #'nc4': nc4,
    }
    save_data(file=params['data_file'], data=data)


def save_data(file, data):
    with open(file, 'wb+') as f:
        pickle.dump(data, f)


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', help="Path to the file containing the parameters for the experiment",
                        type=str, default='temp_cfg/0.json')
    args = parser.parse_args(arguments)
    cfg_file = args.c

    with open(cfg_file, 'r') as f:
        params = json.load(f)

    online_expr(params)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
