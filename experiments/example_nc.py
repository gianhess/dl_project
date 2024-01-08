"""Example using neural collapse metrics"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

from util.neural_collapse import NC1, NC2, NC3, NC4


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.embed(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def embed(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        return x


def train(log_interval, model, device, train_loader, optimizer, epoch, num_classes, metrics_history):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            nc1_single_batch = NC1(model=model, inputs=data, targets=target, num_classes=num_classes)
            nc2_single_batch = NC2(model=model)
            nc3_single_batch = NC3(model=model, inputs=data, targets=target, num_classes=num_classes, use_cache=True)
            nc4_single_batch = NC4(model=model, inputs=data, targets=target, num_classes=num_classes, use_cache=True)

        metrics_history['nc1_single_batch'].append(nc1_single_batch.cpu().numpy())
        metrics_history['nc2_single_batch'].append(nc2_single_batch)
        metrics_history['nc3_single_batch'].append(nc3_single_batch)
        metrics_history['nc4_single_batch'].append(nc4_single_batch)

        print(
            'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tNC1: {:.6f}\tNC2: {:.6f}\tNC3: {:.6f}\tNC4: {:.6f}'
            .format(epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(), nc1_single_batch, nc2_single_batch,
                    nc3_single_batch, nc4_single_batch)
        )

        if batch_idx % log_interval == 0:
            print('---------------------------------------')
            print('Metrics computed for the whole data set')

            with torch.no_grad():
                # average over all batches
                nc1_batch_avg = nc2_batch_avg = nc3_batch_avg = nc4_batch_avg = 0
                for data, target in train_loader:
                    nc1_batch_avg += NC1(model=model, inputs=data, targets=target, num_classes=num_classes)
                    nc2_batch_avg += NC2(model=model)
                    nc3_batch_avg += NC3(model=model, inputs=data, targets=target, num_classes=num_classes,
                                         use_cache=True)
                    nc4_batch_avg += NC4(model=model, inputs=data, targets=target, num_classes=num_classes,
                                         use_cache=True)

                nc1_batch_avg /= len(train_loader)
                nc2_batch_avg /= len(train_loader)
                nc3_batch_avg /= len(train_loader)
                nc4_batch_avg /= len(train_loader)

                metrics_history['nc1_batch_avg'].append(nc1_batch_avg.cpu().numpy())
                metrics_history['nc2_batch_avg'].append(nc2_batch_avg)
                metrics_history['nc3_batch_avg'].append(nc3_batch_avg)
                metrics_history['nc4_batch_avg'].append(nc4_batch_avg)

                print(
                    'Batch average: \tNC1: {:.6f}\tNC2: {:.6f}\tNC3: {:.6f}\tNC4: {:.6f}'
                    .format(nc1_batch_avg, nc2_batch_avg, nc3_batch_avg, nc4_batch_avg)
                )

                # compute in one pass over data set
                nc1 = NC1(model=model, data_loader=train_loader, num_classes=num_classes)
                nc2 = NC2(model=model)
                nc3 = NC3(model=model, data_loader=train_loader, num_classes=num_classes, use_cache=True)
                nc4 = NC4(model=model, data_loader=train_loader, num_classes=num_classes, use_cache=True)

                metrics_history['nc1'].append(nc1.cpu().numpy())
                metrics_history['nc2'].append(nc2)
                metrics_history['nc3'].append(nc3)
                metrics_history['nc4'].append(nc4)

                metrics_history['steps'].append(len(metrics_history['nc1_single_batch']) - 1)

                print(
                    'All at once: \tNC1: {:.6f}\tNC2: {:.6f}\tNC3: {:.6f}\tNC4: {:.6f}'
                    .format(nc1, nc2, nc3, nc4)
                )
                print('---------------------------------------')


def plot_metrics(metrics_history, log_interval, save_path=None):
    single_steps = np.arange(0, len(metrics_history['nc1_single_batch']))
    steps = metrics_history['steps']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('NC Metrics')

    axes = axes.flatten()

    metrics_to_plot = ['nc1', 'nc2', 'nc3', 'nc4']
    labels = ['Single Batch', 'Batch Average', 'All at Once']

    for i, metric in enumerate(metrics_to_plot):
        axes[i].set_title(metric.upper())
        axes[i].set_xlabel('Steps')
        axes[i].set_ylabel(metric.upper())

        axes[i].plot(single_steps, metrics_history[f'{metric}_single_batch'], label=labels[0])
        axes[i].plot(steps, metrics_history[f'{metric}_batch_avg'], label=labels[1])
        axes[i].plot(steps, metrics_history[metric], label=labels[2])

        axes[i].legend()

    plt.tight_layout(rect=(0, 0, 1, 1))

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def save_metrics(metrics_history, save_path):
    np.savez(save_path, **metrics_history)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Example computing NC (batch-wise)')
    parser.add_argument('--device', type=str, default="cuda",
                        help='device to use for training. Options: "cpu", "mps" (apple silicon) or "cuda". '
                             'Default: "cuda".')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs. Default: 10')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='frequency of logging during training. Default: 100.')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size. Default: 32.')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate. Default: 1e-3.')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.device == "cuda":
        assert torch.cuda.is_available(), "CUDA is not available."

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST('../data', transform=transform, download=True)

    num_classes = len(dataset.classes)

    NUM_DATAPOINTS = 5000
    dataset = torch.utils.data.Subset(dataset, range(NUM_DATAPOINTS))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

    model = ConvNet().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    metrics_history = {
        'nc1_single_batch': [],
        'nc2_single_batch': [],
        'nc3_single_batch': [],
        'nc4_single_batch': [],
        'nc1_batch_avg': [],
        'nc2_batch_avg': [],
        'nc3_batch_avg': [],
        'nc4_batch_avg': [],
        'nc1': [],
        'nc2': [],
        'nc3': [],
        'nc4': [],
        'steps': []
    }

    for epoch in range(1, args.epochs + 1):
        train(args.log_interval, model, args.device, train_loader, optimizer, epoch, num_classes, metrics_history)

    save_metrics(metrics_history, 'metrics.npz')
    plot_metrics(metrics_history, args.log_interval, save_path='metrics.png')


if __name__ == "__main__":
    main()
