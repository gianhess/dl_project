"""Example using neural collapse metrics"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from util.neural_collapse import NC1


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
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        return x


def train(log_interval, model, device, train_loader, optimizer, epoch, num_classes):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            nc1 = NC1(model=model, inputs=data, targets=target, num_classes=num_classes)

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tNC1: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), nc1))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Example computing NC (batch-wise)')
    parser.add_argument('--device', type=str, default="cuda",
                        help='device to use for training. Options: "cpu", "mps" (apple silicon) or "cuda". '
                             'Default: "cuda".')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs. Default: 10')
    parser.add_argument('--log_interval', type=int, default=1,
                        help='frequency of logging during training. Default: 1.')
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

    NUM_DATAPOINTS = 5000

    dataset = datasets.MNIST('../data', transform=transform, download=True)

    num_classes = len(dataset.classes)

    dataset = torch.utils.data.Subset(dataset, range(NUM_DATAPOINTS))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

    model = ConvNet().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args.log_interval, model, args.device, train_loader, optimizer, epoch, num_classes)


if __name__ == "__main__":
    main()
