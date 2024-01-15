from utils.warm_up import warm_up_weight_decay
import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Example computing NC (batch-wise)')
    parser.add_argument('--seed', type=int, default=7, help='random seed')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    warm_up_weight_decay(args.seed, weight_decay=args.weight_decay, max_epoch_warm_up=150)


if __name__ == "__main__":
    main()