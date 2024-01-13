import matplotlib.pyplot as plt
import pickle
import pandas as pd

def plot_pairwise_subplots(df, metrics1, metric2, interp=100, labely2=None):
    if labely2 is None:
        labely2 = metric2

    x = df.index
    y2 = df[metric2]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    fig.suptitle(f'Subplots for {labely2} on CIFAR10', fontsize=16)

    for i, ax in enumerate(axes.flatten()):
        metric1 = metrics1[i] if i < len(metrics1) else metrics1[-1]  # Use the last metric if not enough provided
        y1 = df[metric1]
        y1_sp = df[f'{metric1}_SP']

        line1, = ax.plot(x, y1, '-', color='red', label=metric1, linewidth=1)
        ax.plot(x, y1_sp, '--', color='orange', label=f'{metric1}_SP', linewidth=1)
        ax.set_xlabel('Epochs of Warm Up')
        ax.set_ylabel(f'{metric1}', color='red')
        ax.tick_params('y', colors='red')

        ax2 = ax.twinx()
        line2, = ax2.plot(x, y2, '-', color='blue', label=metric2, linewidth=1)
        ax2.set_ylabel(labely2, color='blue')
        ax2.tick_params('y', colors='blue')

        # Additional lines for metric2_SP
        y2_sp = df[f'{metric2}_SP']
        ax2.plot(x, y2_sp, '--', color='purple', label=f'{metric2}_SP', linewidth=1)

        ax.set_facecolor('#f0f0f0')  # Background color
        ax.grid(True, linestyle='--', alpha=0.7)

        if interp:
            ax.axvline(x=interp, color='grey', linestyle='--', linewidth=1, label='Interpolation')

        # Create individual legend for each subplot
        lines = [line1, line2, ax.lines[-2], ax2.lines[-1]]
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, loc='upper right')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to avoid overlapping titles

    plt.show()


def subplots_from_seed(seed, metrics1, metric2, interp=100, labely2=None, trunc_rate=0.1):
    results_path = f'results/seed{seed}/results.pkl'
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
        index = range(5, (len(results)+1)* 5, 5)
        results = pd.DataFrame(results, index = index)
    
    results = results.iloc[int(len(results) * (trunc_rate)):]
    plot_pairwise_subplots(results, metrics1, metric2, interp, labely2)
    


def plot_correlations(results, fixed_metric, changing_metrics):
    skips = range(0, 250, 5)

    # Create a 2x2 subplot grid
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

    # Flatten the axes array for easy indexing
    axes = axes.flatten()

    for i, changing_metric in enumerate(changing_metrics):
        corrs = []
        corrs_sp = []  # For correlations with the changing metric + "_SP"

        for skip in skips:
            results_trunc = results.iloc[int(skip/5):, :]
            corrs.append(results_trunc[fixed_metric].corr(results_trunc[changing_metric]))
            
            # Calculate correlation with the changing metric + "_SP"
            changing_metric_sp = changing_metric + "_SP"
            fixed_metric_sp = fixed_metric + "_SP"
            corrs_sp.append(results_trunc[fixed_metric_sp].corr(results_trunc[changing_metric_sp]))

        # Plot on the i-th subplot
        axes[i].plot(skips, corrs, label=f"{changing_metric}", linewidth=0.7)
        axes[i].plot(skips, corrs_sp, label=f"{changing_metric}_SP", linestyle='dashed', linewidth=0.7)
        axes[i].set_xlabel('Epochs skipped')
        axes[i].set_ylabel('Correlation')
        axes[i].set_title(f'Correlation between {fixed_metric} and {changing_metric} \n (w and w/o SP) truncating the first epochs')
        axes[i].grid(True)
        axes[i].hlines(0, 0, 250, colors='red', linestyles='dashed', linewidth=2)
        axes[i].legend()

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_correlations_from_seed(seed, fixed_metric, changing_metrics):
    results_path = f'results/seed{seed}/results.pkl'
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
        index = range(5, (len(results)+1)* 5, 5)
        results = pd.DataFrame(results, index = index)
    
    plot_correlations(results, fixed_metric, changing_metrics)