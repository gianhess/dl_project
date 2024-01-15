import matplotlib.pyplot as plt
import pickle
import pandas as pd

    


def plot_correlations(results, fixed_metric, changing_metrics):
    skips = range(0, 250, 5)

    # Create a 2x2 subplot grid
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

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
        axes[i].set_facecolor('#f0f0f0')
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




def get_results_df(seed, dir = 'results/'):
    path = f'{dir}seed{seed}/results.pkl'
    with open(path, 'rb') as f:
        results = pickle.load(f)
        index = range(5, (len(results)+1)* 5, 5)
        results = pd.DataFrame(results, index = index)
    return results

def get_results(seed, dir = 'results/'):
    path = f'{dir}seed{seed}/results.pkl'
    with open(path, 'rb') as f:
        results = pickle.load(f)
    return results

def plot_pairwise_subplots(df, metrics1, metric2, interp=100, labely2=None, start = None, end = None, title = None):

    if end is None:
        end = df.index[-1]

    if start is None:
        start = 5
    start = int(start//5 -1)
    end = int(end//5)

    df = df.iloc[start:end]
    if labely2 is None:
        labely2 = metric2

    x = df.index
    y2 = df[metric2]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    if title is not None: fig.suptitle(f'Subplots for {labely2} on CIFAR10', fontsize=16)

    for i, ax in enumerate(axes.flatten()):
        metric1 = metrics1[i] if i < len(metrics1) else metrics1[-1]  # Use the last metric if not enough provided
        y1 = df[metric1]
        y1_sp = df[f'{metric1}_SP']

        line1, = ax.plot(x, y1, '-', color='red', label=metric1, linewidth=1)
        #ax.plot(x, y1_sp, '--', color='orange', label=f'{metric1}_SP', linewidth=1)
        ax.set_xlabel('Epochs of Warm Up', fontsize=13)
        ax.set_ylabel(f'{metric1}', color='red', fontsize=13)
        ax.tick_params('y', colors='red')

        ax2 = ax.twinx()
        line2, = ax2.plot(x, y2, '-', color='blue', label=metric2, linewidth=1)
        ax2.set_ylabel(labely2, color='blue', fontsize=13)
        ax2.tick_params('y', colors='blue')

        # Additional lines for metric2_SP
        y2_sp = df[f'{metric2}_SP']
        #ax2.plot(x, y2_sp, '--', color='purple', label=f'{metric2}_SP', linewidth=1)

        ax.set_facecolor('#f0f0f0')  # Background color
        ax.grid(True, linestyle='--', alpha=0.7)

        if interp:
            ax.axvline(x=interp, color='grey', linestyle='--', linewidth=1, label='Interpolation')

        # Create individual legend for each subplot
        lines = [line1, line2]
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, loc='upper right')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to avoid overlapping titles

    plt.show()


def sliding_window_correlation(df, col1, col2, window_size):

    window_size = window_size //5

    # Initialize an empty list to store correlation values
    correlations = []

    # Iterate over the DataFrame with a sliding window
    for i in range(len(df) - window_size + 1):
        # Extract the window
        window = df.iloc[i:i + window_size]

        # Calculate the correlation between col1 and col2 in the window
        correlation = window[col1].corr(window[col2])

        # Append the correlation value to the list
        correlations.append(correlation)

    # Convert the list of correlations to a pandas Series
    result_series = pd.Series(correlations, index=df.index[window_size - 1:])

    return result_series

def plot_correlation(dfs, col1, col2, window_size):
        correlation_series = sliding_window_correlation(dfs[0], col1, col2, window_size)
        for df in dfs[1:]:
            # Calculate the correlation series
            correlation_serie = sliding_window_correlation(df, col1, col2, window_size)
            correlation_series = pd.concat([correlation_series,correlation_serie], axis = 1)
        mean_correlation = correlation_series.mean(axis=1)
        std_correlation = correlation_series.std(axis=1)
 
    
        # Plot the correlation series
        plt.figure(figsize=(8, 6))
        plt.title(f'Correlation between {col1} and {col2} with a window size of {window_size}')
        plt.xlabel('Last Warm Up epoch in the window')
        plt.ylabel(f'Correlation')
        plt.plot(mean_correlation, color='red', linewidth=1, label='Mean')
        plt.fill_between(mean_correlation.index, mean_correlation - std_correlation, mean_correlation + std_correlation,
                         color='red', alpha=0.2, label='Std')
        plt.legend(loc='lower right')
        plt.grid()
        ax = plt.gca()
        ax.axhline(0, color='grey', linewidth=1, linestyle='--')
        ax.set_facecolor('#f0f0f0')  # Background color
        plt.show()



def plot_metrics_with_error_bars(metric_names, y_label, metric_labels):
    final_dfs = [pd.read_csv(f'results/seed{seed}/final_df.csv', index_col= 0) for seed in range(7,12)]

    # plotting it
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.get_cmap('plasma', len(metric_names))
    for i, metric_name in enumerate(metric_names):
        metric = final_dfs[0][metric_name]
        for df in final_dfs:
            metric = pd.concat([metric, df[metric_name]], axis = 1)
        metric_mean = metric.mean(axis = 1)
        metric_std = metric.std(axis = 1)
        if i <2:
            ax.plot(metric_mean, label=metric_labels[i], color= colors(i))
            ax.fill_between(metric_mean.index, metric_mean - metric_std, metric_mean + metric_std, alpha=0.3, color=colors(i))
        else:
            # Gian's fault
            ax.plot(metric_mean, label=metric_labels[i], color= 'green')
            ax.fill_between(metric_mean.index, metric_mean - metric_std, metric_mean + metric_std, alpha=0.3, color='green')
    ax.set_xlabel('Warm Up Epochs')
    ax.set_ylabel(y_label)

    ax.set_facecolor('#f0f0f0')
    plt.title(f'Mean and Std of {y_label}')
    ax.grid()
    ax.legend(loc='upper left')
    plt.show()