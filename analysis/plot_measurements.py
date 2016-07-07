import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot(input_csv):
    analysis = pd.read_csv(input_csv)
    alpha_primes = analysis['learning_factor'].unique()

    mins = {
        'dest_reached_rate': min(analysis['dest_reached_rate']),
        'mean_normed_time_left': min(analysis['mean_normed_time_left']),
        'mean_normed_n_penalties': min(analysis['mean_normed_n_penalties']),
        'mean_normed_cumulative_reward': min(analysis['mean_normed_cumulative_reward']),
        }
    maxs = {
        'dest_reached_rate': max(analysis['dest_reached_rate']),
        'mean_normed_time_left': max(analysis['mean_normed_time_left']),
        'mean_normed_n_penalties': max(analysis['mean_normed_n_penalties']),
        'mean_normed_cumulative_reward': max(analysis['mean_normed_cumulative_reward']),
        }

    for alpha_prime in alpha_primes:
        data = analysis[analysis['learning_factor'].isin([alpha_prime])]
        
        x_axis = data['discount_factor'].unique()
        y_axis = data['exploration_rate'].unique()

        dest_reached_rate = data['dest_reached_rate'].reshape(len(x_axis), len(y_axis))
        mean_normed_time_left = data['mean_normed_time_left'].reshape(len(x_axis), len(y_axis))
        mean_normed_n_penalties = data['mean_normed_n_penalties'].reshape(len(x_axis), len(y_axis))
        mean_normed_cumulative_reward = data['mean_normed_cumulative_reward'].reshape(len(x_axis), len(y_axis))

        fig, axes = plt.subplots(2, 2, figsize=(18,15))

        sns.heatmap(dest_reached_rate, ax=axes[0,0], xticklabels=x_axis.round(2), yticklabels=y_axis.round(2), vmin=mins['dest_reached_rate'], vmax=maxs['dest_reached_rate'])
        axes[0,0].set_title("Destination reached\nsuccess rate")
        sns.heatmap(mean_normed_time_left, ax=axes[0,1], xticklabels=x_axis.round(2), yticklabels=y_axis.round(2), vmin=mins['mean_normed_time_left'], vmax=maxs['mean_normed_time_left'])
        axes[0,1].set_title("Mean normalized time\nleft to destination")
        sns.heatmap(mean_normed_n_penalties, ax=axes[1,0], xticklabels=x_axis.round(2), yticklabels=y_axis.round(2), vmin=mins['mean_normed_n_penalties'], vmax=maxs['mean_normed_n_penalties'])
        axes[1,0].set_title("Mean normalized no. of\npenalties incurred")
        sns.heatmap(mean_normed_cumulative_reward, ax=axes[1,1], xticklabels=x_axis.round(2), yticklabels=y_axis.round(2), vmin=mins['mean_normed_cumulative_reward'], vmax=maxs['mean_normed_cumulative_reward'])
        axes[1,1].set_title("Mean normalized\ncumulative rewards")
        
        for ax in axes.flat:
            plt.setp(ax.get_xticklabels(), rotation=90)
            plt.setp(ax.get_yticklabels(), rotation=0)
            ax.set_xlabel("Discount factor")
            ax.set_ylabel("Exploration rate")
        
        fig.tight_layout()
        fig.savefig("plots_learning_{:.1f}.png".format(alpha_prime))

if __name__ == "__main__":
    # No error checks; use carefully
    plot(sys.argv[1])