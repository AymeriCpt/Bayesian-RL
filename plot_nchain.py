import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


METHODS = ['random', 'bss', 'bfs3', 'optimal']
HORIZONS = [5, 10, 15]


def plot_cumulative_rewards():
    all_steps, all_cum_rews = {}, {}

    for method in METHODS:
        steps, cum_rews = [], []

        logfiles = glob.glob("./logs/{}*".format(method + '_h5' if method in ('bss', 'bfs3') else method))
        print('method {}, n logs = {}'.format(method, len(logfiles)))

        for logfile in logfiles:
            with open(logfile, 'rb') as f:
                logs = pickle.load(f)

            cum_rew = 0
            for step, rew in enumerate(logs['rewards']):
                cum_rew += rew
                steps.append(step)
                cum_rews.append(cum_rew)
        
        all_steps[method] = np.array(steps, dtype=np.long)
        all_cum_rews[method] = np.array(cum_rews, dtype=np.float)

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in METHODS:
        label = method.upper() if method in ('bss', 'bfs3') else method.capitalize()
        sns.lineplot(x=all_steps[method], y=all_cum_rews[method], ax=ax, ci='sd', label=label)
    
    ax.set_title('Cumulative Rewards on 5-Chain')
    ax.set_xlim(0, 1000)
    ax.set_xlabel('steps')
    ax.set_ylim(0, 4000)
    ax.set_ylabel('cumulative reward')
    ax.grid(axis='y')
    ax.legend(loc='best')

    os.makedirs('./plots/', exist_ok=True)
    fig.savefig('./plots/cumulative_rewards.png')


def plot_horizon_comparison():
    all_cum_rews, all_horizons  = {}, {}
    for method in ('bss', 'bfs3'):
        cum_rews, horizons = [], []

        for horizon in HORIZONS:  
            logfiles = glob.glob('./logs/{}_h{}*'.format(method, horizon))
            print('method {}, horizon {}, n logs = {}'.format(method, horizon, len(logfiles)))

            for logfile in logfiles:
                with open(logfile, 'rb') as f:
                    logs = pickle.load(f)

                cum_rews.append(logs['cumulative_reward'])
                horizons.append(horizon)

            all_cum_rews[method] = np.array(cum_rews, dtype=np.float)
            all_horizons[method] = np.array(horizons, dtype=np.long)

    fig, ax = plt.subplots(figsize=(8, 5))
            
    for method in ('bss', 'bfs3'):
        label = method.upper()
        sns.lineplot(x=all_horizons[method], y=all_cum_rews[method], ci='sd', ax=ax,
            label=label, err_style='bars', err_kws=dict(capsize=10))
    
    ax.set_title('Planning Horizon Comparison')
    ax.set_xlabel('planning horizon')
    ax.set_xticks(HORIZONS)
    ax.set_ylabel('cumulative reward')
    ax.grid(axis='y')
    ax.legend(loc='lower center', bbox_to_anchor=(0.25, 0.05))

    os.makedirs('./plots/', exist_ok=True)
    fig.savefig('./plots/horizon_comparison.png')


if __name__ == "__main__":
    plot_cumulative_rewards()
    plot_horizon_comparison()