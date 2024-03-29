""" Helper functions for EDA. """
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from textwrap import wrap

def plot_boxplot(df, columns, n_rows, n_cols, title=None, save_path=None, align_axes=False):
    # n_rows, n_cols = 5, 10 # 5 rows, 10 cols = 50 total vars 
    # plt.figure(figsize=(4*n_rows, 2*n_cols))
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(2*n_rows, 5*n_cols))
    if align_axes: min_, max_ = df.min().min(), df.max().max()
    ax_ = ax.ravel()
    for i in range(len(columns)):
        sns.set_style('whitegrid')
        sns.boxplot(df[columns[i]].values,color='green',orient='v', ax=ax_[i])
        ax_[i].set_title('\n'.join(wrap(columns[i], 40)), fontsize=8)
        if align_axes: ax_[i].set_ylim(min_, max_)
        # plt.title(l[i], wrap=True)
        plt.tight_layout()
    if title is not None: fig.suptitle(title, y=1.05, fontsize=16)
    if save_path is not None: plt.savefig(save_path, dpi=300)
    plt.close()

def plot_dist(df, columns, n_rows, n_cols, title=None, save_path=None, align_axes=False):
    # n_rows, n_cols = 5, 10 # 5 rows, 10 cols = 50 total vars 
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(3*n_rows, 8*n_cols))
    if align_axes: min_, max_ = df.min().min(), df.max().max()    
    ax_ = ax.ravel()
    for i in range(len(columns)):
        g = sns.histplot(ax=ax_[i], data=df, x=columns[i], kde=True)
        # g.set(xticklabels=[])
        g.set(xlabel=None)
        g.set(ylabel=None)
        ax_[i].set_title('\n'.join(wrap(columns[i], 20)), fontsize=10)
        if align_axes: ax_[i].set_ylim(min_, max_)
        plt.tight_layout()
    if title is not None: fig.suptitle(title, y=1.05, fontsize=16)
    if save_path is not None: plt.savefig(save_path, dpi=300)
    plt.close()

