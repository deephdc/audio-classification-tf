""""
Miscellaneous functions to plot.

Date: September 2018
Author: Ignacio Heredia
Email: iheredia@ifca.unican.es
Github: ignacioheredia
"""

import os
import json

import matplotlib.pylab as plt
import numpy as np

from audioclas import paths


def training_plots(conf, stats, show_val=True, show_ckpt=True):
    """
    Plot the loss and accuracy metrics for a timestamped training.

    Parameters
    ----------
    conf : dict
        Configuration dict
    stats : dict
        Statistics dict
    show_val: bool
        Plot the validation data if available
    show_ckpt : bool
        Plot epochs at which ckpts have been made
    """
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # Training
    axs[0].plot(stats['epoch'], stats['loss'], label='Training')
    axs[1].plot(stats['epoch'], stats['acc'], label='Training')

    # Validation
    if (conf['training']['use_validation']) and show_val:
        axs[0].plot(stats['epoch'], stats['val_loss'], label='Validation')
        axs[1].plot(stats['epoch'], stats['val_acc'], label='Validation')

    # Model Checkpoints
    if (conf['training']['ckpt_freq'] is not None) and show_ckpt:
        period = max(1, int(conf['training']['ckpt_freq'] * conf['training']['epochs']))
        ckpts = np.arange(0, conf['training']['epochs'], period)
        for i, c in enumerate(ckpts):
            label = None
            if i == 0:
                label = 'checkpoints'
            axs[0].axvline(c, linestyle= '--', color='#f9d1e0')
            axs[1].axvline(c, linestyle= '--', color='#f9d1e0', label=label)

    axs[1].set_ylim([0, 1])
    axs[0].set_xlabel('Epochs'), axs[0].set_title('Loss')
    axs[1].set_xlabel('Epochs'), axs[1].set_title('Accuracy')
    axs[0].legend(loc='upper right')


def multi_training_plots(timestamps, legend_loc='upper right'):
    """
    Compare the loss and accuracy metrics for a timestamped training.

    Parameters
    ----------
    timestamps : str, or list of strs
        Configuration dict
    legend_loc: str
        Legend position
    """
    if timestamps is str:
        timestamps = [timestamps]

    fig, axs = plt.subplots(2, 2, figsize=(16, 16))
    axs = axs.flatten()

    for ts in timestamps:

        # Set the timestamp
        paths.timestamp = ts

        # Load training statistics
        stats_path = os.path.join(paths.get_stats_dir(), 'stats.json')
        with open(stats_path) as f:
            stats = json.load(f)

        # Load training configuration
        conf_path = os.path.join(paths.get_conf_dir(), 'conf.json')
        with open(conf_path) as f:
            conf = json.load(f)

        # Training
        axs[0].plot(stats['epoch'], stats['loss'], label=ts)
        axs[1].plot(stats['epoch'], stats['acc'], label=ts)

        # Validation
        if conf['training']['use_validation']:
            axs[2].plot(stats['epoch'], stats['val_loss'], label=ts)
            axs[3].plot(stats['epoch'], stats['val_acc'], label=ts)

    axs[1].set_ylim([0, 1])
    axs[3].set_ylim([0, 1])

    for i in range(4):
        axs[0].set_xlabel('Epochs')

    axs[0].set_title('Training Loss')
    axs[1].set_title('Training Accuracy')
    axs[2].set_title('Validation Loss')
    axs[3].set_title('Validation Accuracy')

    axs[0].legend(loc=legend_loc)
