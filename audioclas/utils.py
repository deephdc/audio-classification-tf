"""
Miscellaneous utils

Date: September 2018
Author: Ignacio Heredia
Email: iheredia@ifca.unican.es
Github: ignacioheredia
"""

import os
import subprocess
from distutils.dir_util import copy_tree
from multiprocessing import Process

import numpy as np
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K

from audioclas import paths
from audioclas.optimizers import customSGD, customAdam, customAdamW


def create_dir_tree():
    """
    Create directory tree structure
    """
    dirs = paths.get_dirs()
    for d in dirs.values():
        if not os.path.isdir(d):
            print('creating {}'.format(d))
            os.makedirs(d)


def remove_empty_dirs():
    basedir = paths.get_base_dir()
    dirs = os.listdir(basedir)
    for d in dirs:
        d_path = os.path.join(basedir, d)
        if not os.listdir(d_path):
            os.rmdir(d_path)


def backup_splits():
    """
    Save the data splits used during training to the timestamped dir.
    """
    src = paths.get_splits_dir()
    dst = paths.get_ts_splits_dir()
    copy_tree(src, dst)


def get_custom_objects():
    return {'customSGD': customSGD,
            'customAdam': customAdam,
            'customAdamW': customAdamW}


class LR_scheduler(callbacks.LearningRateScheduler):
    """
    Custom callback to decay the learning rate. Schedule follows a 'step' decay.

    Reference
    ---------
    https://github.com/keras-team/keras/issues/898#issuecomment-285995644
    """
    def __init__(self, lr_decay=0.1, epoch_milestones=[]):
        self.lr_decay = lr_decay
        self.epoch_milestones = epoch_milestones
        super().__init__(schedule=self.schedule)

    def schedule(self, epoch):
        current_lr = K.eval(self.model.optimizer.lr)
        if epoch in self.epoch_milestones:
            new_lr = current_lr * self.lr_decay
            print('Decaying the learning rate to {}'.format(new_lr))
        else:
            new_lr = current_lr
        return new_lr


class LRHistory(callbacks.Callback):
    """
    Custom callback to save the learning rate history

    Reference
    ---------
    https://stackoverflow.com/questions/49127214/keras-how-to-output-learning-rate-onto-tensorboard
    """
    def __init__(self):  # add other arguments to __init__ if needed
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr).astype(np.float64)})
        super().on_epoch_end(epoch, logs)


def launch_tensorboard(port, logdir):
    subprocess.call(['tensorboard',
                     '--logdir', '{}'.format(logdir),
                     '--port', '{}'.format(port),
                     '--host', '0.0.0.0'])


def get_callbacks(CONF, use_lr_decay=True):
    """
    Get a callback list to feed fit_generator.
    #TODO Use_remote callback needs proper configuration
    #TODO Add ReduceLROnPlateau callback?

    Parameters
    ----------
    CONF: dict

    Returns
    -------
    List of callbacks
    """

    calls = []

    # Add mandatory callbacks
    calls.append(callbacks.TerminateOnNaN())
    calls.append(LRHistory())

    # Add optional callbacks
    if use_lr_decay:
        milestones = np.array(CONF['training']['lr_step_schedule']) * CONF['training']['epochs']
        milestones = milestones.astype(np.int)
        calls.append(LR_scheduler(lr_decay=CONF['training']['lr_step_decay'],
                                  epoch_milestones=milestones.tolist()))

    if CONF['monitor']['use_tensorboard']:
        calls.append(callbacks.TensorBoard(log_dir=paths.get_logs_dir(),
                                           write_graph=False,
                                           profile_batch=0))  # https://github.com/tensorflow/tensorboard/issues/2084#issuecomment-483395808

        # # Let the user launch Tensorboard
        # print('Monitor your training in Tensorboard by executing the following comand on your console:')
        # print('    tensorboard --logdir={}'.format(paths.get_logs_dir()))

        # Run Tensorboard on a separate Thread/Process on behalf of the user
        port = os.getenv('monitorPORT', 6006)
        port = int(port) if len(str(port)) >= 4 else 6006
        subprocess.run(['fuser', '-k', '{}/tcp'.format(port)])  # kill any previous process in that port
        p = Process(target=launch_tensorboard, args=(port, paths.get_logs_dir()), daemon=True)
        p.start()

    if CONF['monitor']['use_remote']:
        calls.append(callbacks.RemoteMonitor())

    if CONF['training']['use_validation'] and CONF['training']['use_early_stopping']:
        calls.append(callbacks.EarlyStopping(patience=int(0.1 * CONF['training']['epochs'])))

    if CONF['training']['ckpt_freq'] is not None:
        calls.append(callbacks.ModelCheckpoint(
            os.path.join(paths.get_checkpoints_dir(), 'epoch-{epoch:02d}.hdf5'),
            verbose=1,
            period=max(1, int(CONF['training']['ckpt_freq'] * CONF['training']['epochs']))))

    if not calls:
        calls = None

    return calls
