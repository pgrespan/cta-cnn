from __future__ import absolute_import
from __future__ import print_function

import keras
from keras import backend as K
import numpy as np


class LearningRateScheduler(keras.callbacks.History):

    def __init__(self, patience, spike_epochs=None, spike_multiple=10, min_delta=0.002,
                 decay_factor=0.5, mode='min', loss_type='val_acc'):

        super(LearningRateScheduler, self).__init__()

        self.patience = patience
        self.spike_epochs = spike_epochs
        self.spike_multiple = spike_multiple
        self.min_delta = min_delta
        self.decay_factor = decay_factor
        self.loss_type = loss_type
        self.wait = 0
        self.monitor_op = np.less

        if mode == 'max':
            self.monitor_op = np.greater

        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs={}):

        current_lr = K.get_value(self.model.optimizer.lr)

        # current = self.history[self.loss_type]

        current = logs.get(self.loss_type)

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            # print(' '.join(('Learning rate:', str(current_lr))))
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.best = current
                self.wait = 0
                print(' '.join(
                    ('Changing learning rate from', str(current_lr), 'to', str(current_lr * self.decay_factor))))
                K.set_value(self.model.optimizer.lr, current_lr * self.decay_factor)
                current_lr = current_lr * self.decay_factor

        if self.spike_epochs is not None and len(self.epoch) in self.spike_epochs:
            print(' '.join(
                ('Spiking learning rate from', str(current_lr), 'to', str(current_lr * self.spike_multiple))))
            K.set_value(self.model.optimizer.lr, current_lr * self.spike_multiple)
