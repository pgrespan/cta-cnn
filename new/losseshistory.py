import keras
from keras import backend as K


class LossHistoryC(keras.callbacks.Callback):
    def __init__(self):
        object.__init__(self)
        self.losses = []
        self.val_losses = []
        self.accuracy = []
        self.val_accuracy = []
        # self.precision = []
        # self.val_precision = []
        # self.recall = []
        # self.val_recall = []
        self.lrs = []
        self.dic = {}

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.accuracy = []
        self.val_accuracy = []
        # self.precision = []
        # self.val_precision = []
        # self.recall = []
        # self.val_recall = []
        self.dic = {}

    # def on_batch_end(self, batch, logs={}):
        # self.losses.append(logs.get('loss'))
        # self.val_losses.append(logs.get('val_loss'))
        # self.accuracy.append(logs.get('acc'))
        # self.val_accuracy.append(logs.get('val_acc'))

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.accuracy.append(logs.get('acc'))
        self.val_accuracy.append(logs.get('val_acc'))
        # self.precision.append(logs.get('precision'))
        # self.val_precision.append(logs.get('val_precision'))
        # self.recall.append(logs.get('recall'))
        # self.val_recall.append(logs.get('val_recall'))
        self.lrs.append(K.eval(self.model.optimizer.lr))

    def on_train_end(self, logs=None):

        self.dic['losses'] = self.losses
        self.dic['val_losses'] = self.val_losses
        self.dic['accuracy'] = self.accuracy
        self.dic['val_accuracy'] = self.val_accuracy
        # self.dic['precision'] = self.precision
        # self.dic['val_precision'] = self.val_precision
        # self.dic['recall'] = self.recall
        # self.dic['val_recall'] = self.val_recall
        self.dic['lrs'] = self.lrs


class LossHistoryR(keras.callbacks.Callback):
    def __init__(self):
        object.__init__(self)
        self.losses = []
        self.val_losses = []
        self.lrs = []
        self.dic = {}

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.dic = {}

    # def on_batch_end(self, batch, logs={}):
        # self.losses.append(logs.get('loss'))
        # self.val_losses.append(logs.get('val_loss'))

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.lrs.append(K.eval(self.model.optimizer.lr))

    def on_train_end(self, logs=None):

        self.dic['losses'] = self.losses
        self.dic['val_losses'] = self.val_losses
        self.dic['lrs'] = self.lrs
