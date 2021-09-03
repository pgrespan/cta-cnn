import argparse
import os
import pickle

import matplotlib.pyplot as plt


def train_plots(filen, tran):
    folder = os.path.dirname(filen)

    with open(filen, 'rb') as f:
        x = pickle.load(f)
        losses = x['losses']
        val_losses = x['val_losses']
        lrs = x['lrs']

    fig = plt.figure(figsize=(6, 6))

    epochs = range(1, len(losses) + 1)

    plt.plot(epochs, losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss [mean_absolute_error]')
    plt.title('Training loss')
    plt.grid(True)

    plt.suptitle('Training history')

    plt.savefig(folder + '/regressor_training.png', transparent=tran)

    fig2 = plt.figure(figsize=(6, 6))

    epochs = range(1, len(val_losses) + 1)

    plt.plot(epochs, val_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss [mean_absolute_error]')
    plt.title('Validation loss')
    plt.grid(True)

    plt.suptitle('Validation history')

    plt.savefig(folder + '/regressor_validation.png', transparent=tran)

    fig3 = plt.figure(figsize=(6, 6))

    epochs = range(1, len(val_losses) + 1)

    plt.plot(epochs, losses, label='loss', marker='.')
    plt.plot(epochs, val_losses, label='val_loss', marker='.')
    plt.xlabel('Epoch')
    plt.ylabel('Loss [mean absolute error]')
    plt.title('Training + Validation loss')
    plt.legend(loc='upper right', fancybox=True, framealpha=0.)
    plt.grid(True, which="major", linestyle='--')

    plt.suptitle('History')

    plt.savefig(folder + '/regressor_train_valid.png', transparent=tran)

    # lr
    fig3, axs3 = plt.subplots(nrows=1, ncols=1, figsize=(6.5, 5))

    axs3.plot(epochs, lrs, label='Learning rate', marker='.', linewidth=0.5)
    axs3.set_xlabel('Epoch')
    axs3.set_ylabel('Learning rate')
    axs3.set_title('Learning rate')
    axs3.grid(True, linestyle='--')

    # fig3.suptitle('Training history')

    fig3.savefig(folder + '/lr.eps', format='eps', transparent=tran)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--log', type=str, default='', help='Training history file.', required=True)
    parser.add_argument(
        '--transparent', type=bool, default=False, help='Specify whether plots have to be transparent.', required=False)

    FLAGS, unparsed = parser.parse_known_args()

    filen = FLAGS.log
    tran = FLAGS.transparent

    train_plots(filen, tran)
