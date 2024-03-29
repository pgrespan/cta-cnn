import argparse
import os
import pickle

# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def train_plots(filen, tran):
    folder = os.path.dirname(filen)

    # read data from training history file
    with open(filen, 'rb') as f:
        x = pickle.load(f)
        losses = x['losses']
        val_losses = x['val_losses']
        accuracy = x['accuracy']
        val_accuracy = x['val_accuracy']
        lrs = x['lrs']

    # get number of epochs
    epochs = range(1, len(losses) + 1)

    # training loss & accuracy
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    ax = axs[0]
    ax.plot(epochs, losses, marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss [binary crossentropy]')
    ax.set_title('Training loss')
    ax.grid(True, )

    ax = axs[1]
    ax.plot(epochs, accuracy, label='Accuracy', marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training accuracy')
    ax.grid(True)

    # fig.suptitle('Training history')


    ax = axs[1]
    ax.plot(epochs, accuracy, label='Accuracy', marker='.')
    ax.plot(epochs, val_accuracy, label='Validation accuracy', marker='.')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training + Validation Accuracy')
    ax.grid(True, linestyle='--')
    ax.legend(loc='lower right', fancybox=True, framealpha=0.)

    fig3.suptitle('Training history')

    fig3.savefig(folder + '/classifier_train_val.pdf', format='pdf', transparent=tran, bbox_inches="tight")

    # lr
    fig3, axs3 = plt.subplots(nrows=1, ncols=1, figsize=(6.5, 5))

    axs3.plot(epochs, lrs, label='Learning rate', marker='.', linewidth=0.5)
    axs3.set_xlabel('Epoch')
    axs3.set_ylabel('Learning rate')
    axs3.set_title('Learning rate')
    axs3.grid(True, linestyle='--')

    # fig3.suptitle('Training history')

    fig3.savefig(folder + '/lr.pdf', format='pdf', transparent=tran)

    # plt.show()


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
