import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve


def test_plots(csv):
    folder = os.path.dirname(csv)

    df = pd.read_csv(csv)

    r = np.arange(0, 1, 0.01)
    fp = np.zeros(r.shape[0])
    fn = np.zeros(r.shape[0])
    p_c = np.zeros(r.shape[0])
    g_c = np.zeros(r.shape[0])
    significance = np.zeros(r.shape[0])
    n_test_protons = (df['GroundTruth'] == 0).sum()
    n_test_gammas = (df['GroundTruth'] == 1).sum()
    print('Number of protons in the test set: ', n_test_protons)
    print('Number of gammas in the test set: ', n_test_gammas)
    n_test = df.shape[0]
    for i, thr in enumerate(r):
        fp[i] = df[(df['GroundTruth'] == 0) & (df['Predicted'] >= thr)].count()[0]
        fn[i] = df[(df['GroundTruth'] == 1) & (df['Predicted'] <= thr)].count()[0]

    x = PrettyTable()

    x.field_names = ['Threshold', 'Accepted gammas', 'Accepted protons']

    for i, thr in enumerate(r):
        p_c[i] = df[(df['GroundTruth'] == 0) & (df['Predicted'] >= thr)].count()[0]
        g_c[i] = df[(df['GroundTruth'] == 1) & (df['Predicted'] >= thr)].count()[0]
        significance[i] = (g_c[i] / n_test_gammas) / math.sqrt(p_c[i] / n_test_protons)
        x.add_row([round(thr, 2), round(g_c[i] / n_test_gammas, 2), round(p_c[i] / n_test_protons, 2)])

    y_gt = df['GroundTruth']
    y_pr = df['Predicted']

    ar = roc_auc_score(y_gt, y_pr)
    fpr, tpr, thresholds = roc_curve(y_gt, y_pr)
    accscore = accuracy_score(df['GroundTruth'], df['Predicted'].round(), normalize=True)

    print('AUC_ROC: ', ar)
    print('Accuracy: ', accscore)

    with open(folder + '/test_table.txt', 'w') as f:
        print(x, file=f)
        print('AUC_ROC: ', ar, file=f)
        print('Accuracy: ', accscore, file=f)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    ax = axs[0]

    ax.plot(fpr, tpr)
    ax.set_title('ROC')
    ax.set_xlabel('False Positive Rate (1-specitivity)')
    ax.set_ylabel('True Positive Rate (sensitivity)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)

    ax = axs[1]

    ax.plot(fp / n_test_protons, label="Protons")
    ax.plot(fn / n_test_gammas, label="Gammas")
    ax.set_title(r'$\zeta$ distribution')
    ax.set_xlabel(r'$\zeta$ [%]')
    ax.set_ylabel('Percentage %')
    # ax.set_xlim(0, 100)
    # ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend(borderaxespad=0.)

    fig.suptitle(r'ROC and $\zeta$ distribution')

    fig.savefig(folder + '/ROC.eps', format='eps', transparent=False)

    fig2, axs2 = plt.subplots(nrows=1, ncols=1)

    ax = axs2

    ax.set_title('Significance')
    ax.scatter(r, significance, s=3)
    ax.set_xlabel(r'$\zeta$')
    ax.set_ylabel('eg/Sqrt(ep)')

    fig2.savefig(folder + '/significance.eps', format='eps', transparent=False)

    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--csv', type=str, default='', help='CSV test file.', required=True)

    FLAGS, unparsed = parser.parse_known_args()

    csv = FLAGS.csv

    test_plots(csv)
