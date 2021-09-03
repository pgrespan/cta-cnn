import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
import matplotlib.pyplot as plt

def plot_ROC(dataframe, outpath, n_curves=5, energy_is_log=True):
    df = dataframe
    fpr, tpr, _ = roc_curve(df['class'], df['gammaness'])

    energy = 'energy_reco'
    if not energy_is_log:
        df[energy] = np.log10(df[energy])

    edges = np.linspace(-2, 2, n_curves + 1)

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    ax = axs
    #ax = axs[0]

    cmap = plt.get_cmap("tab10")
    '''
    ax.plot(fpr, tpr, label='Classifier', color=cmap(0))
    ax.set_xlabel('False Positive Rate (1-specitivity)')
    ax.set_ylabel('True Positive Rate (sensitivity)')
    ax.legend(loc='lower right', fancybox=True, framealpha=0., prop={'size': 18})
    ax.grid(b=True, which='major', linestyle='--')
    ax.grid(b=True, which='minor', linestyle='--')

    ax = axs[1]
    '''

    #ax.set_prop_cycle('color', plt.cm.Spectral(np.linspace(0, 1, n_curves)))

    for i in range(n_curves):
        edge1 = edges[i]
        edge2 = edges[i + 1]
        dfbe = df[(df[energy] >= edge1) & (df[energy] < edge2)]
        try:
            # print(i)
            fpr, tpr, _ = roc_curve(dfbe['class'], dfbe['gammaness'])
            roc_auc = roc_auc_score(dfbe['class'], dfbe['gammaness'])
            ax.plot(fpr, tpr, linewidth=3, label='AUC={:.3f}, [{:.2f}, {:.2f}] TeV'.format(roc_auc, 10 ** edge1, 10 ** edge2))
        except:
            print("Merda")

    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.legend(loc='lower right', fancybox=True, framealpha=0., prop={'size': 18})
    ax.set_xlabel('False Positive Rate (1-specitivity)', fontsize=20)
    ax.set_ylabel('True Positive Rate (sensitivity)', fontsize=20)
    ax.grid(b=True, which='major', linestyle='--')
    ax.grid(b=True, which='minor', linestyle='--')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    fig.tight_layout()
    plt.show()
    fig.savefig(outpath+'roc_rocs.eps', format='eps', bbox_inches='tight', transparent=True)
    plt.close()
    return None