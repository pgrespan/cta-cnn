import argparse

import numpy as np
import pandas as pd
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from keras.utils.data_utils import OrderedEnqueuer
from keras.utils.generic_utils import Progbar
from tabulate import tabulate
from scipy.stats import norm

from generators import DataGeneratorChain
from utils import get_all_files


def get_mus_sigmas(df, npoints):

    edges = np.linspace(-2, 2, npoints + 1)
    mus = np.array([])
    sigmas = np.array([])

    for i in range(npoints):
        edge1 = edges[i]
        edge2 = edges[i + 1]
        dfbe = df[(df['mc_energy_reco'] >= edge1) & (df['mc_energy_reco'] < edge2)]
        # histogram
        difE = ((dfbe['mc_energy'] - dfbe['mc_energy_reco']) * np.log(10))
        # difE = difE[abs(difE) < 1.5]
        mu, sigma = norm.fit(difE)
        # mu, sigma = norm.fit(difE[abs(difE) < 1.5])
        mus = np.append(mus, mu)
        sigmas = np.append(sigmas, sigma)

    edges = np.power(10, edges)
    bin_centers = np.sqrt((edges[:-1] * edges[1:]))

    return bin_centers, mus, sigmas


def get_theta2_68(df, npoints):

    edges = np.linspace(-2, 2, npoints + 1)
    theta2_68 = np.array([])

    for i in range(npoints):
        edge1 = edges[i]
        edge2 = edges[i + 1]
        dfbe = df[(df['mc_energy_reco'] >= edge1) & (df['mc_energy_reco'] < edge2)]
        theta2 = (dfbe['d_alt'] - dfbe['d_alt_reco']) ** 2 + (dfbe['d_az'] - dfbe['d_az_reco']) ** 2
        # 68% containement computation
        total = len(theta2)
        hist = np.histogram(theta2, bins=1000)
        for k in range(0, len(hist[0]) + 1):
            fraction = np.sum(hist[0][:k]) / total
            if fraction > 0.68:
                # if rf:
                #    print('\nTotal: ', total)
                #    print('0.68 of total:', np.sum(hist[0][:k]))
                #    print('Fraction:', fraction)
                theta2_68 = np.append(theta2_68, hist[1][k])
                break

    # back to linear
    edges = np.power(10, edges)
    bin_centers = np.sqrt((edges[:-1] * edges[1:]))

    return bin_centers, theta2_68


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dirs', type=str, default='', nargs='+', help='Folders that contains test data.', required=True)
    parser.add_argument(
        '--time', type=bool, default='', help='Specify if feed the network with arrival time.', required=False)
    parser.add_argument(
        '--model_sep', type=str, default='', help='Path of the model to load.', required=True)
    parser.add_argument(
        '--model_energy', type=str, default='', help='Path of the model to load.', required=True)
    parser.add_argument(
        '--model_azalt', type=str, default='', help='Path of the model to load.', required=True)

    FLAGS, unparsed = parser.parse_known_args()
    dirs = FLAGS.dirs
    time = FLAGS.time
    class_path = FLAGS.model_sep
    rege_path = FLAGS.model_energy
    altaz_path = FLAGS.model_azalt

    h5files = get_all_files(dirs)
    batch_size = 64

    classifier = load_model(class_path)
    reg_energy = load_model(rege_path)
    reg_direction = load_model(altaz_path)

    print('Building test generator...')
    test_generator = DataGeneratorChain(h5files, batch_size=batch_size, arrival_time=time, shuffle=True)

    # retrieve ground truth
    print('Inference on data...')
    steps_done = 0
    steps = len(test_generator)
    # steps = 2

    enqueuer = OrderedEnqueuer(test_generator, use_multiprocessing=True)
    enqueuer.start(workers=4, max_queue_size=10)
    output_generator = enqueuer.get()

    progbar = Progbar(target=steps)

    table = np.array([]).reshape(0, 9)

    while steps_done < steps:
        generator_output = next(output_generator)
        x, y, intensity, energy, altaz = generator_output

        y_prd = classifier.predict_on_batch(x)
        e_reco = reg_energy.predict_on_batch(x)
        altaz_reco = reg_direction.predict_on_batch(x)

        alt = altaz[:, 0].reshape(altaz.shape[0], 1)
        az = altaz[:, 1].reshape(altaz.shape[0], 1)

        alt_reco = altaz_reco[:, 0].reshape(altaz_reco.shape[0], 1)
        az_reco = altaz_reco[:, 1].reshape(altaz_reco.shape[0], 1)

        batch = np.concatenate((y, y_prd, intensity, energy, e_reco, alt, az, alt_reco, az_reco), axis=1)
        table = np.concatenate((table, batch), axis=0)

        steps_done += 1
        progbar.update(steps_done)

    cols = ['Label', 'gammanes', 'Intensity', 'mc_energy', 'mc_energy_reco', 'd_alt', 'd_az', 'd_alt_reco',
            'd_az_reco']
    df = pd.DataFrame(table, columns=cols)

    with open('lstch_analysis.txt', 'w') as f:
        print('CNN LST Chain - Full analysis\n', file=f)
        print('Separation Network:' + str(class_path) + '\n', file=f)
        print('Energy Network:' + str(rege_path) + '\n', file=f)
        print('Direction Network:' + str(altaz_path) + '\n', file=f)
        print('Number of files in the test set:' + str(len(h5files)) + '\n', file=f)
        print(tabulate(df, headers='keys', tablefmt='psql'), file=f)

    df.to_pickle('lstch_analysis.pkl')

    # print(df)

    # ------------------------- plots --------------------------

    # ROCs
    # fig = plt.figure(figsize=(7, 6))
    #
    # matplotlib.rcParams.update({'font.size': 13})
    # cmap = plt.get_cmap("tab10")
    #
    fpr, tpr, _ = roc_curve(df['Label'], df['gammanes'])
    #
    # plt.plot(fpr, tpr, label='Classifier', color=cmap(0))
    # plt.xlabel('False Positive Rate (1-specitivity)')
    # plt.ylabel('True Positive Rate (sensitivity)')
    # plt.legend(loc='lower right', fancybox=True, framealpha=0.)
    # plt.grid(b=True, which='major', linestyle='--')
    # plt.grid(b=True, which='minor', linestyle='--')
    #
    # plt.tight_layout()
    # plt.show()
    # fig.savefig('class_res.eps', format='eps', bbox_inches='tight', transparent=True)

    ##########

    n_curves = 12

    energy = 'mc_energy_reco'
    edges = np.linspace(-2, 2, n_curves + 1)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

    ax = axs[0]

    cmap = plt.get_cmap("tab10")
    ax.plot(fpr, tpr, label='Classifier', color=cmap(0))
    ax.set_xlabel('False Positive Rate (1-specitivity)')
    ax.set_ylabel('True Positive Rate (sensitivity)')
    ax.legend(loc='lower right', fancybox=True, framealpha=0., prop={'size': 18})
    ax.grid(b=True, which='major', linestyle='--')
    ax.grid(b=True, which='minor', linestyle='--')

    ax = axs[1]
    ax.set_prop_cycle('color', plt.cm.Spectral(np.linspace(0, 1, n_curves)))

    for i in range(n_curves):
        edge1 = edges[i]
        edge2 = edges[i + 1]
        dfbe = df[(df[energy] >= edge1) & (df[energy] < edge2)]
        try:
            # print(i)
            fpr, tpr, _ = roc_curve(dfbe['Label'], dfbe['gammanes'])
            roc_auc = roc_auc_score(dfbe['Label'], dfbe['gammanes'])
            ax.plot(fpr, tpr, label='AUC={:.3f}, [{:.2f}, {:.2f}] TeV'.format(roc_auc, 10 ** edge1, 10 ** edge2))
        except:
            pass

    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.legend(loc='lower right', fancybox=True, framealpha=0., prop={'size': 18})
    ax.set_xlabel('False Positive Rate (1-specitivity)')
    ax.set_ylabel('True Positive Rate (sensitivity)')
    ax.grid(b=True, which='major', linestyle='--')
    ax.grid(b=True, which='minor', linestyle='--')

    fig.tight_layout()
    plt.show()
    fig.savefig('roc_rocs.eps', format='eps', bbox_inches='tight', transparent=True)

    # ################### ENERGY BIAS AND RESOLUTION ####################################

    npoints = 12
    bin_centers, mus, sigmas = get_mus_sigmas(df, npoints)

    fig = plt.figure(figsize=(7, 6))

    matplotlib.rcParams.update({'font.size': 13})
    cmap = plt.get_cmap("tab10")

    plt.semilogx(bin_centers, mus, label='Energy regressor', color=cmap(0), marker='o')
    plt.xlabel('$E_{\mathrm{reco}}$[TeV]')
    plt.ylabel(r'$\Delta E$')

    plt.legend(loc='upper left', fancybox=True, framealpha=0.)
    plt.grid(b=True, which='major', linestyle='--')
    plt.grid(b=True, which='minor', linestyle='--')

    plt.tight_layout()
    fig.savefig('energy_bias.eps', format='eps', bbox_inches='tight', transparent=True)

    # reso

    fig = plt.figure(figsize=(7, 6))

    matplotlib.rcParams.update({'font.size': 13})
    cmap = plt.get_cmap("tab10")

    plt.semilogx(bin_centers, sigmas, label='Energy regressor', color=cmap(0), marker='o')
    plt.ylabel(r'$(\Delta E/E)_{68}$')
    plt.xlabel('$E_{\mathrm{reco}}$[TeV]')
    plt.legend(loc='upper left', fancybox=True, framealpha=0.)
    plt.grid(b=True, which='major', linestyle='--')
    plt.grid(b=True, which='minor', linestyle='--')

    plt.tight_layout()
    fig.savefig('energy_reso.eps', format='eps', bbox_inches='tight', transparent=True)

    # angular resolution
    bin_centers, theta2_68 = get_theta2_68(df, n_curves)

    fig = plt.figure(figsize=(7, 6))

    plt.semilogx(bin_centers, np.sqrt(theta2_68), label='Direction regressor', color=cmap(0), marker='o')
    plt.ylabel(r'$\theta_{68}$ [deg]')
    plt.xlabel('$E_{\mathrm{reco}}$[TeV]')
    plt.legend(loc='upper right', fancybox=True, framealpha=0.)
    plt.grid(b=True, which='major', linestyle='--')
    plt.grid(b=True, which='minor', linestyle='--')

    plt.tight_layout()
    fig.savefig('dirreco_results.eps', format='eps', bbox_inches='tight', transparent=True)

