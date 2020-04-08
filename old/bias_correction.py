import argparse

import numpy as np
import pandas as pd
from keras.models import load_model
from keras.utils.data_utils import OrderedEnqueuer
import matplotlib
from scipy.stats import norm
from keras.utils.generic_utils import Progbar
from scipy import interpolate
import matplotlib.pyplot as plt

from generators import DataGeneratorChain
from utils import get_all_files


def get_mus_sigmas(df, npoints, energy_reco='mc_energy_reco'):

    energy = 'mc_energy'

    edges = np.linspace(min(df[energy]), 2, npoints + 1)
    mus = np.array([])
    sigmas = np.array([])

    for i in range(npoints):
        edge1 = edges[i]
        edge2 = edges[i + 1]
        dfbe = df[(df[energy] >= edge1) & (df[energy] < edge2)]
        # histogram
        difE = ((dfbe[energy] - dfbe[energy_reco]) * np.log(10))
        mu, sigma = norm.fit(difE)
        mus = np.append(mus, mu)
        sigmas = np.append(sigmas, sigma)

    edges = np.power(10, edges)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    return bin_centers, mus, sigmas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--val_dirs', type=str, default='', nargs='+', help='Folders that contains test data.', required=True)
    parser.add_argument(
        '--test_dirs', type=str, default='', nargs='+', help='Folders that contains test data.', required=True)
    parser.add_argument(
        '--model_energy', type=str, default='', help='Path of the model to load.', required=True)

    FLAGS, unparsed = parser.parse_known_args()
    val_dirs = FLAGS.val_dirs
    test_dirs = FLAGS.test_dirs
    rege_path = FLAGS.model_energy

    val_h5files = get_all_files(val_dirs)
    test_h5files = get_all_files(test_dirs)

    # try to use hal of the test files for bias estimation and the other half to correct
    val_h5files = test_h5files[:int(len(test_h5files)/2)]
    test_h5files = test_h5files[int(len(test_h5files) / 2):]
    ######################################################################################

    reg_energy = load_model(rege_path)

    batch_size = 128

    print('Building val & test generator...')
    validation_generator = DataGeneratorChain(val_h5files, batch_size=batch_size, arrival_time=True, shuffle=True)
    test_generator = DataGeneratorChain(test_h5files, batch_size=batch_size, arrival_time=True, shuffle=True)

    print('Inference on validation data...')
    steps_done = 0
    steps = len(validation_generator)
    # steps = 200

    enqueuer = OrderedEnqueuer(validation_generator, use_multiprocessing=True)
    enqueuer.start(workers=4, max_queue_size=10)
    output_generator = enqueuer.get()

    progbar = Progbar(target=steps)

    table = np.array([]).reshape(0, 2)

    while steps_done < steps:
        generator_output = next(output_generator)
        x, _, _, energy, _ = generator_output

        e_reco = reg_energy.predict_on_batch(x)

        batch = np.concatenate((energy, e_reco), axis=1)
        table = np.concatenate((table, batch), axis=0)

        steps_done += 1
        progbar.update(steps_done)

    cols = ['mc_energy', 'mc_energy_reco']
    df_val = pd.DataFrame(table, columns=cols)

    npoints = 14
    regressor_bc, regressor_m, regressor_s = get_mus_sigmas(df_val, npoints)

    # linear interpolate the bias to get the its function
    f = interpolate.interp1d(regressor_bc, regressor_m, fill_value='extrapolate')

    # plot estimated bias
    fig = plt.figure(figsize=(7, 6))
    matplotlib.rcParams.update({'font.size': 13})
    cmap = plt.get_cmap("tab10")

    plt.errorbar(regressor_bc, regressor_m, yerr=regressor_s, label='Regressor', color=cmap(0), marker='o')
    # plt.plot(regressor_bc, f(regressor_bc), label='Interpolated', color=cmap(1), marker='1')
    plt.xscale('log', nonposx='clip')
    plt.xlabel('$E_{gammas}[TeV]$')
    plt.ylabel(r'$\Delta E$')
    plt.legend(loc='upper left', fancybox=True, framealpha=0.)
    plt.grid(b=True, which='major', linestyle='--')
    plt.grid(b=True, which='minor', linestyle='--')

    plt.tight_layout()
    fig.savefig('estimated_bias.eps', format='eps', transparent=False)

    print('Inference on test data...')
    steps_done = 0
    steps = len(test_generator)
    # steps = 200

    enqueuer = OrderedEnqueuer(test_generator, use_multiprocessing=True)
    enqueuer.start(workers=4, max_queue_size=10)
    output_generator = enqueuer.get()

    progbar = Progbar(target=steps)

    table = np.array([]).reshape(0, 3)

    while steps_done < steps:
        generator_output = next(output_generator)
        x, _, _, energy, _ = generator_output

        e_reco = reg_energy.predict_on_batch(x)

        # bias estimation and correction
        b = f(10**e_reco)
        e_reco_corr = e_reco+np.log10(b)

        e_reco_corr = np.log10(10**e_reco + b * 10**e_reco)

        batch = np.concatenate((energy, e_reco, e_reco_corr), axis=1)
        table = np.concatenate((table, batch), axis=0)

        steps_done += 1
        progbar.update(steps_done)

    cols = ['mc_energy', 'mc_energy_reco', 'mc_energy_reco_corr']
    df_test = pd.DataFrame(table, columns=cols)

    npoints = 14
    regressor_bc, regressor_m, regressor_s = get_mus_sigmas(df_test, npoints)
    regressor_bc_c, regressor_m_c, regressor_s_c = get_mus_sigmas(df_test, npoints, energy_reco='mc_energy_reco_corr')

    fig = plt.figure(figsize=(7, 6))

    matplotlib.rcParams.update({'font.size': 13})
    cmap = plt.get_cmap("tab10")

    plt.semilogx(regressor_bc, regressor_m, label='Bias', color=cmap(0), marker='o')
    plt.semilogx(regressor_bc_c, regressor_m_c, label='BiasC', color=cmap(1), marker='o')
    plt.xlabel('$E_{gammas}[TeV]$')
    plt.ylabel(r'$\Delta E$')

    plt.legend(loc='upper left', fancybox=True, framealpha=0.)
    plt.grid(b=True, which='major', linestyle='--')
    plt.grid(b=True, which='minor', linestyle='--')

    plt.tight_layout()
    fig.savefig('energy_bias_corrected.eps', format='eps', transparent=False)
    ########################################################################################################
    # reso & improvement
    fig = plt.figure(figsize=(7, 6))

    plt.semilogx(regressor_bc, regressor_s, label='Regressor', color=cmap(0), marker='o')
    plt.xlabel('$E_{\mathrm{gammas}}$[TeV]')
    plt.ylabel(r'$(\Delta E/E)_{68}$')
    plt.legend(loc='upper left', fancybox=True, framealpha=0.)
    plt.grid(b=True, which='major', linestyle='--')
    plt.grid(b=True, which='minor', linestyle='--')

    plt.tight_layout()
    fig.savefig('energy_reso_corrected.eps', format='eps', transparent=False)

