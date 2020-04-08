import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_squared_error


def test_plots(pkl, feature):
    folder = os.path.dirname(pkl)
    df = pd.read_pickle(pkl)

    # print(df)

    if feature == 'energy':

        n_rows = 6  # how many rows figures
        n_cols = 2  # how many cols figures
        n_figs = n_rows * n_cols

        edges = np.linspace(min(df['GroundTruth']), max(df['GroundTruth']), n_figs + 1)
        mus = np.array([])
        sigmas = np.array([])

        # print('Edges: ', edges)

        fig = plt.figure(figsize=(13, 30))

        plt.suptitle('Histograms - Energy reconstruction', fontsize=30)

        for i in range(n_rows):
            for j in range(n_cols):
                # df with ground truth between edges
                edge1 = edges[n_cols * i + j]
                edge2 = edges[n_cols * i + j + 1]
                # print('\nEdge1: ', edge1, ' Idxs: ', n_cols * i + j)
                # print('Edge2: ', edge2, ' Idxs: ', n_cols * i + j + 1)
                dfbe = df[(df['GroundTruth'] >= edge1) & (df['GroundTruth'] < edge2)]
                # histogram
                subplot = plt.subplot(n_rows, n_cols, n_cols * i + j + 1)
                difE = ((dfbe['GroundTruth'] - dfbe['Predicted']) * np.log(10))
                section = difE[abs(difE) < 1.5]
                mu, sigma = norm.fit(section)
                mus = np.append(mus, mu)
                sigmas = np.append(sigmas, sigma)
                n, bins, patches = plt.hist(difE, 100, density=1, alpha=0.75)
                y = norm.pdf(bins, mu, sigma)
                plt.plot(bins, y, 'r--', linewidth=2)
                plt.xlabel('$(log_{10}(E_{gammas}[TeV])-log_{10}(E_{rec}[TeV]))*log_{N}(10)$', fontsize=10)
                # plt.figtext(0.15, 0.9, 'Mean: ' + str(round(mu, 4)), fontsize=10)
                # plt.figtext(0.15, 0.85, 'Std: ' + str(round(sigma, 4)), fontsize=10)
                plt.title('Energy [' + str(round(edge1, 3)) + ', ' + str(
                    round(edge2, 3)) + '] $log_{10}(E_{gammas}[TeV])$' + ' Mean: ' + str(round(mu, 3)) + ' Std: ' + str(
                    round(sigma, 3)))

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(folder + '/histograms.eps', format='eps', transparent=False)

        fig = plt.figure()

        hE = plt.hist2d(df['GroundTruth'], df['Predicted'], bins=100)
        plt.colorbar(hE[3])
        plt.xlabel('$log_{10}E_{gammas}[TeV]$', fontsize=15)
        plt.ylabel('$log_{10}E_{rec}[TeV]$', fontsize=15)
        plt.plot(df['GroundTruth'], df['GroundTruth'], "-", color='red')

        plt.title('Histogram2D - Energy reconstruction')
        plt.tight_layout()
        plt.savefig(folder + '/histogram2d.eps', format='eps', transparent=False)

        fig = plt.figure()

        # back to linear
        edges = np.power(10, edges)
        bin_centers = (edges[:-1] + edges[1:]) / 2

        plt.semilogx(bin_centers, mus, label='Mean')
        plt.semilogx(bin_centers, sigmas, label='Std')
        plt.grid(which='major')
        plt.legend()
        plt.ylabel(r'$\Delta E, \sigma$', fontsize=15)
        plt.xlabel('$E_{gammas}[TeV]$', fontsize=15)
        plt.title('Energy resolution')
        fig.tight_layout()
        plt.savefig(folder + '/energy_res.eps', format='eps', transparent=False)

        # save energy mus and sigmas
        np.savez(folder + '/mus_sigmas.npz', mus=mus, sigmas=sigmas, bin_centers=bin_centers)

        mae_energy = mean_absolute_error(df['GroundTruth'], df['Predicted'])
        mse_energy = mean_squared_error(df['GroundTruth'], df['Predicted'])
        mape_energy = np.mean(np.abs((df['GroundTruth'] - df['Predicted']) / df['GroundTruth'])) * 100

        # writing summary on file
        f = open(folder + '/mae_mse_mape.txt', 'w')
        f.write('MAE: ' + str(mae_energy) + '\n')
        f.write('MSE: ' + str(mse_energy) + '\n')
        f.write('MAPE: ' + str(mape_energy))
        f.close()

    elif feature == 'xy':

        n_rows = 6  # how many rows figures
        n_cols = 2  # how many cols figures
        n_figs = n_rows * n_cols

        edges = np.linspace(min(df['energy']), max(df['energy']), n_figs + 1)
        theta2_68 = np.array([])

        # print('Edges: ', edges)

        fig = plt.figure(figsize=(13, 30))

        plt.suptitle('Histograms - Direction reconstruction', fontsize=30)

        for i in range(n_rows):
            for j in range(n_cols):
                # df with ground truth between edges
                edge1 = edges[n_cols * i + j]
                edge2 = edges[n_cols * i + j + 1]
                dfbe = df[(df['energy'] >= edge1) & (df['energy'] < edge2)]
                # histogram
                subplot = plt.subplot(n_rows, n_cols, n_cols * i + j + 1)
                theta2 = (dfbe['src_x'] - dfbe['src_x_rec']) ** 2 + (dfbe['src_y'] - dfbe['src_y_rec']) ** 2
                # section = theta2[abs(theta2) < 1.5]
                # mu, sigma = norm.fit(section)
                # 68% containement computation
                # total = np.sum(theta2)
                total = len(theta2)
                # theta2_68 = np.append(theta2_68, np.percentile(theta2, 68))
                hist = np.histogram(theta2, bins=1000)
                for k in range(0, len(hist[0]) + 1):
                    fraction = np.sum(hist[0][:k]) / total
                    if fraction > 0.68:
                        print('\nTotal: ', total)
                        print('0.68 of total:', np.sum(hist[0][:k]))
                        print('Fraction:', fraction)
                        theta2_68 = np.append(theta2_68, hist[1][k])
                        break
                # n, bins, patches = plt.hist(theta2, bins=100, range=(0, hist[1][k]))
                n, bins, patches = plt.hist(theta2, bins=100)
                plt.axvline(hist[1][k], color='r', linestyle='dashed', linewidth=1)
                plt.yscale('log', nonposy='clip')
                # patches[k].set_fc('r')
                # y = norm.pdf(bins, mu, sigma)
                # plt.plot(bins, y, 'r--', linewidth=2)
                plt.xlabel(r'$\theta^{2}(º)$', fontsize=10)
                plt.title(
                    'Energy [' + str(round(edge1, 3)) + ', ' + str(round(edge2, 3)) + '] $log_{10}(E_{gammas}[TeV])$')
                # + ' Mean: ' + str(round(mu, 3)) + ' Std: ' + str(round(sigma, 3)))

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(folder + '/histograms.eps', format='eps', transparent=False)

        fig = plt.figure()

        # back to linear
        edges = np.power(10, edges)
        bin_centers = (edges[:-1] + edges[1:]) / 2

        plt.semilogx(bin_centers, np.sqrt(theta2_68), label='theta2_68')
        plt.grid(which='major')
        # plt.legend()
        plt.ylabel(r'$\sqrt{\theta^2_{68}}(º)$', fontsize=15)
        plt.xlabel('$E_{gammas}[TeV]$', fontsize=15)
        plt.title('Angular resolution')
        fig.tight_layout()
        plt.savefig(folder + '/angular_res.eps', format='eps', transparent=False)

        # save angular resolution
        np.savez(folder + '/ang_reso_plt.npz', sqrttheta268=np.sqrt(theta2_68), bin_centers=bin_centers)

        mae_direction = mean_absolute_error([df['src_x'], df['src_y']],
                                            [df['src_x_rec'], df['src_y_rec']])

        mse_direction = mean_squared_error([df['src_x'], df['src_y']],
                                           [df['src_x_rec'], df['src_y_rec']])

        # writing summary on file
        f = open(folder + '/mae.txt', 'w')
        f.write('MAE: ' + str(mae_direction))
        f.write('\nMSE: ' + str(mse_direction))
        f.close()

    print('Plots done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--pkl', type=str, default='', help='pkl test file.', required=True)
    parser.add_argument(
        '--feature', type=str, default='energy', help='Feature to train/predict.', required=True)

    FLAGS, unparsed = parser.parse_known_args()

    pkl = FLAGS.pkl
    feature = FLAGS.feature

    test_plots(pkl, feature)
