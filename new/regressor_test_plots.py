import argparse
import os
from decimal import Decimal
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from scipy.stats import norm, cauchy, t, nct
from sklearn.metrics import mean_absolute_error, mean_squared_error


def test_plots(pkl):
    #folder = os.path.join(os.path.dirname(pkl), "..")
    folder = os.path.dirname(pkl)
    df = pd.read_pickle(pkl)

    # print(df)



    ######################## HIST 2D ###################################
    fig = plt.figure()
    lim_max = max(max(df['energy_reco']), max(df['energy_true']))
    lim_min = min(min(df['energy_reco']), min(df['energy_true']))
    hE = plt.hist2d(df['energy_true'], df['energy_reco'], bins=50, norm=LogNorm())#, range=[[lim_min,lim_max],[lim_min,lim_max]])
    plt.colorbar(hE[3])
    plt.ylabel('$\log_{10}{E_R}$ [TeV]', fontsize=15)
    plt.xlabel('$\log_{10}{E_T}}$ [TeV]', fontsize=15)
    plt.plot(df['energy_true'], df['energy_true'], linestyle="dotted", color='red')
    plt.grid(b=True, which="minor", linestyle='-')
    plt.title('Migration Matrix')
    plt.tight_layout()
    plt.savefig(folder + '/histogram2d.eps', format='eps', transparent=False)



    ############## HISTOGRAMS ########################
    n_rows = 2 # how many rows figures
    n_cols = 2  # how many cols figures
    n_figs = n_rows * n_cols
    #edges = np.linspace(min(df['energy_reco']), max(df['energy_true']), n_figs + 1)
    edges = np.array([-2.,-1.,0.,1.,2.])
    edges = np.power(10, edges)
    df['energy_true'] = np.power(10, df['energy_true'])
    df['energy_reco'] = np.power(10, df['energy_reco'])
    mus = np.array([]) # saranno le medie del fit gaussiano
    sigmas = np.array([]) # saranno le sigma del fit gaussiano

    # print('Edges: ', edges)

    fig = plt.figure(figsize=(32, 24))

    plt.suptitle('Histograms - Energy reconstruction', fontsize=30)
    for i in range(n_rows):
        for j in range(n_cols):
            # df with ground truth between edges
            edge1 = edges[n_cols * i + j]
            edge2 = edges[n_cols * i + j + 1]
            # print('\nEdge1: ', edge1, ' Idxs: ', n_cols * i + j)
            # print('Edge2: ', edge2, ' Idxs: ', n_cols * i + j + 1)
            dfbe = df[(df['energy_true'] >= edge1) & (df['energy_true'] < edge2)]
            # histogram
            subplot = plt.subplot(n_rows, n_cols, n_cols * i + j + 1)
            difE = (1 - dfbe['energy_reco']/dfbe['energy_true'])
            xrange = (-2,2)
            if i == 0 and j == 0:
                xrange=(-6,6)
                section = difE[(difE > -4.0) & (difE < 0)]
            elif i ==0 and j ==1:
                section = difE[ (difE > -1.5) & (difE < 0.5)]
            elif i ==0 and j ==2:
                section = difE[ (difE > -1.5) & (difE < 1)]
            elif i==1 and j == 0:
                section = difE[(difE > -0.8) & (difE < 0.8)]
            elif i==1:
                section = difE[(difE > -0.7)& (difE < 0.7)]
            elif i==2:
                section = difE[(difE > -0.4) & (difE < 0.55)]
            else:
                section = difE[(difE > -0.55) & (difE < 0.65)]
            n, bins, patches = plt.hist(difE, 100, density=1, alpha=0.75, range=xrange)
            '''
            if i !=0 or j !=0:
                mu, sigma= norm.fit(section)
                mus = np.append(mus, mu)
                sigmas = np.append(sigmas, sigma)
                y = norm.pdf(bins, mu, sigma)
                plt.plot(bins, y, 'r--', linewidth=2)
            '''

            if i==3:
                plt.xlabel('$( E_{TRUE} - E_{EST} ) / E_{TRUE}$', fontsize=24)
            # plt.figtext(0.15, 0.9, 'Mean: ' + str(round(mu, 4)), fontsize=10)
            # plt.figtext(0.15, 0.85, 'Std: ' + str(round(sigma, 4)), fontsize=10)
            plt.title(  "{:.1e} TeV < E_est < {:.1e} TeV".format(edge1, edge2), fontsize=24)#  Decimal(str(np.power(10, edge1))), Decimal(str(np.power(10, edge2)))  ), fontsize=20   )

    #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(folder + '/histograms_energy.eps', format='eps', transparent=False)

    ##################### ENERGY RESOLUTION ############################
    '''
    fig = plt.figure()
    # back to linear
    #edges = np.power(10, edges)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    #plt.semilogx(bin_centers, mus, label='Biases', marker='o')
    plt.semilogx(bin_centers[1:], sigmas, label='Energy resolution', marker='o')
    plt.grid(b=True, which='major', linestyle='--')
    plt.grid(b=True, which='minor', linestyle='--')
    plt.legend()
    plt.ylabel(r'$\Delta E / E$', fontsize=15)
    plt.xlabel('$E_{TRUE} \t [TeV]$', fontsize=15)
    plt.title('Energy resolution')
    fig.tight_layout()
    plt.savefig(folder + '/energy_res.eps', format='eps', transparent=False)

    # save energy mus and sigmas
    np.savez(folder + '/mus_sigmas.npz', mus=mus, sigmas=sigmas, bin_centers=bin_centers)

    mae_energy = mean_absolute_error(df['energy_true'], df['energy_reco'])
    mse_energy = mean_squared_error(df['energy_true'], df['energy_reco'])
    mape_energy = np.mean(np.abs((df['energy_true'] - df['energy_reco']) / df['energy_true'])) * 100

    # writing summary on file
    f = open(folder + '/mae_mse_mape.txt', 'w')
    f.write('MAE: ' + str(mae_energy) + '\n')
    f.write('MSE: ' + str(mse_energy) + '\n')
    f.write('MAPE: ' + str(mape_energy))
    f.close()
    '''

    n_rows = 1 # how many rows figures
    n_cols = 1  # how many cols figures
    n_figs = n_rows * n_cols

    edges = np.logspace(np.log10(min(df['energy_true'])), np.log10(max(df['energy_true'])), n_figs + 1)
    theta2_68 = np.array([])

    # print('Edges: ', edges)

    fig = plt.figure(figsize=(10, 7))

    #plt.suptitle('Histograms - Direction reconstruction', fontsize=30)

    for i in range(n_rows):
        for j in range(n_cols):
            # df with ground truth between edges
            edge1 = edges[n_cols * i + j]
            edge2 = edges[n_cols * i + j + 1]
            print(df)
            print(min(df['energy_reco']))
            dfbe = df[(df['energy_reco'] >= edge1) & (df['energy_reco'] < edge2)]
            # histogram
            subplot = plt.subplot(n_rows, n_cols, n_cols * i + j + 1)
            theta2 = (dfbe['d_alt_true'] - dfbe['d_alt_reco']) ** 2 + (dfbe['d_az_true'] - dfbe['d_az_reco']) ** 2
            # section = theta2[abs(theta2) < 1.5]
            # mu, sigma = norm.fit(section)
            # 68% containement computation
            # total = np.sum(theta2)
            total = len(theta2)
            # theta2_68 = np.append(theta2_68, np.percentile(theta2, 68))
            hist = np.histogram(theta2, bins=10000)
            for k in range(0, len(hist[0]) + 1):
                fraction = np.sum(hist[0][:k]) / total
                if fraction > 0.68:
                    print('\nTotal: ', total)
                    print('0.68 of total:', np.sum(hist[0][:k]))
                    print('Fraction:', fraction)
                    theta2_68 = np.append(theta2_68, hist[1][k])
                    break
            # n, bins, patches = plt.hist(theta2, bins=100, range=(0, hist[1][k]))
            #plt.semilogx()
            n, bins, patches = plt.hist(theta2, bins=250, range=(0,1),edgecolor='gray', linewidth=0.1)
            #n, bins, patches = plt.hist(theta2, bins=250, edgecolor='gray', linewidth=0.1)
            plt.axvline(hist[1][k], color='r', linestyle='dashed', linewidth=1, label=r'68th percentile: $\theta^2$ = {:.3f} deg$^2$'.format(hist[1][k]))

            #plt.yscale('log', nonposy='clip')
            # patches[k].set_fc('r')
            # y = norm.pdf(bins, mu, sigma)
            # plt.plot(bins, y, 'r--', linewidth=2)
            plt.grid(linestyle='--')
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.xlabel(r'$\theta^{2}$ [deg$^2$]', fontsize=20)
            plt.ylabel('Counts', fontsize=20)
            plt.title("Mid-cuts, Full Range: {:.0f} GeV - {:.0f} TeV".format(edge1*1000, edge2), fontsize=22)
            plt.legend(fontsize=16)
            #plt.xticks([0,hist[1][k], 0.2, 0.4, 0.6, 0.8, 1.0])
            #plt.title(
            #    'Energy [' + str(round(edge1, 3)) + ', ' + str(round(edge2, 3)) + '] $log_{10}(E_{gammas}[TeV])$')
            # + ' Mean: ' + str(round(mu, 3)) + ' Std: ' + str(round(sigma, 3)))

    #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(folder, 'histograms.eps'), format='eps')

    fig = plt.figure()

    # back to linear
    #edges = np.power(10, edges)
    #bin_centers = (edges[:-1] + edges[1:]) / 2
    bin_centers = np.sqrt((edges[:-1] * edges[1:]))
    plt.semilogx(bin_centers, np.sqrt(theta2_68), linewidth=0.5, label='theta2_68', marker='.')
    plt.grid(which='major')
    # plt.legend()
    plt.ylabel(r'$\sqrt{\theta^2_{68}}[deg]$', fontsize=15)
    plt.xlabel('$E_{TRUE}[TeV]$', fontsize=15)
    plt.title('Angular resolution')
    fig.tight_layout()
    plt.savefig(folder + '/angular_res.eps', format='eps', transparent=False)

    # save angular resolution
    np.savez(folder + '/ang_reso_plt.npz', sqrttheta268=np.sqrt(theta2_68), bin_centers=bin_centers)

    mae_direction = mean_absolute_error([df['d_alt_true'], df['d_az_true']],
                                        [df['d_alt_reco'], df['d_az_reco']])

    mse_direction = mean_squared_error([df['d_alt_true'], df['d_az_true']],
                                       [df['d_alt_reco'], df['d_az_reco']])

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

    FLAGS, unparsed = parser.parse_known_args()

    pkl = FLAGS.pkl

    test_plots(pkl)
