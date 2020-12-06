import numpy as np
import uproot  # you can replace
import matplotlib.pyplot as plt
import glob
import argparse
import pandas

parser = argparse.ArgumentParser(description='make lc comparison plot')
parser.add_argument('-d','--directory',help='directory where output files exist')
args=parser.parse_args()

#plt.style.use('/Users/ssakurai/Research/Backup/MAGIC/magicsoft/mymultigraph.mplstyle')
def main():
    filenames = glob.glob('{0}/Output*.root'.format(args.directory))  # you can replace where you put the files.
    print(filenames)
    fig = plt.figure(figsize=(8,6))
    #ax3=fig.add_subplot(311)
    ax1=fig.add_subplot(211)
    ax2=fig.add_subplot(212)
    mjd_array = []
    flux_array = []
    flux_e_array = []
    db_array = []
    for i,j in enumerate(sorted(filenames)):
        tmp_filename = j
        ##### HERE #####
        in_file = uproot.open(tmp_filename)  # you can replace the method which load flux values.
        #print(type(in_file))
        light_curve = in_file['LightCurve']
        #print(type(light_curve._fX[0]))
        fx_array = light_curve._fX # the array of MJD
        fy_array = light_curve._fY # the array of  Flux
        fey_array = light_curve._fEY # the  array of Flux Error
        ##### UNTIL HERE #####
        mjd_array.append(fx_array)
        flux_array.append(fy_array)
        flux_e_array.append(fey_array)
        ax1.errorbar(fx_array,fy_array,yerr=fey_array,fmt='o--',label=j[j.rfind("Az"):],capsize=5)
        #ax3.errorbar(fx_array,fy_array/np.max(fy_array),yerr=fey_array/np.max(fy_array),fmt='o--',label=j,capsize=5)
        db_tmp = pandas.DataFrame({
            'time':fx_array,
            'flux':fy_array
        })
        db_array.append(db_tmp)

    ax1.set_ylabel('Flux (>300 GeV)\n [cm$^{-2}$s$^{-1}$]', fontsize=12)
    ax1.set_xlabel('MJD', horizontalalignment='right', x=1.0, fontsize=12)
    ax1.grid(linestyle='--')
    #ax1.set_xlim(57300,57350)
    #ax3.set_ylabel('Normalized Flux\nby its maximum')
    #ax3.set_xlim(57300,57350)
    #ax1.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)

    #print(db_array[0]['time'])

    mjds_unique = None
    tmp_array = np.array([])
    for k,l in enumerate(mjd_array):
        tmp_array = np.concatenate((tmp_array,l))
    mjds_unique = np.unique(np.sort(tmp_array))
    length = len(filenames) -1
    for m in range(length):
        for n in range(length - m):
            x_siml = []
            ratio_siml = []
            ratio_e_siml = []
            for i,mjd in enumerate(mjds_unique):
                gets1 = [k for k,h in enumerate(mjd_array[m]) if abs(h-mjd)<0.05]
                gets2 = [k for k,h in enumerate(mjd_array[n]) if abs(h-mjd)<0.05]
                if len(gets1) * len(gets2):
                    t1=mjd_array[m][gets1[0]]
                    t2=mjd_array[n][gets2[0]]
                    f2=flux_array[n][gets2[0]]
                    f1=flux_array[m][gets1[0]]
                    fe2=flux_e_array[n][gets2[0]]
                    fe1=flux_e_array[m][gets1[0]]
                    t = (t1 + t2)/2.0
                    f_ratio = abs(f2-f1)
                    #f_ratio_e = np.sqrt((fe2/f2)**2+(fe1/f1)**2)*f_ratio
                    f_ratio_e = np.sqrt((fe2)**2+(fe1)**2)
                    x_siml.append(t)
                    ratio_siml.append(f_ratio/fe1)
                    ratio_e_siml.append(f_ratio_e/fe1)
            ax2.errorbar(x_siml,ratio_siml,yerr=ratio_e_siml,fmt='o',color='C{}'.format(m+1),capsize=5)

    ax2.plot([min(mjd_array[0]),max(mjd_array[1])],[1.0,1.0],'b--')

    ax2.set_xlabel('MJD', horizontalalignment='right', x=1.0, fontsize=12)
    ax2.set_ylabel('Flux ratio', fontsize=12)
    ax2.grid(linestyle='--')
    #ax2.set_ylim(0.8,1.2)
    #ax2.set_xlim(57300,57350)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()