from os import listdir
from os.path import isfile, join
import numpy as np
from ctapipe.instrument import CameraGeometry
import matplotlib.pyplot as plt

def get_all_files(folders):

    all_files = []

    for path in folders:
        files = [join(path, f) for f in listdir(path) if (isfile(join(path, f)) and f.endswith("_interp.h5"))]
        all_files = all_files + files

    return all_files

def save_interp_images(charges, title, energies, outputdir, start=0):

    charges = np.array(charges) # if charges is a list -> convert to np.array
    cam = CameraGeometry.from_name('LSTCam')
    x = cam.pix_x
    y = cam.pix_y
    points = np.array([np.array(x),
                       np.array(y)]).T
    path_fig = outputdir
    rows = charges.shape[1]
    cols = charges.shape[2]
    charges = charges.reshape(-1, rows, cols)
    for i, interp_crg in enumerate(charges):
        index = start*len(charges)+i
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set(title= title + " - MC energy: {:.1f}".format(energies[i]), xlabel="x [m]", ylabel="y [m]")
        a = plt.imshow(interp_crg.T, extent=(points.min(), points.max(), points.min(), points.max()), origin='lower')
        name = join(path_fig, title + "_" + str(index) + "_MC-energy_{:.1f}.png".format(energies[i]))
        b = plt.savefig(fname=name)
        c = plt.clf()
        d = plt.close(fig)

    print("Plots stored in folder: ", path_fig)