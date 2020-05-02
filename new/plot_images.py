#from os.path import join
import numpy as np
import argparse
from utils import get_all_files, save_interp_images
from generators import DataGeneratorR
from tqdm import trange
def main(input, output, emin, emax, log_energy, its, lkg):

    files = get_all_files(input)
    print("Indexing data...")
    generator = DataGeneratorR(files, feature="energy", shuffle=False, intensity=its, emin=emin, emax=emax)
    print("Loading data and printing images...")
    title = "Gamma"
    for i in trange(generator.__len__()):
        images, energies = generator.__getitem__(i)
        save_interp_images(charges=images, title=title, energies=energies, start=i, outputdir=output)
    print("Finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--inputdir', type=str, default='', nargs='+', help='Folders containing data.', required=True)
    parser.add_argument(
        '-o', '--outputdir', type=str, default='SameAsInput', help='Folder where to store output plots.', required=False)
    parser.add_argument(
        '--emin', type=float, default=-5, help='Lower energy threshold (in log10 if --log activated).', required=False)
    parser.add_argument(
        '--emax', type=float, default=5, help='Upper energy threshold (in log10 if --log activated).', required=False)
    parser.add_argument(
        '--log', help='Activate if you set the energy threshold options above in log10 scale.', action="store_true")
    parser.add_argument(
        '--intensity', type=float, default=200, help='Intensity cut (events with intensity UNDER the set value are rejected). Default: 200', required=False)
    parser.add_argument(
        '--leakage2', type=float, default=0.2, help='Leakage2 cut (events with Leakage2 ABOVE the set value are rejected). Default: 0.2', required=False)

    FLAGS, unparsed = parser.parse_known_args()

    input = FLAGS.inputdir
    output = FLAGS.outputdir
    emin = FLAGS.emin
    emax = FLAGS.emax
    log_energy = FLAGS.log
    intensity = FLAGS.intensity
    leakage2 = FLAGS.leakage2

    main(input, output, emin, emax, log_energy, intensity, leakage2)