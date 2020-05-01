from multiprocessing import Process
from os import listdir, remove
from os.path import isfile, join
import tables
import h5py
from tables.exceptions import HDF5ExtError, NoSuchNodeError
import argparse
import multiprocessing as mp


def get_LST_data(data):

    data_LST = data.root.LST_LSTCam

    # LST data
    # TODO: compute intensity and leakage from original (not interp) data
    LST_event_index = [x['event_index'] for x in data_LST.iterrows()]
    num_events = {'tot':len(LST_event_index) - 1, 'cut':0}
    return num_events

def chunkit(seq, num):

    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def worker(h5files, i, return_dict, interp, threshold):

    lengths = {'tot':0, 'cut':0}

    if interp:
        for l, f in enumerate(h5files):
            h5f = h5py.File(f, 'r')
            e_intensity = h5f['LST/intensities'][1:]
            e_leakage = h5f['LST/intensities_width_2'][1:]
            lst_index = h5f['LST/LST_event_index'][1:]
            lengths['tot'] += len(lst_index)
            lengths['cut'] += len(lst_index[(e_intensity >= threshold['int']) & (e_leakage <= threshold['lkg'])])
            h5f.close()
    else:
        for l, f in enumerate(h5files):
            # get the data from the file
            try:
                data_p = tables.open_file(f)
                LST_event_len = get_LST_data(data_p)

                for key in lengths: lengths[key] += LST_event_len[key]

            except HDF5ExtError:

                print('\nUnable to open file' + f)

            except NoSuchNodeError:

                print('\nThis file has a problem with the data structure: ' + f)

    return_dict[i] = lengths


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dirs', type=str, default='', nargs='+', help='Folder that contain .h5 files.', required=True)
    parser.add_argument(
        '--interp', type=bool, default=False, help='Specify if count on interpolated files or not.', required=False)
    parser.add_argument(
        '--int', type=float, default='0', help='Set a lower limit (threshold) on event intensity.', required=False)
    parser.add_argument(
        '--lkg', type=float, default='1', help='Set an upper limit to leakage on intensity.', required=False)

    FLAGS, unparsed = parser.parse_known_args()

    ncpus = mp.cpu_count()
    print("\nNumber of CPUs: " + str(ncpus))

    # get all the parameters given by the command line
    folders = FLAGS.dirs
    interp = FLAGS.interp
    threshold = {'int':FLAGS.int, 'lkg':FLAGS.lkg}

    print('Interpolated', FLAGS.interp)

    print('Folders: ' + str(folders) + '\n')

    print('Min. event intensity:', threshold['int'])

    print('Max. event leakage:', threshold['lkg'])

    # create a single big list containing the paths of all the files
    all_files = []

    if interp:
        ew = "_interp.h5"
        print('\nTrying to find interpolated files...')
    else:
        ew = ".h5"
        print('Trying to find not interpolated files...')

    for path in folders:
        files = [join(path, f) for    f in listdir(path) if (isfile(join(path, f)) and f.endswith(ew))]
        all_files = all_files + files

    # print('Files: ' + '\n' + str(all_files) + '\n')

    num_files = len(all_files)

    processes = []

    manager = mp.Manager()
    return_dict = manager.dict()

    if ncpus >= num_files:
        print('ncpus >= num_files')
        for i, f in enumerate(all_files):
            p = Process(target=worker, args=([f], i, return_dict, interp, threshold))
            p.start()
            processes.append(p)
    else:
        print('ncpus < num_files')
        c = chunkit(all_files, ncpus)
        for i, f in enumerate(c):
            p = Process(target=worker, args=(f, i, return_dict, interp, threshold))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()

    #print(return_dict)

    print('Number of files: ' + str(num_files))
    tot_len = 0
    cut_len = 0
    for k, v in return_dict.items():
        tot_len += v['tot']
        cut_len += v['cut']

    print('\nNumber of events: ' + str(tot_len))
    if interp:
        print('Number of selected events: {} ({:.1f}%)\n'.format(cut_len, cut_len/tot_len*100))


