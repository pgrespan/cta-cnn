from os import listdir
from os.path import isfile, join
import subprocess

path_int_train = "/mnt/data2t/nmarinel_2020.01.10/simulations/proton/train/"
path_int_val = "/mnt/data2t/nmarinel_2020.01.10/simulations/proton/val/"
path_org_train = "/mnt/data2t/CTA_DATA_EXT/protons/training/"
path_org_test = "/mnt/data2t/CTA_DATA_EXT/protons/test/"

path_to_test = "/home/pgrespan/simulations/proton/test/"

interp = listdir(path_int_train) + listdir(path_int_val)
original = [path_org_train, path_org_test]
#original = listdir(path_org_train) + listid(path_org_test)

missing = []
for path in original:
    miss = [join(path,f) for f in listdir(path) if f[:-3]+'_interp.h5' not in interp]
    missing += miss
print('# of already interpolated files: {}'.format(len(interp)))
print( '# of original files: {}'.format(len( listdir(path_org_test)+listdir(path_org_train) )) )
print('# of not yet interpolated files: {}'.format(len(missing)))
#print(missing)

check = input('Proceed? [y/n] ')
if (check is 'y'):
    for f in missing:
        status = subprocess.call('cp {} {}'.format(f, path_to_test), shell=True)
elif (check is 'n'):
    print('Not copying. Exiting...')
else:
    print('No valid option selected. Aborting.')