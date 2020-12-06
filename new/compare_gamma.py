from os import listdir
from os.path import isfile, join
import subprocess

path_int_train = "/mnt/data2t/nmarinel_2020.01.10/simulations/gamma/train/"
path_int_val = "/mnt/data2t/nmarinel_2020.01.10/simulations/gamma/val/"
path_org_train = "/mnt/data2t/interp_CTA_DATA/gamma_diffuse/training/"
path_org_test = "/mnt/data2t/interp_CTA_DATA/gamma_diffuse/test/"

path_to_test = "/home/pgrespan/simulations/gamma/test/"

interp = listdir(path_int_train) + listdir(path_int_val)
original = [path_org_train, path_org_test]
#original = listdir(path_org_train) + listid(path_org_test)

missing = []
for path in original:
    miss = [join(path,f) for f in listdir(path) if f not in interp]
    missing += miss
print('# of train/ and val/ files: {}'.format(len(interp)))
print( '# of total interpolated gamma diffuse: {}'.format(len( listdir(path_org_test)+listdir(path_org_train) )) )
print('# of not included files: {}'.format(len(missing)))
#print(missing)

check = input('Proceed? [y/n] ')
if (check is 'y'):
    for i in range(13):
        print('Copying {}th file...'.format(i))
        status = subprocess.call('cp {} {}'.format(missing[i], path_to_test), shell=True)
elif (check is 'n'):
    print('Not copying. Exiting...')
else:
    print('No valid option selected. Aborting.')