import h5py
import numpy as np
import pickle
import random
from tqdm import tqdm
from os.path import join

output_folder = "/home/pgrespan/simulations/small_test/"

def acquire_data(file_list, label):
    train_data = []
    for f in file_list:
        h5f = h5py.File(f, 'r')
        i = h5f['LST/intensities'][:]
        l = h5f['LST/intensities_width_2'][:]
        for j, x in enumerate(tqdm(h5f['LST/LST_image_charge_interp'])):
            train_data.append([x, int(label), i[j], l[j]])
        h5f.close()
    return train_data

gfiles = [
    ['/home/pgrespan/simulations/gamma_diff/train/gamma-diffuse_20deg_180deg_runs131-140___cta-prod3-demo-2147m-LaPalma-baseline-mono_interp.h5'],
    ['/home/pgrespan/simulations/gamma_diff/train/gamma-diffuse_20deg_180deg_runs141-150___cta-prod3-demo-2147m-LaPalma-baseline-mono_interp.h5']]

pfiles = [
    ['/home/pgrespan/simulations/proton/train/proton_20deg_180deg_runs4341-4350___cta-prod3-demo-2147m-LaPalma-baseline-mono_interp.h5',
    '/home/pgrespan/simulations/proton/train/proton_20deg_180deg_runs441-450___cta-prod3-demo-2147m-LaPalma-baseline-mono_interp.h5',
    '/home/pgrespan/simulations/proton/train/proton_20deg_180deg_runs4441-4450___cta-prod3-demo-2147m-LaPalma-baseline-mono_interp.h5',
    '/home/pgrespan/simulations/proton/train/proton_20deg_180deg_runs4451-4460___cta-prod3-demo-2147m-LaPalma-baseline-mono_interp.h5',
    '/home/pgrespan/simulations/proton/train/proton_20deg_180deg_runs4481-4490___cta-prod3-demo-2147m-LaPalma-baseline-mono_interp.h5',
    '/home/pgrespan/simulations/proton/train/proton_20deg_180deg_runs4501-4510___cta-prod3-demo-2147m-LaPalma-baseline-mono_interp.h5',
    '/home/pgrespan/simulations/proton/train/proton_20deg_180deg_runs451-1000___cta-prod3-demo-2147m-LaPalma-baseline-mono_interp.h5',
    '/home/pgrespan/simulations/proton/train/proton_20deg_180deg_runs4521-4530___cta-prod3-demo-2147m-LaPalma-baseline-mono_interp.h5',
    '/home/pgrespan/simulations/proton/train/proton_20deg_180deg_runs4531-4540___cta-prod3-demo-2147m-LaPalma-baseline-mono_interp.h5',
    '/home/pgrespan/simulations/proton/train/proton_20deg_180deg_runs4571-4580___cta-prod3-demo-2147m-LaPalma-baseline-mono_interp.h5'],

    ['/home/pgrespan/simulations/proton/train/proton_20deg_180deg_runs1451-1460___cta-prod3-demo-2147m-LaPalma-baseline-mono_interp.h5',
    '/home/pgrespan/simulations/proton/train/proton_20deg_180deg_runs1481-1490___cta-prod3-demo-2147m-LaPalma-baseline-mono_interp.h5',
    '/home/pgrespan/simulations/proton/train/proton_20deg_180deg_runs1491-1500___cta-prod3-demo-2147m-LaPalma-baseline-mono_interp.h5',
    '/home/pgrespan/simulations/proton/train/proton_20deg_180deg_runs1541-1550___cta-prod3-demo-2147m-LaPalma-baseline-mono_interp.h5',
    '/home/pgrespan/simulations/proton/train/proton_20deg_180deg_runs1551-1560___cta-prod3-demo-2147m-LaPalma-baseline-mono_interp.h5',
    '/home/pgrespan/simulations/proton/train/proton_20deg_180deg_runs1561-1570___cta-prod3-demo-2147m-LaPalma-baseline-mono_interp.h5',
    '/home/pgrespan/simulations/proton/train/proton_20deg_180deg_runs1581-1590___cta-prod3-demo-2147m-LaPalma-baseline-mono_interp.h5',
    '/home/pgrespan/simulations/proton/train/proton_20deg_180deg_runs1601-1610___cta-prod3-demo-2147m-LaPalma-baseline-mono_interp.h5',
    '/home/pgrespan/simulations/proton/train/proton_20deg_180deg_runs1611-1620___cta-prod3-demo-2147m-LaPalma-baseline-mono_interp.h5']]


print("Acquiring data...")
training_data = [[],[]]
for e, lixt in enumerate(training_data):
    lixt += acquire_data(gfiles[e], 1)
    lixt += acquire_data(pfiles[e], 0)
    random.shuffle(lixt)
#print("Training data element: ", training_data[100])
print("Training data 0 length: ", len(training_data[0]))
print("Training data 1 length: ", len(training_data[1]))

# create tensors for keras
IMG_SIZE = 100
X = [[],[]]
y = [[],[]]
intens = [[],[]]
lkg = [[],[]]

print("Saving data...")
for e, lixt in enumerate(tqdm(training_data)):
    for features,label,i,l in tqdm(lixt):
        X[e].append(features)
        y[e].append(label)
        intens[e].append(i)
        lkg[e].append(l)
    X[e] = np.array(X[e]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y[e] = np.array(y[e])
    intens[e] = np.array(intens[e])
    lkg[e] = np.array(lkg[e])
    np.save(file=join(output_folder, "data"+str(e), "images"+str(e)), arr=X[e])#, allow_pickle=False)
    np.save(file=join(output_folder, "data"+str(e), "labels"+str(e)), arr=y[e])#, allow_pickle=False)
    np.save(file=join(output_folder, "data"+str(e), "intens"+str(e)), arr=intens[e])#, allow_pickle=False)
    np.save(file=join(output_folder, "data"+str(e), "leakg2"+str(e)), arr=lkg[e])#, allow_pickle=False)

# save data
'''
pickle_out = open(join(output_folder, "images1.pickle"),"wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open(join(output_folder,"labels1.pickle"),"wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_out = open(join(output_folder,"intens1.pickle"),"wb")
pickle.dump(intens, pickle_out)
pickle_out.close()

pickle_out = open(join(output_folder, "lkg1.pickle"),"wb")
pickle.dump(lkg, pickle_out)
pickle_out.close()
'''