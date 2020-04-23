import warnings
warnings.simplefilter('ignore')

import argparse
import h5py
import keras
from keras.models import Model
# from keras import Sequential
# from keras_contrib.applications.resnet import ResNet
# from classifier_selector import select_classifier
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten, Input
from keras.callbacks import TensorBoard
import keras.backend as K
import numpy as np
import datetime
from tqdm import tqdm
import random
import tensorflow as tf
from os.path import join

from ctapipe.instrument import CameraGeometry
import matplotlib.pyplot as plt


def save_plots(charges, label, start=0):
    cam = CameraGeometry.from_name('LSTCam')
    x = cam.pix_x
    y = cam.pix_y
    points = np.array([np.array(x),
                       np.array(y)]).T
    path_fig = '/home/pgrespan/fig/'
    for i, interp_crg in enumerate(charges):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set(title=label + str(i), xlabel="x [m]", ylabel="y [m]")
        plt.imshow(interp_crg.T, extent=(points.min(), points.max(), points.min(), points.max()), origin='lower')
        plt.savefig(fname=path_fig + label + str(i + start))
        plt.clf()
        plt.close(fig)

'''
def acquire_data(file_list, label, intens):
    train_data = []
    for f in file_list:
        h5f = h5py.File(f, 'r')
        i = h5f['LST/intensities'][:]
        l = h5f['LST/intensities_width_2'][:]
        for j, x in enumerate(tqdm(h5f['LST/LST_image_charge_interp'])):
            if (i[j] >= intens) & (l[j] <= 0.2):
                train_data.append([x, int(label)])
        h5f.close()
    return train_data
'''
parser = argparse.ArgumentParser()

parser.add_argument(
    '-m', '--model', type=str, default='VGG19', help='Model type between VGG (default) and ResNet.', required=False)
parser.add_argument(
    '-e', '--epochs', type=int, default=10, help='Number of epochs.', required=True)
parser.add_argument(
    '-bs', '--batch_size', type=int, default=64, help='Batch size.', required=True)
parser.add_argument(
    '-lr', '--learning_rate', type=str, default='adam', help='Specify the learning rate.', required=False)
parser.add_argument(
    '-i', '--intensity_cut', type=float, default='50', help='Specify the lowest intensity accepted for an event.', required=False)
parser.add_argument(
    '--gpu_fraction', type=float, default=0, help='Set limit to fraction of GPU memory usage. IMPORTANT: between 0 and 1.', required=False)

FLAGS, unparsed = parser.parse_known_args()

# cmd line parameters
mod = FLAGS.model
epochs = FLAGS.epochs
batchsize = FLAGS.batch_size
lr = FLAGS.learning_rate
intens_cut = FLAGS.intensity_cut
gpu_fraction = FLAGS.gpu_fraction


##################################
# TensorFlow wizardry for GPU dynamic memory allocation
gpu_fraction = float(gpu_fraction)
if gpu_fraction != 0 and gpu_fraction <= 1:
    config = tf.ConfigProto()
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True
    # Only allow a fraction of the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
    # Create a session with the above options specified.
    K.tensorflow_backend.set_session(tf.Session(config=config))
###################################

model_name = mod + '-lr' + str(lr) + '-bs' + str(batchsize) + '-'
now = datetime.datetime.now()
NAME = now.strftime(model_name + '-' + '%Y-%m-%d_%H-%M')
tensorboard = TensorBoard(log_dir='/home/pgrespan/tb/{}'.format(NAME))

path = '/home/pgrespan/simulations/small_test'

print("Loading first data tranche...")
i0 = np.load(join(path, 'data0', 'intens0.npy'))
l0 = np.load(join(path, 'data0', 'leakg20.npy'))
X0 = np.load(join(path, 'data0', 'images0.npy'))[ (i0 >= intens_cut) & (l0<=0.2) ]
y0 = np.load(join(path, 'data0', 'labels0.npy'))[ (i0 >= intens_cut) & (l0<=0.2) ]

print("Loading second data tranche...")
i1 = np.load(join(path, 'data1', 'intens1.npy'))
l1 = np.load(join(path, 'data1', 'leakg21.npy'))
X1 = np.load(join(path, 'data1', 'images1.npy'))[ (i1 >= intens_cut) & (l1<=0.2) ]
y1 = np.load(join(path, 'data1', 'labels1.npy'))[ (i1 >= intens_cut) & (l1<=0.2) ]

print("Appending...")
X = np.append(X0,X1, axis=0)
y = np.append(y0,y1)

print("Done!\nImages shape (total): {}".format(X.shape))
print("Labels shape (total): {}".format(y.shape))

IMG_SIZE = 100
input_shape = (IMG_SIZE, IMG_SIZE, 1)
input_img = Input(input_shape, name='input_img')

if mod == "ResNet":
    base = keras.applications.resnet.ResNet50(include_top=False, weights=None, input_tensor=input_img, pooling='max')
else:
    base = keras.applications.vgg19.VGG19(include_top=False, weights=None, input_tensor=input_img, pooling='max')

x = base.layers[-1].output
x = Dense(1, name='gammaness', activation='sigmoid')(x)
model = Model(inputs=input_img, output=x)

# adam
a_lr = float(lr)
a_beta_1 = 0.9
a_beta_2 = 0.999
a_epsilon = None
a_decay = 0
amsgrad = True

adam = keras.optimizers.Adam(lr=a_lr, beta_1=a_beta_1, beta_2=a_beta_2, epsilon=a_epsilon, decay=a_decay,
                             amsgrad=amsgrad)

model.compile(loss="binary_crossentropy",
              optimizer=adam,
              metrics=['accuracy'])

model.fit(X, y, batch_size=batchsize, epochs=int(epochs), verbose=1, validation_split=0.125, shuffle=True,
          callbacks=[tensorboard])
#model.fit(X, y, batch_size=batchsize, epochs=int(epochs), verbose=1, validation_data=(X,y), shuffle=True, callbacks=[tensorboard])

######################################################################################################
##############################         CADAVERI         ##############################################
######################################################################################################

# model, _ = select_classifier(model_name=model_name, hype_print=_, channels=1, img_cols=100, img_rows=100)
'''
model = ka.vgg19.VGG19(
    include_top=False,
    weights=None,
    input_tensor= input_img,
    pooling='max')
'''
'''
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
'''

'''
model = ResNet(input_shape=input_shape,
               block='basic',
               dropout=0,
               repetitions=[2, 2, 2, 2],
               residual_unit='v1',
               classes=1,
               activation='sigmoid')
'''

'''
h5f = h5py.File(gfiles[0], 'r')
intensity = h5f['LST/intensities'][:]
leakage = h5f['LST/intensities_width_2'][:]
images = h5f['LST/LST_image_charge_interp'][:][(intensity>=50) & (leakage<=0.2)]
labels = [1] * len(images)
#save_plots(images, "Gamma ")
h5f.close()

labels = []
images = np.array([], dtype=np.float64).reshape(-1,100,100)

for f in tqdm(gfiles):
    h5f = h5py.File(f, 'r')
    intensity = h5f['LST/intensities'][:]
    leakage = h5f['LST/intensities_width_2'][:]
    crgs = h5f['LST/LST_image_charge_interp'][:][(intensity>=50) & (leakage<=0.2)]
    lbl = [1] * len(crgs)
    images = np.vstack((images,crgs))
    #save_plots(crgs, "Proton ")
    labels+=lbl
    h5f.close()

print("Images shape (only gammas): {}".format(images.shape))
print("Labels shape (only gammas): {}".format(len(labels)))

h5f = h5py.File(pfiles[0], 'r')
#intensity = h5f['LST/intensities'][:]
#leakage = h5f['LST/intensities_width_2'][:]
crgs = h5f['LST/LST_image_charge_interp'][:1500]#[(intensity>=50) & (leakage<=0.2)]
lbl = [0] * len(crgs)
images = np.vstack((images,crgs))
labels+=lbl
h5f.close()

#proton_start = 0
for i, f in enumerate(tqdm(pfiles)):
    h5f = h5py.File(f, 'r')
    intensity = h5f['LST/intensities'][:]
    leakage = h5f['LST/intensities_width_2'][:]
    crgs = h5f['LST/LST_image_charge_interp'][:][(intensity>=50) & (leakage<=0.2)]
    lbl = [0] * len(crgs)
    #if i < 9:
    #    save_plots(crgs, "Proton ", proton_start)
    images = np.vstack((images,crgs))
    labels+=lbl
    h5f.close()
    if i > 8:
        break
    #proton_start+=len(crgs)

X = images.reshape(-1,100,100,1)
y = np.array(labels)
'''
