import h5py
import keras
from keras.models import Model
#from keras import Sequential
#from keras_contrib.applications.resnet import ResNet
#from classifier_selector import select_classifier
from keras.layers import Dense, Input
from keras.callbacks import TensorBoard
import keras.backend as K
import numpy as np
import datetime
import pickle
###################################
import tensorflow as tf
# TensorFlow wizardry for GPU dynamic memory allocation
config = tf.ConfigProto()
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
# Only allow a fraction of the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.50
# Create a session with the above options specified.
K.tensorflow_backend.set_session(tf.Session(config=config))
###################################

epochs = input("How many epochs? ")
model_name = input("Insert model name: ")
now = datetime.datetime.now()
NAME = now.strftime(model_name + '-' + '%Y-%m-%d_%H-%M')
tensorboard = TensorBoard(log_dir='/home/pgrespan/tb/{}'.format(NAME))

pickle_in = open("/home/pgrespan/catndogs/X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("/home/pgrespan/catndogs/y.pickle","rb")
y = pickle.load(pickle_in)
y = np.array(y)
print("Images shape: {}".format(X.shape))
print("Labels shape: {}".format(y.shape))

input_shape = (100,100,1)

'''
model = ResNet(input_shape=input_shape,
               block='basic',
               dropout=0,
               repetitions=[2, 2, 2, 2],
               residual_unit='v1',
               classes=1,
               activation='sigmoid')
'''

input_img = Input(input_shape, name='input_img')
#model, _ = select_classifier(model_name=model_name, hype_print=_, channels=1, img_cols=100, img_rows=100)


#base = keras.applications.resnet.ResNet50(include_top=False, weights=None, input_tensor=input_img, pooling='max')
base = keras.applications.vgg19.VGG19(include_top=False, weights=None, input_tensor=input_img, pooling='max')
x = base.layers[-1].output
x = Dense(1, name='gammaness', activation='sigmoid')(x)
model = Model(inputs=input_img, output=x)

# adam
a_lr = 1e-05
a_beta_1 = 0.9
a_beta_2 = 0.999
a_epsilon = None
a_decay = 0
amsgrad = True

adam = keras.optimizers.Adam(lr=a_lr, beta_1=a_beta_1, beta_2=a_beta_2, epsilon=a_epsilon, decay=a_decay, amsgrad=amsgrad)

model.compile(loss="binary_crossentropy",
             optimizer=adam,
             metrics=['accuracy'])

model.fit(X, y, batch_size=128, epochs=int(epochs), verbose=1, validation_split=0.125, shuffle=True, callbacks=[tensorboard])