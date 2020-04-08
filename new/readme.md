# Training & test of CNNs for LST data

### Gear

- Tensorflow 1.12.0
- Keras 2.2.4 (install from repo and comment one line of code)
- Ctapipe 0.7.0

### Software installation
- Install Tensorflow via Conda
- Install Keras from the GitHub source:
    
First, clone Keras using `git`:

```sh
git clone https://github.com/keras-team/keras.git
```

Now comment the line 

```python
sys.stderr.write('Using TensorFlow backend.\n')
```

in ```keras/backend/load_backend.py```

 Then, `cd` to the Keras folder and run the install command:
```sh
cd keras
sudo python setup.py install
```

It is necessary to comment this line in order to avoid many prints of 'Using TensorFlow backend.' in the stdout, as the way Keras/Tensorflow was modified by the line:

```python
mp.set_start_method('spawn', force=True)
```

This is to avoid the deadlock problem when training using hdf5 files.

### 1 Data interpolation
First of all the data needs to be interpolated.
To interpolate the data it is sufficient to use the script ```lst_interpolate.py``` with:

```sh
python lst_interpolate.py --dirs path/to/folder1 path/to/folder2 ... --rem_org 0 --rem_corr 0 --rem_nsnerr 0 --parallel 1
```

If you encounter some problems, for example the memory consumption becomes too high and some files are not interpolated due to silent killing behaviour of the processes by the OS, then you can try to remove the flag parallel so that it performs interpolation using a single task and so less memory.

### 2 Classifier training

You can check the first lines of the main to see all the available options. A typical training can be

```sh
python classifier_training.py --dirs ~/simulations/gamma_diffuse/train/ ~/simulations/proton/train/ --val_dirs ~/simulations/gamma/val/ ~/simulations/proton/val/ --model BaseLine --time 1 --epochs 50 --batch_size 64 --opt adam --lrop 1 --workers 4 
```

### 3 Regressor training

```sh
python regressor_training.py  
```

### 4 Test chain on test data

```sh
python cnn-lst-chain.py --dirs ~/simulations/gamma_pointlike/test/ ~/simulations/proton/test/ --time 1 --model_sep /path/to/classifier.h5 --model_energy /path/to/nrg_regressor.h5 --model_azalt /path/to/dir_regressor.h5
```

reintroduce ADABOUND???