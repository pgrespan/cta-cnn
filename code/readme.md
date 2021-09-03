# Training & test of CNNs for LST data

### Gear

- Tensorflow 1.12.0
- Keras 2.2.5
- Ctapipe 0.8.0

### 1 Data interpolation
First of all the data needs to be interpolated.
To interpolate the data it is sufficient to use the script ```lst_interpolate.py``` with:

```sh
python lst_interpolate.py --dirs INPUT_IMAGES_FOLDERS --out OUTPUT_DIRECTORY --parallel
```

If you encounter some problems, for example the memory consumption becomes too high and some files are not interpolated due to silent killing behaviour of the processes by the OS, then you can try to remove the flag parallel so that it performs interpolation using a single task and so less memory.

### 2 Classifier training

Run with the --help option to see the availavle options.

```sh
python classifier_training.py --dirs PATHS_TO_TRAINING_FOLDERS --val_dirs PATHS_TO_VALIDATION_FOLDERS --model MODEL_NAME --time --epochs NUMBER_OF_EPOCHS --batch_size BATCH_SIZE --clr --workers NUM_OF_WORKERS 
```

### 3 Regressor training
Similar to step n. 2

```sh
python regressor_training.py ...
```

### 4 Test on test data

```sh
python fullEventReconstructor.py --dirs PATH_TO_TEST_DATA_FOLDER -s GH_SEPARATORS_FOLDER -e ENERGY_REGRESSORS_FOLDER -d DIRECTION_REGRESSORS_PATH --output OUTPUT_FILE_PATH --time -i INTENSITY_CUT -lkg LEAKAGE_CUT
```

