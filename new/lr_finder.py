#from classifier_training import classifier_training_main
from regressor_training import regressor_training_main
from keras import backend as K
import argparse

def lr_finder(ftr):

    cnns = [
            'VGG16',
            'ResNet50V2',
            'InceptionResNetV2',
            'DenseNet121',
            'ResNet50',
            'InceptionV3',
            'Xception',
            ]

    dirs = [
        "/home/pgrespan/simulations/gamma_diff/train",
        "/home/pgrespan/simulations/gamma_diff/test"
    ]

    val_dirs = [
        "/home/pgrespan/simulations/gamma_diff/val",
        "/home/pgrespan/simulations/gamma_diff/test_small"
    ]

    test_dirs = [
        "/home/pgrespan/simulations/gamma/class_test"
    ]

    hp = {
        "time":True,
        "epochs":5,
        "batch_size":128,
        "opt":"adam",
        "lr":[1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1.],
        "lropf":False,
        "sd":False,
        "es":True,
        "feature":ftr,
        "workers":4,
        "intensity":50,
        "tb":True,
    }

    for model in cnns:
        for lr in hp['lr']:
            regressor_training_main(
                folders=dirs,
                val_folders=val_dirs,
                test_dirs='',
                model_name=model,
                time=hp['time'],
                epochs=hp['epochs'],
                batch_size=hp['batch_size'],
                opt=hp["opt"],
                learning_rate=lr,
                lropf=hp['lropf'],
                sd= hp['sd'],
                es=hp['es'],
                feature=hp['feature'],
                workers=hp['workers'],
                intensity_cut=hp['intensity'],
                tb=hp['tb'],
                gpu_fraction=0.5,
                emin=-100,
                emax=100
            )
            K.clear_session()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--feature', type=str, default='', help='Specify feature ("energy" or "xy").', required=True)

    FLAGS, unparsed = parser.parse_known_args()
    f = FLAGS.feature
    lr_finder(f)