from lst_generator import LSTGenerator
import utils, argparse

def bbai(dirs, outfile='./data_info', model=''):

    files = utils.get_all_files(dirs)
    if model == 'fametrecafe':
        model = '/home/pgrespan/trained/VGG16_2020-04-21_11-01/VGG16_13_0.88753_0.86502.h5'
    gen = LSTGenerator(
        h5files=files,
        feature='gammaness',
        shuffle=False,
        intensity=0.0,
        leakage2_intensity=12,
        arrival_time=True,
        gammaness=0.0,
        class_model=model
    )
    try:
        gen.indexes.to_pickle(outfile + '.pkl')
    except:
        print('No riesso catare a cartea, probabilmente no esiste, molton!')

    return gen.indexes


if __name__ == 'main':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dirs', type=str, default='', nargs='+', help='Folders that contains data.', required=True)
    parser.add_argument(
        '--outfile', type=str, help='Full path of output file.', required=False)
    parser.add_argument(
        '--model', type=str, help='Path of the model to evaluate gammaness (optional).', required=False)

    FLAGS, unparsed = parser.parse_known_args()

    dirs = FLAGS.dirs
    outfile = FLAGS.outfile
    model = FLAGS.model

    bbai(dirs,outfile,model)