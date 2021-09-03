import argparse
from os import listdir
from os.path import isfile, join, basename


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dir', type=str, default='', help='Folder that contain .h5 files.', required=True)
    parser.add_argument(
        '--ls', type=str, default='', help='DIRAC main list.', required=True)

    FLAGS, unparsed = parser.parse_known_args()

    dir = FLAGS.dir
    ls = FLAGS.ls

    print('Dir: ', dir, '\n')
    print('Main list: ', ls, '\n')

    files = [f for f in listdir(dir) if (isfile(join(dir, f)) and f.endswith('.h5'))]

    # print('Files: ', files)

    with open(ls) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    newls = []

    for i in content:
        bn = basename(i)
        if bn not in files:
            newls.append(i)

    print(newls)

    with open(ls[:-5] + '_new.list', 'w') as f:
        for item in newls:
            f.write("%s\n" % item)
