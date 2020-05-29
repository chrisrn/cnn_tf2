import os
import numpy as np
import shutil
import argparse


def main(args):
    # # Creating Train / Val / Test folders (One time use)
    root_dir = args.data_dir
    classes = os.listdir(root_dir)

    val_ratio = args.val_ratio
    test_ratio = args.test_ratio

    all_files = 0
    train_files = 0
    val_files = 0
    test_files = 0
    for cls in classes:
        os.makedirs(root_dir +'/train/' + cls)
        os.makedirs(root_dir +'/val/' + cls)
        os.makedirs(root_dir +'/test/' + cls)

        # Creating partitions of the data after shuffling
        src = os.path.join(root_dir, cls) # Folder to copy images from

        all_FileNames = os.listdir(src)
        np.random.shuffle(all_FileNames)
        train_FileNames, val_FileNames, test_FileNames = np.split(np.array(all_FileNames),
                                                                  [int(len(all_FileNames)* (1 - val_ratio - test_ratio)),
                                                                   int(len(all_FileNames)* (1 - test_ratio))])

        train_FileNames = [src+'/' + name for name in train_FileNames.tolist()]
        val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
        test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]

        all_files += len(all_FileNames)
        train_files += len(train_FileNames)
        val_files += len(val_FileNames)
        test_files += len(test_FileNames)

        # Copy-pasting images
        for name in train_FileNames:
            shutil.move(name, root_dir +'/train/' + cls)

        for name in val_FileNames:
            shutil.move(name, root_dir +'/val/' + cls)

        for name in test_FileNames:
            shutil.move(name, root_dir +'/test/' + cls)
        shutil.rmtree(src)

    print('Total images: {}'.format(all_files))
    print('Training: {}'.format(train_files))
    print('Validation: {}'.format(val_files))
    print('Testing: {}'.format(test_files))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Directory of data files')
    parser.add_argument('--val_ratio', type=float, help='Validation ratio')
    parser.add_argument('--test_ratio', type=float, help='Test ratio')

    args = parser.parse_args()
    main(args)
