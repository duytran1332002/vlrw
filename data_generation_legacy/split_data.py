# TODO: every label folder contains test, train, val folders with ratio
import argparse
import os
import shutil
from math import floor
from random import shuffle

from tqdm import tqdm
from utils import check_data_dir, check_dir


def load_args():
    parser = argparse.ArgumentParser(description='Extract word-level video')

    parser.add_argument('-d', '--data-dir',
                        default=None,
                        help='data directory')
    parser.add_argument('--word-video-dir',
                        default=None,
                        help='video directory')
    parser.add_argument('--train-size',
                        type=int,
                        default=None,
                        help='The proportion of training set')
    parser.add_argument('--test-size',
                        type=int,
                        default=None,
                        help='The proportion of test set')
    parser.add_argument('--use-val',
                        default=True,
                        help='Use validation set')

    args = parser.parse_args()
    return args


# args = load_args()


# data_dir = args.data_dir
# word_video_dir = args.word_video_dir
# train_size = args.train_size if args.train_size < 1 else args.train_size / 100
# test_size = args.test_size if args.test_size < 1 else args.test_size / 100
# val_size = 1 - (train_size + test_size)


# for debugging
data_dir = r'D:\Coding\LipReadingProject\data'
word_video_dir = None
train_size = 0.7
test_size = 0.2
val_size = 1 - (train_size + test_size)


# if user provide a data directory
if data_dir is not None:
    dir_names = ['word_videos']
    word_video_dir = check_data_dir(data_dir, dir_names)[0]


# get list of label paths
label_dirs = []
if os.path.isdir(word_video_dir):
    labels = os.listdir(word_video_dir)
    labels_dirs = [os.path.join(word_video_dir, label)
                   for label in labels]
else:
    raise Exception('Invalid word video directory')


for label_dir in tqdm(label_dirs,
                      desc='Labels',
                      total=len(label_dirs),
                      leave=True,
                      unit=' label',
                      dynamic_ncols=True):
    samples = os.listdir(label_dir)
    n_sample = len(samples)
    train_idx = floor(train_size * n_sample)
    val_idx = train_idx + floor(val_size * n_sample)
    # test_idx = val_idx + floor(test_size * n_sample)

    # shuffle data
    shuffle(samples)

    train_set = samples[:train_idx+1]
    val_set = samples[train_idx+1:val_idx+1]
    test_set = samples[val_idx+1:]

    # create train, val and test folders
    train_dir = os.path.join(label_dir, 'train')
    check_dir(train_dir)
    val_dir = os.path.join(label_dir, 'val')
    check_dir(val_dir)
    test_dir = os.path.join(label_dir, 'test')
    check_dir(test_dir)

    # move samples
    for sample in train_set:
        sample_path = os.path.join(label_dir, sample)
        shutil.move(sample_path, train_dir)
    for sample in val_set:
        sample_path = os.path.join(label_dir, sample)
        shutil.move(sample_path, val_dir)
    for sample in test_set:
        sample_path = os.path.join(label_dir, sample)
        shutil.move(sample_path, test_dir)

    print(f'Label: {os.path.basename(label_dir)} -', end=' ')
    print(f'train: {len(train_set)} -', end=' ')
    print(f'val: {len(val_set)} -', end=' ')
    print(f'test: {len(test_set)}')
