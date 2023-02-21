import argparse
from utils import check_data_dir
from label_processor import LabelProcessor


def load_args():
    parser = argparse.ArgumentParser(description='Extract word-level video')

    parser.add_argument('-d', '--data-dir',
                        default=None,
                        help='data directory')
    parser.add_argument('--word-video-dir',
                        default=None,
                        help='video directory')
    parser.add_argument('--train-size',
                        type=float,
                        default=None,
                        help='The proportion of training set')
    parser.add_argument('--test-size',
                        type=float,
                        default=None,
                        help='The proportion of test set')
    parser.add_argument('--use-val',
                        default=True,
                        help='Use validation set')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = load_args()

    data_dir = args.data_dir
    word_video_dir = args.word_video_dir
    train_size = args.train_size
    test_size = args.test_size

    # for debugging
    # data_dir = r'D:\Coding\LipReadingProject\data'
    # word_video_dir = None
    # train_size = 0.7
    # test_size = 0.2

    # if user provide a data directory
    if data_dir is not None:
        dir_names = ['word_videos']
        word_video_dir = check_data_dir(data_dir, dir_names)[0]

    # start splitting
    label_processor = LabelProcessor(data_dir=data_dir,
                                     word_video_dir=word_video_dir)
    label_processor.split_train_val_test(train_size=train_size,
                                         test_size=test_size)
