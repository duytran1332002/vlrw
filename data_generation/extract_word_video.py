import argparse

from label_processor import LabelProcessor
from utils import check_data_dir, check_dir


def load_args():
    parser = argparse.ArgumentParser(description='Extract word-level video')

    parser.add_argument('-d', '--data-dir',
                        default=None,
                        help='data directory')
    parser.add_argument('--video-dir',
                        default=None,
                        help='video directory')
    parser.add_argument('--srt-dir',
                        default=None,
                        help='srt directory')
    parser.add_argument('--word-video-dir',
                        default=None,
                        help='the directory of saving word-level video')
    parser.add_argument('-s', '--start-date',
                        default=None,
                        help='Which date to start')
    parser.add_argument('-e', '--end-date',
                        default=None,
                        help='Which date to stop')
    parser.add_argument('-m', '--mode',
                        default='override',
                        help='How to deal with existing files \
                                override, skip or append')
    parser.add_argument('--train-size',
                        type=float,
                        default=0.7,
                        help='The proportion of training set')
    parser.add_argument('--test-size',
                        type=float,
                        default=0.1,
                        help='The proportion of test set')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = load_args()

    data_dir = args.data_dir
    video_dir = args.video_dir
    srt_dir = args.srt_dir
    word_video_dir = args.word_video_dir
    start_date = args.start_date
    end_date = args.end_date
    mode = args.mode
    train_size = args.train_size
    test_size = args.test_size

    # for debugging
    # data_dir = r'D:\Coding\LipReadingProject\test_data'
    # start_date = '20220701'
    # end_date = '20220701'
    # mode = 'override'
    # train_size = 0.7
    # test_size = 0.2

    # check mode
    assert mode in ['override', 'skip'], 'Invalid mode'

    # if user provides a data directory
    if data_dir is not None:
        dir_names = ['videos', 'srt_transcripts',
                     'csv_transcripts', 'word_videos']
        video_dir, srt_dir, csv_dir, word_video_dir = check_data_dir(
            data_dir, dir_names)

    # check save_dir
    check_dir(word_video_dir)

    # extract video
    label_processor = LabelProcessor(data_dir=data_dir,
                                     video_dir=video_dir,
                                     srt_dir=srt_dir,
                                     word_video_dir=word_video_dir,
                                     start_date=start_date,
                                     end_date=end_date)
    label_processor.generate_video(mode=mode)
    # label_processor.split_train_val_test(train_size=train_size,
    #                                      test_size=test_size)
