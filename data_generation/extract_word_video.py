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
    parser.add_argument('--fix-error',
                        default=False,
                        help='Fix error word')

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

    # for debugging
    # data_dir = r'D:\Coding\LipReadingProject\test_data'
    # video_dir = None
    # srt_dir = None
    # save_dir = None
    # start_date = '20220701'
    # end_date = '20220809'
    # mode = 'override'

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
    label_processor = LabelProcessor(data_dir, video_dir, srt_dir, word_video_dir,
                                     start_date, end_date)
    label_processor.process(mode)
