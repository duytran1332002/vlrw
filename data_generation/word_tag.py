import argparse
from utils import check_data_dir
from label_processor import LabelProcessor


def load_args():
    parser = argparse.ArgumentParser(description='Extract word-level video')

    parser.add_argument('-d', '--data-dir',
                        default=None,
                        help='data directory')
    parser.add_argument('--srt-dir',
                        default=None,
                        help='srt directory')
    parser.add_argument('--word-freq-path',
                        default=None,
                        help='File that contains words and their number of \
                              sample')
    parser.add_argument('-s', '--start-date',
                        default=None,
                        help='Which date to start')
    parser.add_argument('-e', '--end-date',
                        default=None,
                        help='Which date to stop')
    parser.add_argument('--n-class',
                        type=int,
                        default=400,
                        help='Get top n-class')
    parser.add_argument('-t', '--threshold',
                        type=int,
                        default=15,
                        help='Minimum number of sample per class')
    parser.add_argument('-m', '--mode',
                        default='override',
                        help='How to deal with existing tags that is not \
                              needed: override, skip')
    parser.add_argument('--tag',
                        default='0',
                        help='Define tag')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = load_args()

    data_dir = args.data_dir
    srt_dir = args.srt_dir
    word_freq_path = args.word_freq_path
    start_date = args.start_date
    end_date = args.end_date
    n_class = args.n_class
    threshold = args.threshold
    mode = args.mode
    tag = args.tag

    # for debugging
    # data_dir = r'D:\Coding\LipReadingProject\test_data'
    # srt_dir = None
    # word_freq_path = None
    # start_date = '20220810'
    # end_date = '20221227'
    # n_class = 400
    # threshold = 20
    # mode = 'override'
    # tag = '_0'

    # check mode
    assert mode in ['override', 'skip'], 'Invalid mode'

    # check tag
    assert len(tag) == 1, 'Invalid tag'

    # if user provides a data directory
    if data_dir is not None:
        dir_names = ['srt_transcripts']
        srt_dir = check_data_dir(data_dir, dir_names)[0]

    # start tagging
    label_processor = LabelProcessor(data_dir=data_dir,
                                     srt_dir=srt_dir,
                                     start_date=start_date,
                                     end_date=end_date,
                                     n_class=400)
    label_processor.tag(threshold, mode, tag)
