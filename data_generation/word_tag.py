import argparse
import pandas as pd
import pysrt
import os
from tqdm import tqdm
from utils import check_data_dir, get_file_list


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
                        default=400,
                        help='Get top n-class')
    parser.add_argument('-t', '--threshold',
                        default=20,
                        help='Minimum number of sample per class')
    parser.add_argument('-m', '--mode',
                        default='override',
                        help='How to deal with existing tags that is not \
                              needed: override, skip')
    parser.add_argument('--tag',
                        default='_0',
                        help='Define tag')

    args = parser.parse_args()
    return args


def is_tagged(word):
    return word[-2:] == args.tag


args = load_args()


# check mode
assert args.mode in ['override', 'skip'], 'Invalid mode'


# if user provides a data directory
if args.data_dir is not None:
    dir_names = ['srt_transcripts']
    srt_dir = check_data_dir(args.data_dir, dir_names)[0]
    freq_path = os.path.join(args.data_dir, 'word_freq.csv')
else:
    srt_dir = args.srt_dir
    freq_path = args.word_freq_path


# get list of srt paths
srt_paths = []
if os.path.isdir(srt_dir):
    # get list of srt files from start to end
    srt_files = get_file_list(files=os.listdir(srt_dir),
                              start=args.start_date,
                              end=args.end_date)
    srt_paths = [os.path.join(srt_dir, srt_file)
                 for srt_file in srt_files]
elif os.path.isfile(srt_dir):
    srt_files = [os.path.basename(srt_dir)]
    srt_paths.append(srt_dir)
else:
    raise Exception('Invalid srt(s)')


# get dictionary of top n-class words and their number of sample
if os.path.isfile(freq_path):
    freq_df = pd.read_csv(freq_path)
    top_freq_list = sorted(list(zip(freq_df['word'], freq_df['frequency'])),
                           key=lambda x: x[1])[:args.n_class]
    top_freq_dict = dict(top_freq_list)
else:
    raise Exception('Invalid word frequency file')


n_tagged = 0
# add tag to words that have the number of sample below threshold
for srt_path in tqdm(srt_paths,
                     total=len(srt_paths),
                     desc='srt files',
                     unit=' file',
                     dynamic_ncols=True,
                     postfix=f'Number of tag: {n_tagged}'):
    subs = pysrt.open(srt_path)
    for sub in subs:
        word = sub.text
        if word in top_freq_dict:
            if top_freq_dict[word] < args.threshold:
                if not is_tagged(word):
                    sub.text = word + args.tag
                n_tagged += 1
            else:
                if is_tagged(word) and args.mode == 'override':
                    sub.text = word[:-2]
    subs.save(srt_path)


print(f'Number of tagged words: {n_tagged}')
