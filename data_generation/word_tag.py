import argparse
import pandas as pd
import pysrt
import os
from tqdm import tqdm
from utils import *


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


# if user provides a data directory
if data_dir is not None:
    dir_names = ['srt_transcripts']
    srt_dir = check_data_dir(data_dir, dir_names)[0]
    freq_path = os.path.join(data_dir, 'word_freq.csv')


# get list of srt paths
srt_paths = []
if os.path.isdir(srt_dir):
    # get list of srt files from start to end
    srt_files = get_file_list(files=os.listdir(srt_dir),
                              start=start_date,
                              end=end_date)
    srt_paths = [os.path.join(srt_dir, srt_file)
                 for srt_file in srt_files]
elif os.path.isfile(srt_dir):
    srt_files = [os.path.basename(srt_dir)]
    srt_paths.append(srt_dir)
else:
    raise Exception('Invalid srt(s)')


# get dictionary of top n-class words and their number of sample
if os.path.isfile(freq_path):
    freq_list = convert_column_datatype(read_csv_to_list(freq_path),
                                        column=1,
                                        datatype=int)
    top_freq_list = sorted(freq_list, key=lambda x: x[1])[:n_class]
    top_freq_dict = dict(top_freq_list)
else:
    raise Exception('Invalid word frequency file')


n_tagged = 0
n_untagged = 0
# add tag to words that have the number of sample below threshold
for srt_path in tqdm(srt_paths,
                     total=len(srt_paths),
                     desc='srt files',
                     unit=' file',
                     dynamic_ncols=True):
    subs = pysrt.open(srt_path)
    for sub in tqdm(subs, leave=False):
        word = sub.text
        is_tagged = False

        if word[-2:] == tag:
            word = word[:-2]
            is_tagged = True

        if word in top_freq_dict.keys() and top_freq_dict[word] < threshold:
            if not is_tagged:
                sub.text = word + tag
            n_tagged += 1
        else:
            if is_tagged and mode == 'override':
                sub.text = word
                n_untagged += 1
    subs.save(srt_path)


print(f'\nNumber of tagged words: {n_tagged}')
print(f'\nNumber of untagged words: {n_untagged}')

# save list of top words
vocab_path = os.path.join(data_dir, f'{n_class}_vocabs_sorted_list.txt')
vocabs = list(top_freq_dict.keys())
old_vocabs = []
if os.path.isfile(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        old_vocabs = f.read().split()
vocabs = sorted(list(set(vocabs) | set(old_vocabs)))
with open(vocab_path, 'w', encoding='utf-8') as f:
    print(*vocabs, sep='\n', file=f)
print(
    f'\nNumber of new vocabs: {len(set(top_freq_dict.keys()) - set(vocabs))}')
print(f'Number of vocabs in the database: {len(vocabs)}')
print(f'List of them has been stored at: {vocab_path}.')
