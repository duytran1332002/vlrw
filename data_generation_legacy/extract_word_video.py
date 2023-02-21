import argparse
import datetime
import glob
import os
import pandas as pd
import pysrt
from tqdm import tqdm
import io
import contextlib
import csv
from utils import *
from moviepy.video.io.VideoFileClip import VideoFileClip
from vPhon import convert_grapheme_to_phoneme


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
    parser.add_argument('--save-dir',
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
    parser.add_argument('--save-csv',
                        default=False,
                        help='save srt to csv')
    parser.add_argument('--csv-dir',
                        default=None,
                        help='csv directory')
    parser.add_argument('--fix-error',
                        default=False,
                        help='Fix error word')

    args = parser.parse_args()
    return args


def change_boundary(time_boundary: datetime.time,
                    time_change: float) -> datetime.time:
    '''
    Change the boundary of a time object

    Parameters:
        time_boundary: datetime.time
        time_change: float
            the change of time boundary in seconds

    returns:
        datetime.time
    '''
    dt = datetime.datetime.combine(datetime.date.today(), time_boundary)
    if time_change > 0 or time_change < dt.timestamp():
        dt += datetime.timedelta(seconds=time_change)
    else:
        dt -= datetime.timedelta(seconds=-time_change)

    return dt.time()


def read_srt_to_df(srt_path):
    '''
    Read srt file to pandas.dataframe

    Parameters:
        srt_path: str
            the path of srt file
        csv_path: str
            the path of csv file

    Returns:
        pd.DataFrame
            the dataframe of words
    '''
    subs = pysrt.open(srt_path, encoding='utf-8')
    words = pd.DataFrame(columns=['start', 'end', 'word'])

    # TODO: convert small second to mircosecond
    for sub in subs:
        # add padding to start time
        start = change_boundary(time_boundary=sub.start.to_time(),
                                time_change=-0.01)

        # add padding to end time
        end = change_boundary(time_boundary=sub.end.to_time(),
                              time_change=0.01)

        word = pd.DataFrame({'start': [start.strftime('%H:%M:%S.%f')],
                             'end': [end.strftime('%H:%M:%S.%f')],
                             'word': [sub.text]})
        words = pd.concat([words, word], ignore_index=True)

    return words


def cut_video(video, start, end, file_path):
    '''
    Cut vide from start to end and save the cut

    Parameters:
        video: moviepy.video.io.VideoFileClip.VideoFileClip
            the video to be cut
        start: datetime.time
            the start time of the cut
        end: datetime.time
            the end time of the cut
        file_path: str
            the path of the cut video
    '''
    with contextlib.redirect_stdout(io.StringIO()):
        piece = video.subclip(start, end)
        piece.write_videofile(file_path, fps=video.fps)


args = load_args()

data_dir = args.data_dir
video_dir = args.video_dir
srt_dir = args.srt_dir
save_dir = args.save_dir
start_date = args.start_date
end_date = args.end_date
mode = args.mode
save_csv = args.save_csv
csv_dir = args.csv_dir


# for debugging
# data_dir = r'D:\Coding\LipReadingProject\test_data'
# video_dir = None
# srt_dir = None
# save_dir = None
# start_date = '20220810'
# end_date = '20221227'
# mode = 'override'
# save_csv = True
# csv_dir = None


# check mode
assert mode in ['override', 'skip', 'append'], 'Invalid mode'


# if user provides a data directory
if data_dir is not None:
    dir_names = ['videos', 'srt_transcripts', 'csv_transcripts', 'word_videos']
    video_dir, srt_dir, csv_dir, save_dir = check_data_dir(data_dir,
                                                           dir_names)


# get list of video paths
video_paths = []
if os.path.isdir(video_dir):
    # get list of videos from start to end
    videos = get_file_list(files=os.listdir(video_dir),
                           start=start_date,
                           end=end_date)
    video_paths = [os.path.join(video_dir, video)
                   for video in videos]
elif os.path.isfile(video_dir):
    videos = [os.path.basename(video_dir)]
    video_paths.append(video_dir)
else:
    raise Exception('Invalid video(s)')


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


# check missing data
video_dates = [video[:8] for video in videos]
srt_dates = [srt_file[:8] for srt_file in srt_files]
missing_dates = sorted(find_complement(srt_dates, video_dates))
missing_files = []
if len(missing_dates) > 0:
    for date in missing_dates:
        if binary_seach(os.listdir(srt_dir), f'{date}.srt') == -1:
            missing_files.append(f'{date}.srt')
            video_paths.remove(os.path.join(video_dir, f'{date}.mp4'))
        else:
            missing_files.append(f'{date}.mp4')
            srt_paths.remove(os.path.join(srt_dir, f'{date}.srt'))
    # save missing files
    with open(os.path.join(data_dir, 'missing_files.txt'), 'a') as f:
        print(*missing_files, sep='\n', file=f)


# get list of csv paths
if save_csv:
    check_dir(csv_dir)


# check save_dir
check_dir(save_dir)


# generate grapheme dictionary from old one
grapheme_dict = dict()
vocab_path = os.path.join(data_dir, 'vocabs_sorted_list.txt')
if os.path.isfile(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        old_vocabs = f.read().split()
        for vocab in old_vocabs:
            phon = convert_grapheme_to_phoneme(vocab)
            if grapheme_dict.get(phon, None) is None:
                grapheme_dict[phon] = vocab


freq_dict = dict()
new_freq_dict = dict()
errors = pd.DataFrame(columns=['date', 'start', 'end', 'word', 'error'])
# extract word-level video
for video_path, srt_path in tqdm(zip(video_paths, srt_paths),
                                 total=len(video_paths),
                                 desc='Videos',
                                 unit=' video',
                                 dynamic_ncols=True):
    # Load the video file
    video = VideoFileClip(video_path)

    # read srt to dataframe
    df = read_srt_to_df(srt_path)
    if save_csv:
        srt_name = os.path.basename(srt_path)
        csv_path = os.path.join(csv_dir, srt_name.replace('srt', 'csv'))
        df.to_csv(csv_path, index=False)

    for _, row in tqdm(df.iterrows(),
                       total=len(df),
                       desc='Words',
                       unit=' word',
                       leave=False,
                       dynamic_ncols=True):
        # cut the video into smaller pieces
        start = row.start
        end = row.end
        word = row.word

        # merge words that have similar sound
        phon = convert_grapheme_to_phoneme(word)
        if grapheme_dict.get(phon, None) is None:
            grapheme_dict[phon] = word
        else:
            word = grapheme_dict[phon]

        # check word folder
        label_dir = os.path.join(save_dir, word)
        check_dir(label_dir)

        # name the video
        date = os.path.basename(video_path)[:8]
        freq_dict[word] = freq_dict.get(word, 0) + 1
        file_name = f'{date}{str(freq_dict[word]).zfill(5)}.mp4'
        # if the name exists
        if file_name in os.listdir(label_dir):
            if mode == 'append':
                while file_name in os.listdir(label_dir):
                    freq_dict[word] = freq_dict.get(word, 0) + 1
                    file_name = f'{date}{str(freq_dict[word]).zfill(5)}.mp4'
            elif mode == 'skip':
                continue
        file_path = os.path.join(label_dir, file_name)

        # cut video
        try:
            cut_video(video, start, end, file_path)
            new_freq_dict[word] = new_freq_dict.get(word, 0) + 1
        except KeyboardInterrupt:
            print('\n')
            os._exit(0)
        except Exception as e:
            # print(f'Error when process {file_name}')
            # print(e)
            error_df = pd.DataFrame({'date': [file_name[:8]],
                                     'start': [row.start],
                                     'end': [row.end],
                                     'word': [row.word],
                                     'error': [e]})
            errors = pd.concat([errors, error_df], ignore_index=True)


# delete leftover files
leftover_files = glob.glob('*.mp3')
for file in tqdm(leftover_files, desc='Cleaning', unit='iter'):
    os.remove(file)


# save word frequency into csv
freq_path = os.path.join(data_dir, 'word_freq.csv')
print(f'\nTotal sample in this run: {sum(list(freq_dict.values()))}')
print(
    f'Number of new samples from this run: {sum(list(new_freq_dict.values()))}')
if os.path.isfile(freq_path):
    old_freq_list = read_csv_to_list(freq_path)
    print(old_freq_list)
    for word, freq in old_freq_list:
        new_freq_dict[word] = new_freq_dict.get(word, 0) + int(freq)
save_list_to_csv(list(new_freq_dict.items()), freq_path)
print(
    f'Number of samples in the database: {sum(list(new_freq_dict.values()))}')
print(f'The vocabularies\' frequency has been stored at: {freq_path}.')


# save list of vocabularies
vocab_path = os.path.join(data_dir, 'vocabs_sorted_list.txt')
vocabs = list(freq_dict.keys())
old_vocabs = []
if os.path.isfile(vocab_path):
    old_vocabs = read_txt_to_list(vocab_path)
vocabs = sorted(list(set(vocabs) | set(old_vocabs)))
save_list_to_txt(vocabs, vocab_path)
print(f'\nNumber of new vocabs: {len(set(freq_dict.keys()) - set(vocabs))}')
print(f'Number of vocabs in the database: {len(vocabs)}')
print(f'List of them has been stored at: {vocab_path}.')


# save error words
print(f'\nNumber of words that cause error: {len(errors)}')
error_path = os.path.join(data_dir, 'errors.csv')
if os.path.isfile(error_path):
    old_errors = pd.read_csv(error_path)
    errors = pd.concat([old_errors, errors], ignore_index=True)
    errors = errors.drop_duplicates()
errors.to_csv(error_path, encoding='utf-8', index=False)
print(f'List of these error words has been stored at: {error_path}.')


# dealing with error words
# TODO: complete this code
# if args.fix_error:
#     print('\n Try expand word boundary and re-cut')
#     fixed = pd.DataFrame(columns=['date', 'start', 'end', 'word'])
#     for _, row in errors.iterrows():
#         start = change_boundary(row.start, -0.01)
#         end = change_boundary(row.end, 0.01)
#         is_error = True
#         while is_error:
#             try:
#                 cut_video(video, start, end, file_path)
#                 temp = pd.DataFrame({'date': [row.date],
#                                      'start': [start],
#                                      'end': [end],
#                                      'word': [row.word]})
#                 fixed = pd.concat([fixed, temp], ignore_index=True)
#                 is_error = False
#             except Exception as e:
#                 print(f'Error when process {file_name}')
#                 print(e)
#                 start = change_boundary(row.start, -0.01)
#                 end = change_boundary(row.end, 0.01)
#     fixed.to_csv(error_path, encoding='utf-8', index=False)
