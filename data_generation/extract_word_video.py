import argparse
import datetime
import glob
import os
import pandas as pd
import pysrt
from tqdm import tqdm
import io
import contextlib
from utils import binary_seach, check_dir, find_complement, get_file_list
from moviepy.video.io.VideoFileClip import VideoFileClip


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
                        default=True,
                        help='save srt to csv')
    parser.add_argument('--csv-dir',
                        default=None,
                        help='csv directory')
    parser.add_argument('--fix-error',
                        default=False,
                        help='Fix error word')
    # TODO: Add sample threshold

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

# check mode
assert args.mode in ['override', 'skip', 'append'], 'Invalid mode'

# if user provide a data directory
if args.data_dir is not None:
    # get video directory
    video_dir = os.path.join(args.data_dir, 'videos')
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    # get srt directory
    srt_dir = os.path.join(args.data_dir, 'srt_transcripts')
    if not os.path.exists(srt_dir):
        os.makedirs(srt_dir)
    # get csv directory
    csv_dir = os.path.join(args.data_dir, 'csv_transcripts')
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    # get save directory
    save_dir = os.path.join(args.data_dir, 'word_videos')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
else:
    video_dir = args.video_dir
    srt_dir = args.srt_dir
    csv_dir = args.csv_dir
    save_dir = args.save_dir

# get list of video paths
video_paths = []
if os.path.isdir(video_dir):
    # get list of videos from start to end
    videos = get_file_list(files=os.listdir(video_dir),
                           start=args.start_date,
                           end=args.end_date)
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
                              start=args.start_date,
                              end=args.end_date)
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
    with open(os.path.join(args.data_dir, 'missing_files.txt'), 'a') as f:
        print(*missing_files, sep='\n', file=f)

# get list of csv paths
if args.save_csv:
    check_dir(csv_dir)

# check save_dir
check_dir(save_dir)


no_of_samples = 0
word_dict = dict()
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
    if args.save_csv:
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
        text = row.word

        # check word folder
        label_dir = os.path.join(save_dir, text)
        check_dir(label_dir)

        # name the video
        date = os.path.basename(video_path)[:8]
        word_dict[text] = word_dict.get(text, 0) + 1
        file_name = f'{date}{str(word_dict[text]).zfill(5)}.mp4'
        # if the name exists
        if file_name in os.listdir(label_dir):
            if args.mode == 'append':
                while file_name in os.listdir(label_dir):
                    word_dict[text] = word_dict.get(text, 0) + 1
                    file_name = f'{date}{str(word_dict[text]).zfill(5)}.mp4'
            elif args.mode == 'skip':
                continue
        file_path = os.path.join(label_dir, file_name)

        # start extracting
        try:
            cut_video(video, start, end, file_path)
            no_of_samples += 1
        except KeyboardInterrupt:
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
freq_path = os.path.join(args.data_dir, 'word_freq.csv')
freq_df = pd.DataFrame(list(word_dict.items()), columns=['word', 'frequency'])
print(f'\nThere are {freq_df.frequency.sum()} samples in total and', end=' ')
print(f'{no_of_samples} new samples from this run.')
if os.path.isfile(freq_path):
    old_freq_df = pd.read_csv(freq_path)
    freq_df = (pd
               .concat([old_freq_df, freq_df], ignore_index=True)
               .groupby('word')
               .sum())
freq_df.to_csv(freq_path, index=True)
print(f'Now there are {freq_df.frequency.sum()} samples in the database.')
print(f'The vocabularies\' frequency has been stored at: {freq_path}.')


# save list of vocabularies
vocab_path = os.path.join(args.data_dir, 'vocabs_sorted_list.txt')
vocabs = list(word_dict.keys())
old_vocabs = []
if os.path.isfile(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        old_vocabs = f.read().split()
vocabs = sorted(list(set(vocabs) | set(old_vocabs)))
with open(vocab_path, 'w', encoding='utf-8') as f:
    print(*vocabs, sep='\n', file=f)
print(f'\nThere are {len(set(word_dict.keys()) - set(vocabs))} new words.')
print(f'Now there are {len(vocabs)} vocabularies in the database.')
print(f'List of them has been stored at: {vocab_path}.')


# save error words
print(f'\nThere are {len(errors)} words that cause error.')
error_path = os.path.join(args.data_dir, 'errors.csv')
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
