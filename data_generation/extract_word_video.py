import argparse
import datetime
import os
import pandas as pd
import pysrt
from utils import binary_seach
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
    parser.add_argument('--csv-dir',
                        default=None,
                        help='csv directory')
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
    parser.add_argument('--fix-error',
                        default=False,
                        help='Fix error word')

    args = parser.parse_args()
    return args


def change_boundary(time_boundary, time_change):
    '''
    Change the boundary of a time object

    Parameters:
        time_boundary: datetime.time
        time_change: int
            the change of time boundary in seconds

    returns:
        datetime.time
    '''
    dt = datetime.datetime.combine(datetime.date.today(), time_boundary)
    if time_change > 0:
        dt += datetime.timedelta(seconds=time_change)
    else:
        dt -= datetime.timedelta(seconds=-time_change)

    return dt.time()


def convert_srt_to_csv(srt_path, csv_path):
    '''
    Convert srt file to csv file

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

    for sub in subs:
        # add padding to start time
        start = change_boundary(time_boundary=sub.start.to_time(),
                                time_change=-0.01)

        # add padding to end time
        end = change_boundary(time_boundary=sub.end.to_time(),
                              time_change=0.01)

        word = pd.DataFrame({'start': [start],
                             'end': [end],
                             'word': [sub.text]})
        words = pd.concat([words, word], ignore_index=True)
    words.to_csv(csv_path, index=False)
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
    piece = video.subclip(start, end)
    piece.write_videofile(file_path, fps=video.fps)


def check_dir(dir):
    '''
    Check if the directory exists, if not, create it

    Parameters:
        dir: str
            the directory to be checked
    '''
    if not os.path.exists(dir):
        os.makedirs(dir)


def find_differences(arr_a, arr_b, complement='both'):
    '''
    Identify different elements

    Parameters:
        arr_a: list
            list of elements a
        arr_b: list
            list of elements b
        complement: str
            'both': return elements in arr_a and in arr_b except the union
            'a': return elements in arr_a but not in arr_b
            'b': return elements in arr_b but not in arr_a

    Returns:
        list
            list of different elements
    '''
    assert complement in ['both', 'a', 'b'], 'Invalid mode'
    set_a, set_b = set(arr_a), set(arr_b)
    intersection_set = set_a & set_b
    if complement == 'a':
        return list(set_a - intersection_set)
    elif complement == 'b':
        return list(set_b - intersection_set)
    return list(set_a ^ set_b)


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
    videos = os.listdir(video_dir)
    # implement binary search to find start video
    if args.start_date is not None:
        start_idx = binary_seach(videos, args.start_date + '.mp4')
        if start_idx == -1:
            raise Exception('Invalid start date')
    else:
        start_idx = 0
    # implement binary search to find end video
    if args.end_date is not None:
        end_idx = binary_seach(videos, args.end_date + '.mp4')
        if end_idx == -1:
            raise Exception('Invalid end date')
    else:
        end_idx = len(videos) - 1
    # get list of videos from start to end
    videos = videos[start_idx:end_idx+1]
    video_paths = [os.path.join(video_dir, video)
                   for video in videos]
elif os.path.isfile(video_dir):
    video_paths.append(video_dir)
else:
    raise Exception('Invalid video(s)')

# get list of srt paths
srt_paths = []
if os.path.isdir(srt_dir):
    srt_files = os.listdir(srt_dir)
    # implement binary search to find start srt file
    if args.start_date is not None:
        start_idx = binary_seach(srt_files, args.start_date + '.srt')
        if start_idx == -1:
            raise Exception('Invalid start date')
    else:
        start_idx = 0
    # implement binary search to find end srt file
    if args.end_date is not None:
        end_idx = binary_seach(srt_files, args.end_date + '.srt')
        if end_idx == -1:
            raise Exception('Invalid end date')
    else:
        end_idx = len(srt_files) - 1
    # get list of srt files from start to end
    srt_files = srt_files[start_idx:end_idx+1]
    srt_paths = [os.path.join(srt_dir, srt_file)
                 for srt_file in srt_files]
elif os.path.isfile(srt_dir):
    srt_paths.append(srt_dir)
else:
    raise Exception('Invalid srt(s)')

# check missing data
video_dates = [video[:8] for video in videos]
srt_dates = [srt_file[:8] for srt_file in srt_files]
missing_dates = sorted(find_differences(srt_dates, video_dates))
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
check_dir(csv_dir)
csv_paths = [os.path.join(csv_dir, srt_name.replace('srt', 'csv'))
             for srt_name in os.listdir(srt_dir)]

# check save_dir
check_dir(save_dir)

# extract word-level video
word_dict = dict()
errors = pd.DataFrame(columns=['date', 'start', 'end', 'word'])
for video_path, srt_path, csv_path in zip(video_paths, srt_paths, csv_paths):
    # Load the video file
    video = VideoFileClip(video_path)

    # read srt to dataframe
    df = convert_srt_to_csv(srt_path, csv_path)
    # TODO: avoid read from disk
    df = pd.read_csv(csv_path, encoding='utf-8')
    # print(f'{type(df.start.iloc[0])} - {type(df.end.iloc[0])}')

    for _, row in df.iterrows():
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
        except KeyboardInterrupt:
            os._exit(0)
        except Exception as e:
            print(f'Error when process {file_name}')
            print(e)
            # print(f'{type(row.start)} - {type(row.end)}')
            error_df = pd.DataFrame({'date': [file_name[:8]],
                                     'start': [row.start],
                                     'end': [row.end],
                                     'word': [row.word]})
            errors = pd.concat([errors, error_df], ignore_index=True)

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
print(f'\nList of {len(vocabs)} vocabularies has been stored at: {vocab_path}')

# save error words
print(f'\nThere are {len(errors)} words that cause error.')
error_path = os.path.join(args.data_dir, 'errors.csv')
if os.path.isfile(error_path):
    old_errors = pd.read_csv(error_path)
    errors = pd.concat([old_errors, errors], ignore_index=True)
    errors = errors.drop_duplicates()
errors.to_csv(error_path, encoding='utf-8', index=False)
print(f'List of these error words has been stored at: {error_path}')

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
