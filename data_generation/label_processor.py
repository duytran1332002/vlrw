import datetime
import os
import pandas as pd
import pysrt
from tqdm import tqdm
import io
import contextlib
import shutil
from utils import *
from moviepy.video.io.VideoFileClip import VideoFileClip
from vPhon import convert_grapheme_to_phoneme
from math import floor
from random import shuffle


class LabelProcessor:
    def __init__(self, data_dir=None, video_dir=None,
                 srt_dir=None, word_video_dir=None,
                 start_date=None, end_date=None) -> None:
        self.data_dir = data_dir
        self.video_dir = video_dir
        self.srt_dir = srt_dir
        self.word_video_dir = word_video_dir
        self.start_date = start_date
        self.end_date = end_date

        self.vocab_path = os.path.join(self.data_dir, 'vocabs_sorted_list.txt')
        self.vocabs = list()
        if os.path.isfile(self.vocab_path):
            self.vocabs = read_txt_to_list(self.vocab_path)
        self.total_vocabs = 0
        self.n_new_vocab = 0

        self.grapheme_dict = dict()
        for vocab in self.vocabs:
            phon = convert_grapheme_to_phoneme(vocab)
            if self.grapheme_dict.get(phon, None) is None:
                self.grapheme_dict[phon] = vocab

        self.freq_path = os.path.join(self.data_dir, 'freq.csv')
        self.freq_dict = dict()
        if os.path.isfile(self.freq_path):
            freq_list = read_csv_to_list(self.freq_path)
            self.freq_dict = dict(convert_column_datatype(freq_list,
                                                          column=1,
                                                          datatype=int))
        self.total_samples = 0
        self.n_new_sample = 0

        self.error_path = os.path.join(data_dir, 'errors.csv')
        self.errors = list()
        if os.path.isfile(self.error_path):
            self.errors = read_csv_to_list(self.error_path)
        self.total_errors = 0
        self.n_new_error = 0

        self.n_tagged = 0
        self.n_untagged = 0

    def get_video_and_srt(self):
        self.videos, self.video_paths = self.get_file_paths(self.video_dir,
                                                            'mp4')
        self.srt_files, self.srt_paths = self.get_file_paths(self.srt_dir,
                                                             'srt')

    def check_missing_data(self):
        video_dates = [video[:8] for video in self.videos]
        srt_dates = [srt_file[:8] for srt_file in self.srt_files]
        missing_dates = sorted(find_complement(srt_dates, video_dates))
        missing_files = []
        if len(missing_dates) > 0:
            for date in missing_dates:
                if binary_seach(self.srt_files, f'{date}.srt') == -1:
                    missing_files.append(f'{date}.srt')
                    self.video_paths.remove(os.path.join(self.video_dir,
                                                         f'{date}.mp4'))
                else:
                    missing_files.append(f'{date}.mp4')
                    self.srt_paths.remove(os.path.join(self.srt_dir,
                                                       f'{date}.srt'))
            # save missing files
            missing_files_path = os.path.join(
                self.data_dir, 'missing_files.txt')
            with open(missing_files_path, 'a') as f:
                print(*missing_files, sep='\n', file=f)

    def change_boundary(self, time_boundary: datetime.time,
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

    def read_srt_to_df(self, srt_path):
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

        for sub in subs:
            # add padding to start time
            start = self.change_boundary(time_boundary=sub.start.to_time(),
                                         time_change=-0.01)

            # add padding to end time
            end = self.change_boundary(time_boundary=sub.end.to_time(),
                                       time_change=0.01)

            word = pd.DataFrame({'start': [start.strftime('%H:%M:%S.%f')],
                                'end': [end.strftime('%H:%M:%S.%f')],
                                 'word': [sub.text]})
            words = pd.concat([words, word], ignore_index=True)

        return words

    def cut_video(self, video, start, end, file_path):
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

    def get_file_paths(self, dir, ext):
        if os.path.isdir(dir):
            # get list of file paths from start to end
            files = get_file_list(files=filter_extension(dir, ext),
                                  start=self.start_date,
                                  end=self.end_date)
            return files, [os.path.join(dir, file) for file in files]
        raise Exception(f'Invalid {ext} files')

    def merge_label(self, word):
        # merge words that have similar sound
        phon = convert_grapheme_to_phoneme(word)
        if self.grapheme_dict.get(phon, None) is None:
            self.grapheme_dict[phon] = word
            return word
        return self.grapheme_dict[phon]

    def is_tagged(self, word):
        return word[-2:].startswith('_')

    def process(self, mode):
        # get videos and srt files
        self.get_video_and_srt()
        
        # check missing data
        self.check_missing_data()

        # extract word-level video
        for video_path, srt_path in tqdm(zip(self.video_paths, self.srt_paths),
                                         total=len(self.video_paths),
                                         desc='Videos',
                                         unit=' video',
                                         dynamic_ncols=True):
            # Load the video file
            video = VideoFileClip(video_path)

            # read srt to dataframe
            df = self.read_srt_to_df(srt_path)

            for _, row in tqdm(df.iterrows(),
                               total=len(df),
                               desc='Words',
                               unit=' word',
                               leave=False,
                               dynamic_ncols=True):
                self.total_samples += 1
                
                # cut the video into smaller pieces
                start = row.start
                end = row.end
                # remove tag
                word = row.word
                if self.is_tagged(word):
                    word = word[:-2]
                    self.n_tagged += 1
                # merge label
                if word == 'con':  # avoid error when uploading on cloud
                    word = 'kon'
                word = self.merge_label(word)

                # check word folder
                label_dir = os.path.join(self.word_video_dir, word)
                check_dir(label_dir)

                # name the video
                date = os.path.basename(video_path)[:8]
                if self.freq_dict.get(word, 0) == 0:
                    self.freq_dict[word] = 0
                    self.total_vocabs += 1
                # print(f'\n{word} - {date}{self.freq_dict[word]}')
                file_name = f'{date}{str(self.freq_dict[word]+1).zfill(5)}.mp4'
                # if the name exists
                if file_name in os.listdir(label_dir) and mode == 'skip':
                    continue
                file_path = os.path.join(label_dir, file_name)

                # cut video
                try:
                    self.cut_video(video, start, end, file_path)
                    self.n_new_sample += 1
                    self.freq_dict[word] += 1
                except KeyboardInterrupt:
                    print('\n')
                    os._exit(0)
                except Exception as e:
                    self.errors.append([file_name[:8], start, end, word, e])
                    self.total_errors += 1
        # clean leftover
        remove_leftover('*.mp3')

        # update vocabs
        old_vocabs_set = set(self.vocabs)
        new_vocabs_set = set(self.freq_dict.keys())
        self.n_new_vocab = len(list(new_vocabs_set - old_vocabs_set))
        self.vocabs = list(old_vocabs_set ^ new_vocabs_set)

        # save info
        save_list_to_txt(self.vocabs, self.vocab_path)
        save_list_to_csv(list(self.freq_dict.items()), self.freq_path)
        save_list_to_csv(self.errors, self.error_path)

        # print info
        self.print_process_info()
        self.print_database_info()

    def tag(self, n_class, threshold, mode, tag):
        top_freq_dict = dict(sorted(list(zip(self.freq_dict.items())),
                                    key=lambda x: x[1],
                                    reverse=True)[:n_class])

        for srt_path in tqdm(self.srt_paths,
                             total=len(self.srt_paths),
                             desc='srt files',
                             unit=' file',
                             dynamic_ncols=True):
            subs = pysrt.open(srt_path)
            for sub in tqdm(subs, leave=False):
                self.total_samples += 1

                word = sub.text
                is_tagged = False

                if self.is_tagged(word):
                    word = word[:-2]
                    is_tagged = True

                if word in top_freq_dict.keys() and top_freq_dict[word] < threshold:
                    if not is_tagged:
                        sub.text = word + '_' + tag
                    self.n_tagged += 1
                elif is_tagged and mode == 'override':
                    sub.text = word
                    self.n_untagged += 1
            subs.save(srt_path)

        # print tag info
        self.print_tag_info()
        self.print_process_info()
        self.print_database_info()

    def split_train_val_test(self, train_size, test_size):
        val_size = 1 - (train_size + test_size)

        # get list of label directories
        label_dirs = [os.path.join(self.word_video_dir, vocab)
                      for vocab in self.vocabs]

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

            # print(f'Label: {os.path.basename(label_dir)} -', end=' ')
            # print(f'sample: {n_sample} -', end=' ')
            # print(f'train: {len(train_set)} -', end=' ')
            # print(f'val: {len(val_set)} -', end=' ')
            # print(f'test: {len(test_set)}')

        # print database info
        self.print_database_info()

    def merge_train_val_test(self, label_dirs):
        for label_dir in label_dirs:
            train_dir = os.path.join(label_dir, 'train')
            val_dir = os.path.join(label_dir, 'val')
            test_dir = os.path.join(label_dir, 'test')

            for sample in os.listdir(train_dir):
                sample_path = os.path.join(train_dir, sample)
                shutil.move(sample_path, label_dir)
            for sample in os.listdir(val_dir):
                sample_path = os.path.join(val_dir, sample)
                shutil.move(sample_path, label_dir)
            for sample in os.listdir(test_dir):
                sample_path = os.path.join(test_dir, sample)
                shutil.move(sample_path, label_dir)

            # print(f'Label: {os.path.basename(label_dir)} -', end=' ')
            # print(f'sample: {len(os.listdir(label_dir))}')

    def print_process_info(self):
        print('\nIn this run,')
        print(f'    Total samples: {self.total_samples}')
        print(f'    New samples: {self.n_new_sample}')
        print(f'    Total vocabs: {self.total_vocabs}')
        print(f'    New vocabs: {self.n_new_vocab}')
        print(f'    Total errors: {self.total_errors}')
        print(f'    New errors: {self.n_new_error}')

    def print_database_info(self):
        print('\nIn database,')
        print(f'    Total samples: {sum(list(self.freq_dict.values()))}')
        print(f'        at: {self.freq_path}')
        print(f'    Total vocabs: {len(self.vocabs)}')
        print(f'        at: {self.vocab_path}')
        print(f'    Total errors: {len(self.errors)}')
        print(f'        at: {self.error_path}')

    def print_tag_info(self):
        print('\nIn this run,')
        print(f'    Total tagged samples: {self.total_samples}')
        print(f'    New tagged samples: {self.n_new_sample}')
