import contextlib
import datetime
import io
import os
import shutil
from math import floor
from random import shuffle

import pandas as pd
import pysrt
import utils
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm
from vPhon import convert_grapheme_to_phoneme


class LabelProcessor:
    def __init__(self, data_dir=None, video_dir=None,
                 srt_dir=None, word_video_dir=None,
                 start_date=None, end_date=None, n_class=0) -> None:
        self.data_dir = data_dir
        self.video_dir = video_dir
        self.srt_dir = srt_dir
        self.word_video_dir = word_video_dir
        self.start_date = start_date
        self.end_date = end_date
        self.n_class = n_class

        # get vocabs and their samples
        self.freq_path = os.path.join(self.data_dir, 'freq.csv')
        self.vocab_path = os.path.join(self.data_dir, 'vocabs_sorted_list.txt')
        freq_list = utils.read_csv_to_list(self.freq_path)
        # check n_class
        if 0 < self.n_class < len(freq_list):
            # sort freq_list and get top n_class
            freq_list = sorted(freq_list, key=lambda x: x[1], reverse=True)
            freq_list = freq_list[:self.n_class]
        self.freq_dict = dict(utils.convert_column_datatype(freq_list,
                                                            column=1,
                                                            datatype=int))
        self.total_vocabs = 0
        self.n_new_vocab = 0
        self.total_samples = 0
        self.n_new_sample = 0

        # generate grapheme_dict
        self.grapheme_dict = dict()
        for vocab in self.freq_dict.keys():
            phon = convert_grapheme_to_phoneme(vocab)
            self.grapheme_dict[phon] = self.grapheme_dict.get(phon, vocab)

        self.error_path = os.path.join(self.data_dir, 'errors.csv')
        errors = utils.read_csv_to_list(self.error_path)
        self.error_dict = {id: [start, end, word, e]
                           for id, start, end, word, e in errors}
        self.total_errors = 0
        self.n_new_error = 0

        self.n_tagged = 0
        self.n_untagged = 0

    def check_missing_data(self):
        video_dates = [video[:8] for video in self.videos]
        srt_dates = [srt_file[:8] for srt_file in self.srt_files]
        missing_dates = sorted(utils.find_complement(srt_dates, video_dates))
        missing_files = []
        if len(missing_dates) > 0:
            for date in missing_dates:
                if utils.binary_seach(self.srt_files, f'{date}.srt') == -1:
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
            files = utils.get_file_list(files=utils.filter_extension(dir, ext),
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

    def is_splitted(self, label_dir):
        train_dir = os.path.join(label_dir, 'train')
        if os.path.exists(train_dir):
            return True
        return False

    def generate_video(self, mode):
        # get videos and srt files
        self.videos, self.video_paths = self.get_file_paths(self.video_dir,
                                                            'mp4')
        self.srt_files, self.srt_paths = self.get_file_paths(self.srt_dir,
                                                             'srt')

        # check missing data
        self.check_missing_data()

        # get copy of self.freq_dict but all values = 0
        freq_dict = {key: 0 for key in self.freq_dict.keys()}

        # extract word-level video
        for video_path, srt_path in tqdm(zip(self.video_paths, self.srt_paths),
                                         total=len(self.video_paths),
                                         desc='Videos',
                                         unit=' video',
                                         dynamic_ncols=True):
            temp_freq_dict = dict()

            # Load the video file
            video = VideoFileClip(video_path)
            date = os.path.basename(video_path)[:8]

            # read srt to dataframe
            df = self.read_srt_to_df(srt_path)

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

                # remove tag
                if self.is_tagged(word):
                    word = word[:-2]

                # merge label
                if word == 'con':  # avoid error when uploading on cloud
                    word = 'kon'
                # TODO: fix 'bão' vs 'bảo'
                word = self.merge_label(word)

                # only process top n_class
                if word not in self.freq_dict.keys():
                    if self.n_class != 0:
                        continue
                    self.n_new_vocab += 1

                # update total sample and vocabs
                temp_freq_dict[word] = temp_freq_dict.get(word, 0) + 1
                self.total_samples += 1

                # check word folder
                label_dir = os.path.join(self.word_video_dir, word)
                utils.check_dir(label_dir)

                # if the label have train, val, test, merge them
                if self.is_splitted(label_dir):
                    self.merge_train_val_test(label_dir)

                # number of sample of this label
                n_sample = len(os.listdir(label_dir))

                # name the video
                id = f'{date}{str(temp_freq_dict[word]).zfill(5)}'
                sample_name = id + '.mp4'
                if sample_name in os.listdir(label_dir) and mode == 'skip':
                    continue
                sample_path = os.path.join(label_dir, sample_name)

                # cut video
                try:
                    self.cut_video(video, start, end, sample_path)
                    self.n_new_sample += 1
                    if n_sample != len(os.listdir(label_dir)):
                        freq_dict[word] = freq_dict.get(word, 0) + 1
                except KeyboardInterrupt:
                    print('\n')
                    os._exit(0)
                except Exception as e:
                    if self.error_dict.get(id, None) is None:
                        start = utils.convert_str_to_time(start)
                        start = self.change_boundary(start, 1)
                        end = utils.convert_str_to_time(end)
                        end = self.change_boundary(end, -1)
                        self.error_dict[id] = [start, end, word, e]
                        self.n_new_error += 1
                    self.total_errors += 1
        video.close()

        # clean leftover
        utils.remove_leftover('*.mp3')

        # update self.freq_dict
        self.freq_dict = utils.merge_dict(self.freq_dict, freq_dict)

        # update vocabs
        # TODO: wrong total vocab when skip
        for _, value in freq_dict.items():
            self.total_vocabs += 1 if value > 0 else 0

        # save info
        utils.save_list_to_csv(list(self.freq_dict.items()), self.freq_path)
        utils.save_list_to_txt(sorted(list(self.freq_dict.keys()),
                                      reverse=True),
                               self.vocab_path)
        errors = [[id, start, end, word, e]
                  for id, [start, end, word, e] in self.error_dict.items()]
        utils.save_list_to_csv(errors, self.error_path)

        # print info
        self.print_process_info()
        self.print_database_info()

    def tag(self, threshold, mode, tag):
        # TODO: remove code that do with n_class
        for srt_path in tqdm(self.srt_paths,
                             total=len(self.srt_paths),
                             desc='srt files',
                             unit=' file',
                             dynamic_ncols=True):
            subs = pysrt.open(srt_path)
            for sub in tqdm(subs, leave=False):
                self.total_samples += 1

                word = sub.text
                if word == 'con':  # there is only 'kon' in the record
                    word = 'kon'
                word = self.merge_label(word)

                # only process top n_class
                if word not in self.freq_dict.keys():
                    if self.n_class == 0:
                        continue

                is_tagged = False
                if self.is_tagged(word):
                    word = word[:-2]
                    is_tagged = True

                if word in self.freq_dict.keys():
                    if self.freq_dict[word] < threshold:
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
                      for vocab in self.freq_dict.keys()]

        for label_dir in tqdm(label_dirs,
                              desc='Labels',
                              total=len(label_dirs),
                              leave=True,
                              unit=' label',
                              dynamic_ncols=True):
            # merged the splitted before split again
            if self.is_splitted(label_dir):
                self.merge_train_val_test(label_dir)

            samples = utils.filter_extension(label_dir, 'mp4')
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
            utils.check_dir(train_dir)
            val_dir = os.path.join(label_dir, 'val')
            utils.check_dir(val_dir)
            test_dir = os.path.join(label_dir, 'test')
            utils.check_dir(test_dir)

            # move samples
            for sample in train_set:
                sample_path = os.path.join(label_dir, sample)
                annot_path = os.path.join(label_dir, sample[:-4] + '.txt')
                shutil.move(sample_path, train_dir)
                shutil.move(annot_path, train_dir)
            for sample in val_set:
                sample_path = os.path.join(label_dir, sample)
                annot_path = os.path.join(label_dir, sample[:-4] + '.txt')
                shutil.move(sample_path, val_dir)
                shutil.move(annot_path, val_dir)
            for sample in test_set:
                sample_path = os.path.join(label_dir, sample)
                annot_path = os.path.join(label_dir, sample[:-4] + '.txt')
                shutil.move(sample_path, test_dir)
                shutil.move(annot_path, test_dir)

            # print(f'Label: {os.path.basename(label_dir)} -', end=' ')
            # print(f'sample: {n_sample} -', end=' ')
            # print(f'train: {len(train_set)} -', end=' ')
            # print(f'val: {len(val_set)} -', end=' ')
            # print(f'test: {len(test_set)}')

        # print database info
        self.print_database_info()

    def merge_train_val_test(self, label_dir):
        train_dir = os.path.join(label_dir, 'train')
        val_dir = os.path.join(label_dir, 'val')
        test_dir = os.path.join(label_dir, 'test')

        # start moving samples out
        for sample in utils.filter_extension(train_dir, 'mp4'):
            sample_path = os.path.join(train_dir, sample)
            annot_path = os.path.join(train_dir, sample[:-4] + '.txt')
            shutil.move(sample_path, label_dir)
            shutil.move(annot_path, label_dir)
        os.rmdir(train_dir)
        for sample in utils.filter_extension(val_dir, 'mp4'):
            sample_path = os.path.join(val_dir, sample)
            annot_path = os.path.join(val_dir, sample[:-4] + '.txt')
            shutil.move(sample_path, label_dir)
            shutil.move(annot_path, label_dir)
        os.rmdir(val_dir)
        for sample in utils.filter_extension(test_dir, 'mp4'):
            sample_path = os.path.join(test_dir, sample)
            annot_path = os.path.join(test_dir, sample[:-4] + '.txt')
            shutil.move(sample_path, label_dir)
            shutil.move(annot_path, label_dir)
        os.rmdir(test_dir)

        # print(f'Label: {os.path.basename(label_dir)} -', end=' ')
        # print(f'sample: {len(os.listdir(label_dir))}')

    def generate_annotation(self, start, duration, sample_path):
        date = os.path.basename(sample_path)[:8]
        year = date[:4]
        month = date[4:6]
        day = date[6:8]
        annot_path = sample_path[:-4] + '.txt'
        with open(annot_path, 'w') as f:
            print('Disk reference: 6221443953311207281', file=f)
            print('Channel: VTV Daily Weather Forecast Livestream', file=f)
            print(f'Program start: {year}-{month}-{day} 19:00:00 +7', file=f)
            print(f'Clip start: {start} seconds', file=f)
            print(f'Duration: {duration} seconds', file=f)

    def print_process_info(self):
        print('\nIn this run,')
        print(f'    Total samples: {self.total_samples}', end='')
        print(f' - New samples: {self.n_new_sample}')
        print(f'    Total vocabs: {self.total_vocabs}', end='')
        print(f' - New vocabs: {self.n_new_vocab}')
        print(f'    Total errors: {self.total_errors}', end='')
        print(f' - New errors: {self.n_new_error}')

    def print_database_info(self):
        print('\nIn database,')
        print(f'    Total samples: {sum(list(self.freq_dict.values()))}')
        print(f'        at: {self.freq_path}')
        print(f'    Total vocabs: {len(self.freq_dict)}')
        print(f'        at: {self.vocab_path}')
        print(f'    Total errors: {len(self.error_dict)}')
        print(f'        at: {self.error_path}')

    def print_tag_info(self):
        print('\nIn this run,')
        print(f'    Total tags: {self.total_samples}', end='')
        print(f' - New tags: {self.n_new_sample}')
