import os
import re
import utils
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm


class DataCollector:
    def __init__(self, data_dir=None, start_date=None, end_date=None,
                 mode='skip', playlist_url=None, speech2text=None) -> None:
        '''
        Initiate properties

        Parameters:
            video: str or YouTube
                pytube object or name of the video in local
            data_path: str
                Path to data folder
        '''
        self.data_dir = data_dir
        self.video_dir, self.audio_dir, self.srt_dir = self.set_up_dir()

        self.start_date = start_date
        self.end_date = end_date

        self.playlist_url = playlist_url

        self.speech2text = speech2text

        self.mode = mode

    def set_up_dir(self):
        utils.check_dir(self.data_dir)
        dirs = ['videos', 'audios', 'srt_transcripts']
        return utils.check_data_dir(self.data_dir, dirs)

    def get_file_paths(self, dir, ext):
        """_summary_

        Parameters:
            dir (_type_): _description_
            ext (_type_): _description_

        Raises:
            Exception: _description_

        Returns:
            _type_: _description_
        """
        print(dir)
        if os.path.isdir(dir):
            # get list of file paths from start to end
            files = utils.get_file_list(files=utils.filter_extension(dir, ext),
                                        start=self.start_date,
                                        end=self.end_date)
            return files, [os.path.join(dir, file) for file in files]
        raise Exception(f'Invalid {ext} files')

    def get_date(self, title):
        date = re.search(r'(\d{1,2})\/(\d{1,2})\/(\d{4})', title)
        year = date.group(3)
        month = date.group(2) if len(
            date.group(2)) > 1 else '0' + date.group(2)
        day = date.group(1) if len(
            date.group(1)) > 1 else '0' + date.group(1)
        return f'{year}{month}{day}'

    def get_video_dict(self):
        video_dict = dict()
        videos = utils.check_youtube_url(self.playlist_url)
        videos.reverse()
        for video in videos:
            date = self.get_date(video.title)
            video_dict[date] = video
        return video_dict

    def get_date_list(self, dates: list, start=None, end=None):
        '''
        Get particular files (from start to end) in an ordered list.

        Parameters:
            files: list
                list of files
            start: str
                start file
            end: str
                end file

        Returns:
            list
                list of files from start to end
        '''
        # implement binary search to find start srt file
        if start is not None and start >= dates[0]:
            start_idx = utils.binary_seach(dates, start)
            if start_idx == -1:
                raise Exception('Invalid start date')
        else:
            start_idx = 0

        # implement binary search to find end srt file
        if end is not None and end <= dates[-1]:
            end_idx = utils.binary_seach(dates, end)
            if end_idx == -1:
                raise Exception('Invalid end date')
        else:
            end_idx = len(dates) - 1

        dates = dates[start_idx:end_idx+1]

        # check if the date in list is valid
        for date in dates:
            if not utils.is_valid_date(date):
                raise Exception(f'Invalid date or wrong pattern: {date}')

        return dates

    def download(self) -> None:
        '''Download videos from YouTube
        '''
        video_dict = self.get_video_dict()
        dates = self.get_date_list(list(video_dict.keys()),
                                   self.start_date, self.end_date)
        for date in tqdm(dates, desc='Download videos', total=len(dates),
                         unit='video', dynamic_ncols=True):
            try:
                video_path = os.path.join(self.video_dir, f'{date}.mp4')
                if os.path.isfile(video_path) and self.mode == 'skip':
                    continue
                video = video_dict[date]
                stream = video.streams.get_highest_resolution()
                stream.download(filename=f'{date}.mp4',
                                output_path=self.video_dir)
            except Exception as e:
                print(e)

    def extract_audio(self) -> None:
        '''Extract audio

        Parameters:
            start: float
                where to begin trimming
            end: float
                where to end trimming
        '''
        print(self.video_dir)
        videos, video_paths = self.get_file_paths(self.video_dir,
                                                  'mp4')
        for video, video_path in tqdm(zip(videos, video_paths),
                                      desc='Extract audio',
                                      total=len(videos),
                                      unit=' video',
                                      dynamic_ncols=True):
            audio_path = os.path.join(
                self.audio_dir, video.replace('mp4', 'wav'))
            if os.path.isfile(audio_path) and self.mode == 'skip':
                continue
            video = VideoFileClip(video_path)
            video = video.audio
            video.write_audiofile(audio_path)

    def transcribe_and_align(self):
        '''Transcribe the audio and align each word to the speech
        '''
        audios, audio_paths = self.get_file_paths(self.audio_dir, 'wav')
        for audio, audio_path in tqdm(zip(audios, audio_paths),
                                      desc='Transcribe',
                                      total=len(audios),
                                      unit=' video',
                                      dynamic_ncols=True):
            srt_path = os.path.join(self.srt_dir, audio.replace('wav', 'srt'))
            if os.path.isfile(srt_path) and self.mode == 'skip':
                continue
            self.speech2text.save_alignment_to_srt(audio_path,
                                                   srt_path)
