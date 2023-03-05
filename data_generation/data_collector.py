import argparse
import os
import re
import utils
from pytube import YouTube, Playlist
from speech_to_text import SpeechToText
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm


class DataCollector:
    def __init__(self, data_dir=None, video_dir=None,
                 srt_dir=None, audio_dir=None,
                 start_date=None, end_date=None, mode='skip',
                 playlist_url=None, speech2text=None) -> None:
        '''
        Initiate properties

        Parameters:
            video: str or YouTube
                pytube object or name of the video in local
            data_path: str
                Path to data folder
        '''
        self.data_dir = data_dir
        self.video_dir = video_dir
        self.srt_dir = srt_dir
        self.audio_dir = audio_dir
        self.start_date = start_date
        self.end_date = end_date

        self.playlist_url = playlist_url

        self.speech2text = speech2text

        self.mode = mode

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

    def get_url_dict(self):
        video_urls = utils.check_youtube_url(self.playlist_url).reverse()
        for video_url in video_urls:
            video = YouTube(video_url)
            date = self.get_date(video.title)
            self.url_dict[date] = video_url

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
        url_dict = self.get_url_dict()
        dates = self.get_date_list(list(url_dict.keys()),
                                   self.start_date, self.end_date)
        for date in tqdm(dates, desc='Download videos', total=len(dates),
                         unit='video', dynamic_ncols=True):
            try:
                video_path = os.path.join(self.video_dir, f'{date}.mp4')
                if os.path.isfile(video_path) and self.mode == 'skip':
                    continue
                video = YouTube(url_dict[date])
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
        videos, video_paths = self.get_file_paths(self.video_dir,
                                                  'mp4')
        for video, video_path in tqdm(zip(videos, video_paths),
                                      desc='Extract audio',
                                      total=len(videos),
                                      unit=' video',
                                      dynamic_ncols=True):
            audio_path = os.path.join(
                self.srt_dir, video.replace('wav', 'srt'))
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


def load_args():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('-u', '--url', required=False,
                            default='https://www.youtube.com/watch?v=cPAlAOD-Og4&list=PL_UeYNcd7KvpDfdqPILdqdeWVeaLVsjqz',
                            help='Link to YouTube video or playlist')
    arg_parser.add_argument('-d', '--data-dir', required=False,
                            default=r'..\..\data',
                            help='Path to the data folder')
    arg_parser.add_argument('-s', '--start-date',
                            default=None,
                            help='Index of video that we start processing')
    arg_parser.add_argument('-e', '--end-date',
                            default=None,
                            help='Which da')
    arg_parser.add_argument('-o', '--operation', required=False,
                            default=0,
                            help='0-all\n1-download\n2-extract audio\n3-transcribe-and-align')
    arg_parser.add_argument('-m', '--mode',
                            default='override',
                            help='False-offline mode\nTrue-online mode')

    return arg_parser.parse_args()


if __name__ == '__main__':
    args = load_args()

    url = args.url
    data_dir = args.data_dir
    start_date = args.start_date
    end_date = args.end_date
    operation = args.operation
    mode = args.mode

    # for debugging
    # url = args.url
    # data_dir = args.data_dir
    # start_date = args.start_date
    # end_date = args.end_date
    # operation = args.operation
    # mode = args.mode

    assert mode in ['override', 'skip'], 'Invalid mode'
    assert operation in [0, 1, 2, 3], 'Invalid operation'

    utils.check_dir(data_dir)
    dirs = ['videos', 'audios', 'srt_transcripts']
    video_dir, audio_dir, srt_dir = utils.check_data_dir(data_dir, dirs)

    speech2text = SpeechToText()

    collector = DataCollector(data_dir=data_dir,
                              start_date=start_date,
                              end_date=end_date,
                              mode=mode,
                              playlist_url=url,
                              speech2text=speech2text)
    if operation == 1:
        collector.download()
    elif operation == 2:
        collector.extract_audio()
    elif operation == 3:
        collector.transcribe_and_align()
    else:
        collector.download()
        collector.extract_audio()
        collector.transcribe_and_align()

    # error_videos = []

    # if args['mode']:
    #     videos = check_youtube_url(args['url'])
    # if not args['mode'] or videos is None:
    #     videos = os.listdir(os.path.join(data_path, 'raw_videos'))

    # if len(videos) >= args['num-videos'] > 0:
    #     n_videos = args['num-videos']
    # else:
    #     n_videos = len(videos)

    # if len(videos) > args['start-video'] >= 0:
    #     start_video = args['start-video']
    # else:
    #     start_video = 0

    # if len(videos) >= args['end-video'] > 0:
    #     end_video = args['end-video']
    # else:
    #     end_video = len(videos)

    # for i in range(start_video, end_video):
    #     try:
    #         process = DataCollector(video=videos[i],
    #                                 speech2text=speech2text,
    #                                 data_dir=data_path)
    #         if args['operation'] == 1:
    #             process.download()
    #         elif args['operation'] == 2:
    #             process.trim_and_extract_audio()
    #         elif args['operation'] == 3:
    #             process.transcribe_and_align()
    #         elif args['operation'] == 4:
    #             process.convert_to_srt()
    #         else:
    #             process()
    #     except Exception as e:
    #         print(f'Error when preparing {process.id}:')
    #         print(e)
    #         error_videos.append(process.id)

    # # Save name of videos that cause error
    # if error_videos != []:
    #     with open('error_videos.txt', 'w') as f:
    #         print(*error_videos, sep='\n', file=f)

    # print('Preparation is completed!')
