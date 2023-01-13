import ffmpeg
import os
import pandas as pd
import re
import time
from pytube import YouTube, Playlist
from speech_to_text import SpeechToText
import numpy as np


class DataPreparation:
    def __init__(self, video, data_path) -> None:
        '''
            Initiate properties

            Parameters:
                video: str or YouTube
                    pytube object or name of the video in local
                data_path: str
                    Path to data folder
        '''
        self.video = YouTube(video_url)

        # Get video name
        date = re.search(r'(\d{1,2})\/(\d{1,2})\/(\d{4})', self.video.title)
        year = date.group(3)
        month = date.group(2) if len(
            date.group(2)) > 1 else '0' + date.group(2)
        day = date.group(1) if len(
            date.group(1)) > 1 else '0' + date.group(1)
        self.video_name = f'{year}{month}{day}.mp4'

        # Initiate data folders
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        self.data_path = data_path
        self.raw_videos_path = data_path + r'\raw_videos'
        if not os.path.exists(self.raw_videos_path):
            os.makedirs(self.raw_videos_path)
        self.videos_path = data_path + r'\videos'
        if not os.path.exists(self.videos_path):
            os.makedirs(self.videos_path)
        self.audios_path = data_path + r'\audios'
        if not os.path.exists(self.audios_path):
            os.makedirs(self.audios_path)
        self.transcripts_path = data_path + r'\transcripts'
        if not os.path.exists(self.transcripts_path):
            os.makedirs(self.transcripts_path)
        self.csv_transcripts_path = data_path + r'\csv_transcripts'
        if not os.path.exists(self.csv_transcripts_path):
            os.makedirs(self.csv_transcripts_path)
        self.srt_transcripts_path = data_path + r'\srt_transcripts'
        if not os.path.exists(self.srt_transcripts_path):
            os.makedirs(self.srt_transcripts_path)

    def __call__(self) -> None:
        # 1. Download video
        self.download()

        # 2. Trim the first minute of video and
        #    extract audio
        self.trim_and_extract_audio()

        # 3. Transcibe the audio and align transcript
        self.transcribe_and_align()

        # 4. Convert csv to srt
        self.convert_to_srt()

    def download(self) -> None:
        '''
            Download video from YouTube
        '''
        if not os.path.isfile(os.path.join(self.raw_videos_path,
                                           self.video_name)):
            try:
                print(f'\nDownloading {self.video.title} as {self.video_name}')
                stream = self.video.streams.get_highest_resolution()
                stream.download(filename=self.video_name,
                                output_path=self.raw_videos_path)
            except Exception as e:
                print(e)

    def trim_and_extract_audio(self, start=60, end=None) -> None:
        '''
            Trim a part of the video and then extract audio

            Parameters:
                start: float
                    where to begin trimming
                end: float
                    where to end trimming
        '''
        raw_video_path = os.path.join(self.raw_videos_path, self.video_name)
        video_path = os.path.join(self.videos_path, self.video_name)
        audio_path = os.path.join(self.audios_path,
                                  self.video_name.replace('mp4', 'wav'))
        video_stream, audio_stream = self.__get_video_stream(
            raw_video_path, start, end)
        # Trim first minute of the video
        if not os.path.isfile(video_path):
            print(f'Trimming {self.video_name}')
            video_output = ffmpeg.output(ffmpeg.concat(video_stream,
                                                       audio_stream,
                                                       v=1, a=1),
                                         video_path,
                                         format='mp4')
            video_output.run()
        # Extract audio
        if not os.path.isfile(audio_path):
            print(f'Extracting audio from {self.video_name}')
            audio_output = ffmpeg.output(audio_stream,
                                         audio_path,
                                         format='wav')
            audio_output.run()

    def __get_video_stream(self, raw_video_path: str, start, end) -> tuple:
        '''
            Get video stream and audio stream

            Parameters:
                raw_video_path: str
                    Path to video
                start: float
                    Second in video to start streaming
                end: float
                    Second in video to end streaming
        '''
        input_stream = ffmpeg.input(raw_video_path)
        pts = 'PTS-STARTPTS'
        if end is None:
            end = (ffmpeg
                   .probe(raw_video_path)
                   .get('format', {})
                   .get('duration'))    # Get video's length
        video = (input_stream
                 .trim(start=start, end=end)
                 .filter('setpts', pts))
        audio = (input_stream
                 .filter('atrim', start=start, end=end)
                 .filter('asetpts', pts))
        return video, audio

    def transcribe_and_align(self):
        '''
            Transcribe the audio and align each word to the speech
        '''
        audio_name = self.video_name.replace('mp4', 'wav')
        audio_path = os.path.join(self.audios_path, audio_name)
        transcript_path = os.path.join(self.transcripts_path,
                                       self.video_name.replace('mp4', 'txt'))
        csv_transcript_path = os.path.join(
            self.csv_transcripts_path,
            self.video_name.replace('mp4', 'csv')
        )

        if not os.path.isfile(csv_transcript_path):
            print(f'Transcribing and aligning {audio_name}')
            SpeechToText().save_result_to_file(audio_path,
                                               transcript_path,
                                               csv_transcript_path)

    def convert_to_srt(self):
        '''
            Convert csv file to srt file
        '''

        def sec_to_timecode(x: float) -> str:
            '''
                Calculate timecode from second

                Parameters:
                    x: float
                        Second
            '''
            hour, x = divmod(x, 3600)
            minute, x = divmod(x, 60)
            second, x = divmod(x, 1)
            millisecond = int(x * 1000.)
            return '%.2d:%.2d:%.2d,%.3d' % (hour, minute, second, millisecond)

        csv_name = self.video_name.replace('mp4', 'csv')
        csv_path = os.path.join(
            self.csv_transcripts_path,
            csv_name
        )
        srt_name = self.video_name.replace('mp4', 'srt')
        srt_path = os.path.join(
            self.srt_transcripts_path,
            srt_name
        )

        if not os.path.isfile(srt_path):
            df = pd.read_csv(csv_path)

            print(f'Converting {csv_name} to {srt_name}')
            with open(srt_path, 'w', encoding='utf-8') as f:
                for i in range(len(df)):
                    start = df["start"].values[i]
                    end = df["end"].values[i]
                    word = df["word"].values[i]
                    f.write(f'{i+1}\n')
                    f.write(
                        f'{sec_to_timecode(start)} --> {sec_to_timecode(end)}\n')
                    f.write(f'{word}\n\n')


if __name__ == '__main__':
    playlist = Playlist(
        'https://www.youtube.com/watch?v=cPAlAOD-Og4&list=PL_UeYNcd7KvpDfdqPILdqdeWVeaLVsjqz'
    )
    data_path = r'..\..\data'
    # data_path = r'..\data'
    run_time = []
    error_videos = []
    for video_url in playlist.video_urls:
        try:
            process = DataPreparation(video_url, data_path)
            # start = time.time()
            # process.transcribe_and_align()
            # end = time.time()
            # run_time.append(end - start)
            # DataPreparation(video_url, data_path)()
        except Exception as e:
            print(f'Error when preparing {process.video_name}:')
            print(e)
            error_videos.append(video_url)

    # Time measurement
    # run_time = np.array(run_time)
    # print(f'Longest run time: {run_time.max():.2f}s')
    # print(f'Shortest run time: {run_time.min():.2f}s')
    # print(f'Average run time: {run_time.mean():.2f}s')
    # with open('error_videos.txt', 'w') as f:
    #     print(*error_videos, sep='\n', file=f)

    print('\nPreparation completed!')
