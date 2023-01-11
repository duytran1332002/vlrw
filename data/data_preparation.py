import ffmpeg
import librosa
import os
import pandas as pd
import re
import soundfile as sf
from pytube import YouTube, Playlist
from speech_to_text import SpeechToText


class DataPreparation:
    def __init__(self, video_url, data_path) -> None:
        self.video = YouTube(video_url)

        # Get video name
        date = re.search(r'(\d{1,2})\/(\d{1,2})\/(\d{4})', self.video.title)
        year = date.group(3)
        month = date.group(2) if len(
            date.group(2)) > 1 else '0' + date.group(2)
        day = date.group(1) if len(
            date.group(1)) > 1 else '0' + date.group(1)
        self.video_name = f'{year}{month}{day}.mp4'

        # Initiate data folder
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        self.data_path = data_path
        self.raw_videos_path = data_path + r'\raw_videos'
        if not os.path.exists(self.raw_videos_path):
            os.makedirs(self.raw_videos_path)
        self.videos_path = data_path + r'\videos'
        if not os.path.exists(self.videos_path):
            os.makedirs(self.videos_path)
        self.raw_audios_path = data_path + r'\raw_audios'
        if not os.path.exists(self.raw_audios_path):
            os.makedirs(self.raw_audios_path)
        self.audios_path = data_path + r'\audios'
        if not os.path.exists(self.audios_path):
            os.makedirs(self.audios_path)
        self.transcripts_path = data_path + r'\transcripts'
        if not os.path.exists(self.transcripts_path):
            os.makedirs(self.transcripts_path)
        self.aligned_transcripts_path = data_path + r'\aligned_transcripts'
        if not os.path.exists(self.aligned_transcripts_path):
            os.makedirs(self.aligned_transcripts_path)

    def __call__(self) -> None:

        # 1. Download video
        self.download()

        # 2. Trim the first minute of video and
        #    extract raw audio
        self.trim_and_extract_audio()

        # 3. Resampling raw audio that is not sampled at wanted frequency
        self.resampling(target_sampling_rate=16000)

        # 4. Transcibe the audio and align transcript
        self.transcribe_and_align()

        # 5. Convert csv to srt
        self.convert_to_srt()

    def download(self):
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
                videos_name: str
                    path to audio folder
                start: float
                    where to begin trimming
                end: float
                    where to end trimming
        '''
        raw_video_path = os.path.join(self.raw_videos_path, self.video_name)
        video_path = os.path.join(self.videos_path, self.video_name)
        raw_audio_path = os.path.join(self.raw_audios_path,
                                      self.video_name.replace('mp4', 'wav'))
        video_stream, audio_stream = self.get_video_stream(
            raw_video_path, start, end)
        # Trim first minute of the video
        if not os.path.isfile(video_path):
            print('Trimming video')
            video_output = ffmpeg.output(ffmpeg.concat(video_stream,
                                                       audio_stream,
                                                       v=1, a=1),
                                         video_path,
                                         format='mp4')
            video_output.run()
        # Extract audio
        if not os.path.isfile(raw_audio_path):
            print('Extracting audio')
            raw_audio_output = ffmpeg.output(audio_stream,
                                             raw_audio_path,
                                             format='wav')
            raw_audio_output.run()

    def get_video_stream(self, raw_video_path, start, end):
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

    def resampling(self, target_sampling_rate=16000, res_type="kaiser_best"):
        audio_name = self.video_name.replace('mp4', 'wav')
        raw_audio_path = os.path.join(self.raw_audios_path, audio_name)
        audio_path = os.path.join(self.audios_path, audio_name)
        speech_array, sampling_rate = sf.read(raw_audio_path)
        if len(speech_array.shape) == 2:
            speech_array = speech_array[..., 0]
        if sampling_rate != target_sampling_rate:
            print('Resampling audio')
            speech_array = librosa.resample(
                speech_array,
                orig_sr=sampling_rate, target_sr=target_sampling_rate,
                res_type=res_type
            )
        sf.write(audio_path, speech_array, samplerate=target_sampling_rate)

    def transcribe_and_align(self):
        audio_path = os.path.join(self.audios_path,
                                  self.video_name.replace('mp4', 'wav'))
        transcript_path = os.path.join(self.transcripts_path,
                                       self.video_name.replace('mp4', 'txt'))
        aligned_transcript_path = os.path.join(
            self.aligned_transcripts_path,
            self.video_name.replace('mp4', 'csv')
        )
        print('Transcribing and aligning')
        SpeechToText().save_result_to_file(audio_path,
                                           transcript_path,
                                           aligned_transcript_path)

    def convert_to_srt(self):
        aligned_transcript_path = os.path.join(
            self.aligned_transcripts_path,
            self.video_name.replace('mp4', 'csv')
        )
        df = pd.read_csv(self.aligned_alignement_path,
                         names=['start', 'end', 'word'])

        def sec_to_timecode(x: float) -> str:
            hour, x = divmod(x, 3600)
            minute, x = divmod(x, 60)
            second, x = divmod(x, 1)
            millisecond = int(x * 1000.)
            return '%.2d:%.2d:%.2d,%.3d' % (hour, minute, second, millisecond)

        with open(aligned_transcript_path.replace('csv', 'srt'), 'w', encoding='utf-8') as f:
            for i in range(len(df)):
                start = df["start"].values[i]
                end = df["end"].values[i]
                f.write(f'{i+1}\n')
                f.write(
                    f'{sec_to_timecode(start)} --> {sec_to_timecode(end)}\n')
                f.write(f'{df["word"].values[i]}\n\n')


if __name__ == '__main__':
    playlist = Playlist(
        'https://www.youtube.com/watch?v=cPAlAOD-Og4&list=PL_UeYNcd7KvpDfdqPILdqdeWVeaLVsjqz'
    )
    data_path = r'..\..\data'
    for video_url in playlist.video_urls:
        DataPreparation(video_url, data_path).trim_and_extract_audio()
    print('\nPreparation completed!')
