import argparse
from speech_to_text import SpeechToText
from data_collector import DataCollector


def load_args():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('-d', '--data-dir', required=False,
                            default=r'..\..\data',
                            help='Path to the data folder')
    arg_parser.add_argument('-u', '--url', required=False,
                            default='https://www.youtube.com/watch?v=cPAlAOD-Og4&list=PL_UeYNcd7KvpDfdqPILdqdeWVeaLVsjqz',
                            help='Link to YouTube video or playlist')
    arg_parser.add_argument('-s', '--start-date',
                            default=None,
                            help='Index of video that we start processing')
    arg_parser.add_argument('-e', '--end-date',
                            default=None,
                            help='Which da')
    arg_parser.add_argument('-o', '--operation', required=False,
                            default=0, type=int,
                            help='0-all\n1-download\n2-extract audio\n3-transcribe-and-align')
    arg_parser.add_argument('-m', '--mode',
                            default='override',
                            help='How to deal with existing files \
                                override or skip')

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

    # utils.check_dir(data_dir)
    # dirs = ['videos', 'audios', 'srt_transcripts']
    # video_dir, audio_dir, srt_dir = utils.check_data_dir(data_dir, dirs)

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
