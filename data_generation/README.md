# Introduction

To generate lip reading dataset, the process includes these following steps:

1. Video data is collected from online source like YouTube, TikTok,...
2. Speech2Text is performed to generate the transcript for each video.
3. Alignment model is used to generate the timestamp for each word in the transcript.
4. Timestamps are manually corrected by using [SubtitleEdit](https://www.nikse.dk/subtitleedit).
5. Samples are extracted from video by following the processed timestamps, and annotations are produced. Then, they are divided into training, validation and test sets.

In these steps, step 1, 2, 3 and 5 can be automatically achieved by following the next sections.

# Set up data directory

The directory in data generation step will follow the below rules. All you need to do is create an empty `data` directory in advance. The rest will be created automatically.

```
|- data/
|   |- vocabs_sorted_list.txt (generated when run)
|   |- missing_files.txt (generated when run)
|   |- errors.csv (generated when run)
|   |- videos/
|   |- audios/
|   |- srt_transcripts/
|   |- word_videos/
|       |-<label>/
|           |-train/
|               |-<date><id>.mp4
|           |-val/
|               |-<date><id>.mp4
|           |-test/
|               |-<date><id>.mp4
|   |- annotations/
|       |-<label>/
|           |-train/
|               |-<date><id>.txt
|           |-val/
|               |-<date><id>.txt
|           |-test/
|               |-<date><id>.txt
```

# Steps

Before going to the first step, make sure that

- You are in `data_generation` directory of the project:

```shell
cd data_generation
```

- Data directory and its subdirectories are set up as in the previous section.
- All packages and libraries in requirements (in this directory) are installed.

## 1. Download and Alignment

Here is how to use collect_data.py download videos, extract audio and transcribe videos:

```shell
python collect_data.py --data-dir <path_to_data_directory> \
                       --url <link_to_youtube_playlist_or_video> \
                       --start-date <which_date_to_start> \
                       --end-date <which_date_to_stop> \
                       --operation <operation_to_perform> \
                       --mode <mode_to_run>
```

Notes:

- Link to the VTV weather forecast live stream playplist is the default value of `url`.
- `start_date` and `end_date` are in `YYYYMMDD` format
- `operation` includes:
  - Download video(s): `--operation 1`
  - Extract audio: `--operation 2`
  - Transcribe and align: `--operation 3`
  - All above: `--operation 0`
- Except extracting audio, transcribing and align, other operations require internet connection.
- `mode` includes:
  - `override`: override existing files (default).
  - `skip`: if there is an existing file, do nothing.

For example, to do all operations:

```shell
python collect_data.py --data-dir ..\data \
                       --start-date 20230404 \
                       --end-date 20230404 \
```

## 2. Extract samples

Here is how to use extract_word_video.py to cut video of each word based on its duration in .srt file:

```shell
python extract_word_video.py --data-dir <path_to_data_directory> \
                             --start-date <which_date_to_start> \
                             --end-date <which_date_to-stop> \
                             --mode <how_you_want_to_handle_duplicates> \
                             --train-ratios <proportion_of_training_set> \
                             --test-ratios <proportion_of_test_set> \
                             --thresholds <number_of_samples_determining_the_ratio>
```

Notes:
- `mode` includes:
  - `override`: override existing files (default)
  - `skip`: if there is an existing file, do nothing
- `train-ratios` have input of a list. The default value is [0.7, 0.8, 0.9]
- `test-ratios` have input of a list. The default value is [0.15, 0.1, 0.05]
- The ratio of validation set will be decided based on ratios of training set and test set.
- `thresholds` have input of a list. The default value is [100, 1000]

For example:

```shell
python extract_word_video.py --data-dir ..\data \
                             --start-date 20220701 \
                             --end-date 20220806 \
                             --mode override \
```

Explanation for why there are different dividing ratios for different labels is that our dataset retains the natural distribution of the word. This also means that the more frequent a word's appearance is, the more importance it is. As a result, the more important labels is prioritized to have more samples in training set than others.

# References

[![CITE](https://zenodo.org/badge/DOI/10.5281/zenodo.5356039.svg)](https://github.com/vietai/ASR)

```text
@misc{Thai_Binh_Nguyen_wav2vec2_vi_2021,
  author = {Thai Binh Nguyen},
  doi = {10.5281/zenodo.5356039},
  month = {09},
  title = {{Vietnamese end-to-end speech recognition using wav2vec 2.0}},
  url = {https://github.com/vietai/ASR},
  year = {2021}
},

@misc{vPhon
  author = {Kirby, James},
  title = {vPhon: a Vietnamese phonetizer (version 2.1.1)}
  url = {http://github.com/kirbyj/vPhon/},
  year = {2018}
},

@article{mtl_alignment,
  author    = {Jiawen Huang and
               Emmanouil Benetos and
               Sebastian Ewert},
  title     = {Improving Lyrics Alignment through Joint Pitch Detection},
  journal   = {CoRR},
  volume    = {abs/2202.01646},
  year      = {2022},
  url       = {https://arxiv.org/abs/2202.01646},
  eprinttype = {arXiv},
  eprint    = {2202.01646},
  timestamp = {Wed, 09 Feb 2022 15:43:34 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2202-01646.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
