# What I use
- Torch==1.13.1
- Conda==4.13.0
- Linux==22.04.1
- Cuda==11.6
# Set up
I sugget that you should use conda environment that I have introduced how to install it below.
Install all the neccessary library by the code below:
```
pip install -r requirements.txt
```
You also need to install torch library, I have introduced how to install it below.
# Using GPU
## Install conda
You can use conda environment by the code below
```
# Install Miniconda
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
## Install torch
### Install with conda
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```
### Install with pip
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```
You can see detail here: https://pytorch.org/get-started/locally/

# Using CPU
## Install torch
### Install with conda
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
### Install with pip
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```
# Quick start
```python
from speech_to_text import SpeechToText
stt = SpeechToText()
# print and show the audio
stt.print_result('/home/duy1332002/Desktop/Lip_Reading_THDH/data/speechtotext/t5.wav')
# save the result to file
stt.save_result_to_file('/home/duy1332002/Desktop/Lip_Reading_THDH/data/speechtotext/t5.wav', '/home/duy1332002/Desktop/Lip_Reading_THDH/data/speechtotext/test1.txt')
```

## Extract word video
How to use extract_word_video.py to cut video of each word based on its duration in .srt file:
```shell
python extract_word_video.py --data-dir <path_to_data_directory> \
                             --video-dir <path_to_video_or_video_folder> \
                             --srt-dir <path_to_srt_or_srt_folder> \
                             --csv-dir <path_to_csv_or_csv_folder> \
                             --save-dir <path_to_word_video_or_word_video_folder> \
                             --start-date <which date to start> \
                             --end-date <which date to stop> \
                             --mode <how_you_want_to_handle_duplicates> \
```
Notes:
* If `data-dir` is provided, there is no need to provide `video-dir`, `srt-dir`, `csv-dir` and `save-dir`. However, `data-dir` must have at least the structure as follow. Its sub-folder must be remained the exact name.
```
|- data/
|   |- videos/
|   |- srt_transcripts/
|   |- csv_transcripts/ (optional)
|   |- word_videos/ (optional)
```
* The `mode` includes:
  * `override`: override existing files
  * `skip`: if there is an existing file, do nothing
  * `append`: if there is an existing file, change the name of the duplicate

For example:
```shell
python extract_word_video.py --data-dir lipreading/data \
                             --start-date 20220701 \
                             --end-date 20220806 \
                             --mode override \
```

# Citation

[![CITE](https://zenodo.org/badge/DOI/10.5281/zenodo.5356039.svg)](https://github.com/vietai/ASR)

```text
@misc{Thai_Binh_Nguyen_wav2vec2_vi_2021,
  author = {Thai Binh Nguyen},
  doi = {10.5281/zenodo.5356039},
  month = {09},
  title = {{Vietnamese end-to-end speech recognition using wav2vec 2.0}},
  url = {https://github.com/vietai/ASR},
  year = {2021}
}
