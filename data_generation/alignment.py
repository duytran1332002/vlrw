import warnings
from time import time
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils as utils
from alignment_model import AcousticModel, BoundaryDetection, train_audio_transforms

np.random.seed(7)

def preprocess_from_file(audio_file, sampling_rate,
                         transcript_file, word_file=None):
    audio, _ = preprocess_audio(audio_file, sampling_rate)
    words, transcript_p, idx_word_p, idx_line_p = preprocess_transcript(
        transcript_file, word_file)
    return audio, words, transcript_p, idx_word_p, idx_line_p


def align(audio, sampling_rate, words, lyrics_p, idx_word_p, idx_line_p, method="Baseline", cuda=True):

    if len(audio.shape) == 1:
        audio = audio[np.newaxis, :]  # (channel, sample)

    # start timer
    t = time()

    # constants
    resolution = 256 / sampling_rate * 3
    alpha = 0.8

    # decode method
    if "BDR" in method:
        model_type = method[:-4]
        bdr_flag = True
    else:
        model_type = method
        bdr_flag = False
    # print("Model: {} BDR?: {}".format(model_type, bdr_flag))

    # prepare acoustic model params
    if model_type == "Baseline":
        n_class = 41
    elif model_type == "MTL":
        n_class = (41, 47)
    else:
        ValueError("Invalid model type.")

    hparams = {
        "n_cnn_layers": 1,
        "n_rnn_layers": 3,
        "rnn_dim": 256,
        "n_class": n_class,
        "n_feats": 32,
        "stride": 1,
        "dropout": 0.1
    }

    device = 'cuda' if (cuda and torch.cuda.is_available()) else 'cpu'

    ac_model = AcousticModel(
        hparams['n_cnn_layers'], hparams['rnn_dim'], hparams['n_class'],
        hparams['n_feats'], hparams['stride'], hparams['dropout']
    ).to(device)

    # Loading BDR model from checkpoint
    state = utils.load_model(
        ac_model, "./checkpoints/checkpoint_{}".format(model_type), cuda=(device == "gpu"))
    ac_model.eval()

    # reshape input, prepare mel
    x = audio.reshape(1, 1, -1)
    x = utils.move_data_to_device(x, device)
    x = x.squeeze(0)
    x = x.squeeze(1)
    x = train_audio_transforms.to(device)(x)
    x = nn.utils.rnn.pad_sequence(x, batch_first=True).unsqueeze(1)

    # predict
    all_outputs = ac_model(x)
    if model_type == "MTL":
        all_outputs = torch.sum(all_outputs, dim=3)
    all_outputs = F.log_softmax(all_outputs, dim=2)
    batch_num, output_length, num_classes = all_outputs.shape
    song_pred = all_outputs.data.cpu().numpy(
    ).reshape(-1, num_classes)  # total_length, num_classes
    total_length = int(audio.shape[1] / sampling_rate // resolution)
    song_pred = song_pred[:total_length, :]

    # smoothing
    P_noise = np.random.uniform(low=1e-11, high=1e-10, size=song_pred.shape)
    song_pred = np.log(np.exp(song_pred) + P_noise)

    if bdr_flag:
        # boundary model: fixed
        bdr_hparams = {
            "n_cnn_layers": 1,
            "rnn_dim": 32,  # a smaller rnn dim than acoustic model
            "n_class": 1,  # binary classification
            "n_feats": 32,
            "stride": 1,
            "dropout": 0.1,
        }

        bdr_model = BoundaryDetection(
            bdr_hparams['n_cnn_layers'], bdr_hparams['rnn_dim'], bdr_hparams['n_class'],
            bdr_hparams['n_feats'], bdr_hparams['stride'], bdr_hparams['dropout']
        ).to(device)
        
        # Loading BDR model from checkpoint
        state = utils.load_model(
            bdr_model, "./checkpoints/checkpoint_BDR", cuda=(device == "gpu"))
        bdr_model.eval()

        # get boundary prob curve
        bdr_outputs = bdr_model(x).data.cpu().numpy().reshape(-1)
        # apply log
        bdr_outputs = np.log(bdr_outputs) * alpha

        line_start = [d[0] for d in idx_line_p]

        # start alignment
        word_align, score = utils.alignment_bdr(
            song_pred, lyrics_p, idx_word_p, bdr_outputs, line_start)
    else:
        # start alignment
        word_align, score = utils.alignment(song_pred, lyrics_p, idx_word_p)

    t = time() - t
    # print("Alignment Score:\t{}\tTime:\t{}".format(score, t))

    return word_align, words


def preprocess_audio(audio_file, sampling_rate=16000):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        audio, sampling_rate = librosa.load(
            audio_file, sr=sampling_rate, mono=True, res_type='kaiser_best')
    if len(audio.shape) == 1:
        audio = audio[np.newaxis, :]  # (channel, sample)
    return audio, sampling_rate


def preprocess_transcript(raw_lines, word_file=None):
    # with open(lyrics_file, 'r', encoding='utf-8') as f:
    #     raw_lines = f.read().splitlines()
    # process raw
    raw_lines = [raw_lines]
    raw_lines = [" ".join(line.split()) for line in raw_lines if len(line) > 0]
    # concat
    transcript = " ".join(raw_lines)

    if word_file:
        with open(word_file, encoding='utf-8') as f:
            words_lines = f.read().splitlines()
    else:
        words_lines = transcript.split()

    lyrics_p, words_p, idx_word_p, idx_line_p = utils.gen_phone_gt(
        words_lines, raw_lines)

    return words_lines, lyrics_p, idx_word_p, idx_line_p


def write_csv(sampling_rate, pred_file, word_align, words):
    resolution = 256 / sampling_rate * 3

    with open(pred_file, 'w', encoding='utf-8') as f:
        for j in range(len(word_align)):
            word_time = word_align[j]
            f.write("{},{},{}\n".format(
                word_time[0] * resolution, word_time[1] * resolution, words[j]))


# TODO: complete this code
def write_srt():
    pass
