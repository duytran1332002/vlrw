from transformers.file_utils import cached_path, hf_bucket_url
import os
import zipfile
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf
import torch
import kenlm
import numpy as np
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel
import IPython
from alignment import preprocess_transcript, align
import pandas as pd
import librosa


class SpeechToText:
    def __init__(self,
                 model_name_or_path="nguyenvulebinh/wav2vec2-base-vietnamese-250h",
                 lm_gram_name='vi_lm_4grams.bin.zip',
                 cache_dir='cache/',
                 **kwargs):
        # Load the model and the processor
        self.cache_dir = cache_dir
        self.model_name = model_name_or_path
        self.lm_gram_name = lm_gram_name
        self.processor = Wav2Vec2Processor.from_pretrained(
            self.model_name, cache_dir=self.cache_dir)
        self.model = Wav2Vec2ForCTC.from_pretrained(
            self.model_name, cache_dir=self.cache_dir)
        self.lm_file = hf_bucket_url(
            self.model_name, filename=self.lm_gram_name)
        self.lm_file = cached_path(self.lm_file, cache_dir=self.cache_dir)
        # check the zip file is extracted or not
        if not os.path.exists(self.cache_dir + self.lm_gram_name[:-4]):
            with zipfile.ZipFile(self.lm_file, 'r') as zip_ref:
                zip_ref.extractall(self.cache_dir)
        self.lm_file = self.cache_dir + self.lm_gram_name[:-4]
        self.ngram_lm_model = self.get_decoder_ngram_model(
            self.processor.tokenizer, self.lm_file)

    # map array to speech and get simple rate
    def map_to_array(self, batch):
        speech, sampling_rate = sf.read(batch["file"])
        if len(speech.shape) == 1:
            speech = speech[np.newaxis, :]
        batch["speech"] = speech
        batch["sampling_rate"] = sampling_rate
        return batch

    def get_decoder_ngram_model(self, tokenizer, ngram_lm_path):
        vocab_dict = tokenizer.get_vocab()
        sort_vocab = sorted((value, key)
                            for (key, value) in vocab_dict.items())
        vocab = [x[1] for x in sort_vocab][:-2]
        vocab_list = vocab
        # convert ctc blank character representation
        vocab_list[tokenizer.pad_token_id] = ""
        # replace special characters
        vocab_list[tokenizer.unk_token_id] = ""
        # vocab_list[tokenizer.bos_token_id] = ""
        # vocab_list[tokenizer.eos_token_id] = ""
        # convert space character representation
        vocab_list[tokenizer.word_delimiter_token_id] = " "
        # specify ctc blank char index,
        # since conventially it is the last entry of the logit matrix
        alphabet = Alphabet.build_alphabet(
            vocab_list, ctc_token_idx=tokenizer.pad_token_id)
        lm_model = kenlm.Model(ngram_lm_path)
        decoder = BeamSearchDecoderCTC(alphabet,
                                       language_model=LanguageModel(lm_model))
        return decoder

    def _transcribe_with_vector(self, audio_vector, ds):
        # check cuda is available or not
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            input_values = self.processor(
                audio_vector,
                sampling_rate=ds["sampling_rate"],
                return_tensors="pt"
            ).input_values.to("cuda")
            self.model.to("cuda")
        else:
            input_values = self.processor(
                audio_vector,
                sampling_rate=ds["sampling_rate"],
                return_tensors="pt"
            ).input_values
        logits = self.model(input_values).logits[0]
        beam_search_output = self.ngram_lm_model.decode(
            logits.cpu().detach().numpy(), beam_width=500)

        # clean up
        del input_values, logits

        # empty cache
        torch.cuda.empty_cache()

        return beam_search_output

    def _transcribe_with_path(self, audio_path='data/speechtotext/test.wav'):
        # check the audio file is exist or not
        if not os.path.exists(audio_path):
            print('File not found')
            return None

        # load dummy dataset and read soundfiles
        ds = self.map_to_array({"file": audio_path})
        ds['speech'] = np.mean(ds['speech'], axis=1)
        audio_list = []

        # Resampling audio
        ds['speech'], ds['sampling_rate'] = self.resampling(ds['speech'],
                                                            ds['sampling_rate'])

        # trim audio to each 10s
        f = 160000
        while len(ds['speech']) > f:
            audio_list.append(ds['speech'][:f])
            ds['speech'] = ds['speech'][f:]
        audio_list.append(ds['speech'])

        # get the predictions
        transcript = ''
        alignment = pd.DataFrame(columns=['start', 'end', 'word'])

        for trim in range(len(audio_list)):
            try:
                line = self._transcribe_with_vector(
                    audio_vector=audio_list[trim], ds=ds)
                print("Beam search output: {}".format(transcript))

                # align transcript if there speech in the audio
                if len(transcript.strip()) > 0:
                    new_rows = self._align_with_vector(audio_list[trim],
                                                       ds['sampling_rate'],
                                                       transcript)
                    new_rows.loc[:, ['start', 'end']] += trim * 10
                    alignment = pd.concat([alignment, new_rows],
                                          ignore_index=True)

                transcript += " " + line
            except RuntimeError:
                pass

        return transcript, alignment

    def _align_with_vector(self, audio: np.ndarray,
                           sampling_rate: int,
                           transcript: str) -> pd.DataFrame:
        '''
            Align each word in transcript with each word in the speech

            Parameters:
                audio: numpy.ndarray
                    Audio array
                sampling_rate: int
                    Sampling rate of the audio
                transcript: str
                    Transcript of the speech

            Returns:
                new_rows: pandas.DataFrame
                    Aligned words
        '''
        words, transcript_p, idx_word_p, idx_line_p = preprocess_transcript(
            transcript)
        word_align, words = align(
            audio, sampling_rate,
            words, transcript_p, idx_word_p, idx_line_p,
            method='MTL_BDR', cuda=True
        )
        resolution = 256 / sampling_rate * 3
        word_time = np.array(word_align, dtype=np.float64)
        word_time = word_time * resolution
        new_rows = pd.DataFrame({'start': word_time[:, 0],
                                 'end': word_time[:, 1],
                                 'word': words})
        return new_rows

    def print_result(self, audio_path: str) -> None:
        '''
            Play the audio

            Parameters:
                audio_path: str
                    Path to the audio
        '''
        print(self._transcribe_with_path(audio_path))
        IPython.display.Audio(audio_path)

    def save_result_to_file(self, audio_path: str,
                            transcript_path: str,
                            csv_transcript_path: str) -> None:
        '''
            Save transcript as text file and aligned transcript as csv file

            Parameters:
                audio_path: str
                    Path to the audio file
                transcript_path: str
                    Path to save transcript text file
                aligned_transcript_path: str
                    Path to save aligned transcript csv file
        '''
        transcript, alignment = self._transcribe_with_path(audio_path)
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(transcript)
        print("Save transcript to file: ", transcript_path)
        alignment.to_csv(csv_transcript_path,
                         encoding='utf-8', index=False)
        print("Save transcript to file: ", csv_transcript_path)

    @staticmethod
    def resampling(speech_array: np.ndarray,
                   sampling_rate: int,
                   target_sampling_rate: int = 16000,
                   res_type: str = "kaiser_best") -> tuple:
        '''
            Resample the audio

            Parameters:
                speech_array: np.ndarray
                    Audio's values matrix
                sampling_rate: int
                    Sampling rate of the audio
                target_sampling_rate: int (default=16000)
                    Sampling rate that the audio is resampled as
                res_type: str (default="kaiser_best")
                    Resample type

            Returns:
                speech_array: np.ndarray
                    Audio's values matrix after being resampled
                target_sampling_rate: int
                    Sampling rate that the audio is resampled as
        '''
        if len(speech_array.shape) == 2:
            speech_array = speech_array[..., 0]
        if sampling_rate != target_sampling_rate:
            speech_array = librosa.resample(
                speech_array,
                orig_sr=sampling_rate,
                target_sr=target_sampling_rate,
                res_type=res_type
            )
        return speech_array, target_sampling_rate
