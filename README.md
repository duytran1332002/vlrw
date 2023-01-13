# Lip_Reading_THDH
# Rules
- Name of folders, files, parameters: VD: lip_reading.
- Name of functions: first word is a verb, VD: get_the_result.
- Name of classes: VD: SpeechToText.
## Folder
- data: all functions to process the data.
- preprocessing: all functiosn for proprocessing the data for training.
- model: all models are necessary for training.
```
|- data/
|- Lip_Reading_THDH/
|   |- data
|   |- preprocessing
|   |- model
```

## How to comment a function
EXAMPLE:
```python
def check_alignment(audio_path, alignment_path, save_audio=False, path_save_audio=None):
    '''
    check the result of alignment can be predict by speech to text or not
    parameter:
        audio_path: str - path audio
        alignment_path: str - path csv
        save_audio: bool - save audio or not
        path_save_audio: str - path save audio, if save_audio is True, you should give the path
    return:
        pandas table - the alignment is correct or not
    '''
    # check the audio file is exist or not
    pass
```
