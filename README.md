# Lip_Reading_THDH
# Rules
- Name of folders, files, parameters: VD: lip_reading.
- Name of functions: first word is a verb, VD: get_the_result.
- Name of classes: VD: SpeechToText.
## Folder
- "configs": contains configuration files for all models.
- "data_generation": includes functions for repairing and processing data, from video to sample video.
- "demo_sample": provides a sample for testing purposes.
- "labels": contains the labels used for training the models.
- "lipreading": includes all necessary models and functions for model training.
- "preprocessing": includes all functions for preprocessing data before training.
- "lipreading": includes all necessary models for training.

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

# How to use
```python
pip install requirements.txt
```

link download weight of 2 models: https://drive.google.com/drive/folders/1214P6d7X_bLislVTkYV_us_j49xIXvfr?usp=share_link
Go to this link and download weight two model. Open file Lip-reading.ipynb and change path at "model_path" variable:

```python
if "__main__" == __name__:
    # -- config
    config_path = "configs/lrw_resnet18_mstcn.json"
    modality = "video"
    num_classes = 624
    model = get_model_from_json(config_path, modality, num_classes)
    # -- load model
    model_path = "path of model weight you want to test"
    model = load_model(model_path, model)
    #-- evaluate
    label_list = [line.strip() for line in open('labels/labels.txt')]
    criterion = nn.CrossEntropyLoss()
    dset_loaders = get_data_loaders(test=True, modality=modality, data_dir='demo_sample', label_path='labels/labels.txt')
    acc_avg_test, loss_avg_test = evaluate(model, label_list, dset_loaders['test'], criterion)
```
Run all function in jupyter notebook. to se the result.
