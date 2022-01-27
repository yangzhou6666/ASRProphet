# ASRDebugger

This repository provides an implementation of experiments in our [FSE-2022 paper]()

# Requirements
This code was developed with python 3.8.9. <br/>
Create a new virtual environment and install the dependencies by running `bash install_requirements.sh`

# Dataset

### 1. Librispeech

Librispeech dataset is used in our experiments is downloaded from [Librispeech OpenSLR](https://www.openslr.org/12/). The dataset is Open Source, and contains 1000 hours of speeches. 

Prepare librispeech test-clean, test-other, dev-clean, dev-other inside this folder (`data/Librispeech/test-clean`, `data/Librispeech/test-other`, `data/Librispeech/dev-clean`, `data/Librispeech/dev-other`).

```
cd data/Librispeech/

# download test-clean, test-other, dev-clean, dev-other
wget https://www.openslr.org/resources/12/test-clean.tar.gz 
wget https://www.openslr.org/resources/12/test-other.tar.gz 
wget https://www.openslr.org/resources/12/dev-clean.tar.gz 
wget https://www.openslr.org/resources/12/dev-other.tar.gz 

# extract the tar.gz dataset
tar -zxvf test-clean.tar.gz 
tar -zxvf test-other.tar.gz 
tar -zxvf dev-clean.tar.gz 
tar -zxvf dev-other.tar.gz 
```

Prepare `manifests` for the dataset.

```
bash generate_data.sh
```

### 2. Librispeech Train

Prepare Librispeech-train dataset
```
cd data/LibrispeechTrain/

# download train-100
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz

# extract the tar.gz dataset

```

Prepare `manifests` for the next experiments.

```
bash generate_data.sh
```


### 3. ST-AEDS

A free American English corpus by Surfingtech (www.surfing.ai), containing utterances from 10 speakers, Each speaker has about 350 utterances.

Prepare ST-AEDS dataset
```
cd data/ST-AEDS/

# download the dataset
wget https://www.openslr.org/resources/45/ST-AEDS-20180100_1-OS.tgz

# extract the .tgz dataset


```

Please rename the extracted folder into `data`, thus each audio files and the transcription (text.txt) located in `data/ST-AEDS/data/`

Prepare `manifests` for the next experiments.

```
bash generate_data.sh
```

### 4. l2arctic

At the current stage (for better sharing within the collaborators), we upload the dataset to Google Drive. The dataset can be download using a command:

Prepare l2arctic dataset
```
cd data/l2arctic/

# download the dataset
gdown https://drive.google.com/uc?id=1kRA5HgGijT8LhjoQb19ez98fKrJ_9z1r

# unzip the file
unzip l2arctic_release_v5.zip

# Process the dataset
python prepare_l2arctic.py
```

Then, we processed dataset will be stored under `data/l2arctic/processed`.


# ASR Models

### 1. QuartzNet

QuartzNet ASR models are available to use directly after cloning the repository

### 2. Deepspeech

Download the model and the scorer

```
mkdir models/pretrained_checkpoints/deepspeech/

cd models/pretrained_checkpoints/deepspeech/

wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm

wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer
```

Download the checkpoint for fine-tuning

```
mkdir models/pretrained_checkpoints/deepspeech/checkpoints/
cd models/pretrained_checkpoints/deepspeech/checkpoints/

wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-checkpoint.tar.gz

# extract the tar.gz file
```

Prepare code for fine-tuning

```
cd models/deepspeech_asr/

git clone https://github.com/mhilmiasyrofi/FineTuneDeepSpeech

#read the readme from the project
#build the docker image
cd FineTuneDeepSpeech
docker build -t <your username>/traindeepspeech .

#run the docker
docker run --name gpu1-deepspeech --rm -it --gpus '"device=1"' -v <absolute path to FineTuneDeepSpeech>:/DeepSpeech -v <absolute path to ASRDebugger>:<absolute path to ASRDebugger>  <your username>/traindeepspeech /bin/bash
```



# Usage
  * Generate transcripts for the seed+dev set using the pre-trainded ASR (Transcripts are used while training error models)
    ```
    cd models/quartznet_asr
    bash scripts/infer_transcriptions_on_seed_set.sh
    ```
  * Train error model by aligning the references and generated transcripts for seed+dev set
    ```
    cd models/error_model
    bash train_error_model.sh
    ```
  * Infer the trained error model on the set of sentences from which we wish to do the selection
    ```
    cd models/error_model
    bash infer_error_model.sh
    ```
  * Select the sentences using the error model as proposed in our paper (Equation-2 and Algorithm-1)
    ```
    cd models/error_model
    bash error_model_sampling.sh
    ```
  * Finetune and Test the ASR on the sentences selected via error model
    ```
    cd models/quartznet_asr
    bash scripts/finetune_on_error_model_seleced_samples.sh
    bash scripts/test_ASR_finetuned_on_error_model_sents.sh
    ```
  * Finetune and Test the ASR on the randomly selected sentences
    ```
    cd models/quartznet_asr
    bash scripts/finetune_on_randomly_seleced_samples.sh
    bash scripts/test_ASR_finetuned_on_random_sents.sh
    ```

# Contents
  * `data/$accent/manifests`:
    - `all.json`: All the samples for a given accented speaker
    - `seed.json`: Randomly selected seed set for learning error model. Also used with selected samples while training ASR.
    - `dev.json`: Dev set used while training ASR. Also used with seed set while training error models
    - `seed_plus_dev.json`: Used for training error models (Concatenation of `seed.json` and `dev.json`)
    - `selection.json`: Sentences are selected from this file (either randomly or through error model, depending on selection startegy)
    - `test.json`: Used for evaluating the error model.
    - `train/error_model/$size/seed_"$seed"/train.json`: Contains `size` number of sentences selected via error model from `selection.json`, appended with `seed.json` for training the ASR model. `seed` represents an independent run.
    - `train/random/$size/seed_"$seed"/train.json`: Contains `size` number of sentences selected randomly from `selection.json`, appended with `seed.json` for training the ASR model. `seed` represents an independent run.
  * `data/indicTTS_audio`:
    - Audio files for each speaker are placed here.
    - After obtaining `.zip` files from [IndicTTS team](https://www.iitm.ac.in/donlab/tts/contact.php), place them in this folder.
    - Run `unzip_indic.sh` to unzip the speaker data of your choice.
  * `models/pretrained_checkpoints/quartznet`: Pretrained Quartznet models:
    - `librispeech/quartznet.pt`: Quartznet ckpt trained on 960hrs librispeech
    - `librispeech_mcv_others/quartznet.pt`: Quartznet ckpt trained on 960hrs librispeech + Mozilla Common Voice + Few other corpus
    - The above checkpoints were originally provided by NVIDIA to work with NeMO toolkit. We modified these slightly to make them work with Jasper's pytorch code.
    - Link to original checkpoints: [Librispeech](https://ngc.nvidia.com/catalog/models/nvidia:quartznet_15x5_ls_sp), [Librispeech+MCV+Others](https://ngc.nvidia.com/catalog/models/nvidia:tlt-jarvis:speechtotext_english_quartznet)
  * `models/pretrained_checkpoints/error_models`
    - `$accent/seed_"$seed"/best/ErrorClassifierPhoneBiLSTM_V2.pt` : Pre-trained error model on `seed_plus_dev.json` outputs.
    - `librispeech/seed_"$seed"/best/ErrorClassifierPhoneBiLSTM_V2.pt`: Pre-trained error model on `dev+test` sets of librispeech.
  * `models/error_model` : Code related to error model
    - `train_error_model.sh`: Trains the error model using the ASR's transcripts on the seed+dev set and gold references
    - `infer_error_model.sh`: Dumps aggregated error probabilities from the error model for sentence scoring
    - `error_model_sampling.sh`: Samples the sentences using error model outputs as per Equation-2 and Algorithm-1 in our paper.
  * `models/quartznet_asr`: Code related to training and inference of [Quartznet model](https://arxiv.org/pdf/1910.10261.pdf), adapted from Nvidia's [implementation of Jasper in PyTorch](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechRecognition/Jasper)
    - `scripts/finetune_on_error_model_seleced_samples.sh`: Finetunes the ASR on sentences utterances selected by error model
    - `scripts/finetune_on_randomly_seleced_samples.sh`: Finetunes the ASR on randomly selected sentences
    - `scripts/test_ASR_finetuned_on_error_model_sents.sh`: Infers the ASR finetuned on error model selected sentences
    - `scripts/test_ASR_finetuned_on_random_sents.sh`: Infers the ASR on randomly selected sentences
    - `scripts/infer_transcriptions_on_seed_set.sh`: Infers the ASR on seed+dev set. Inferred outputs used for training the error model.

# Acknowledgement
  * Dataset for a wide variety of Indian accents was provided by [IndicTTS](https://www.iitm.ac.in/donlab/tts/contact.php) team at [IIT Madras](https://www.iitm.ac.in/). Dataset for other accents was obtained from [L2-Arctic Corpus](https://psi.engr.tamu.edu/l2-arctic-corpus/)
  * Code for Quartznet ASR model is adapted from Nvidia's [implementation of Jasper in PyTorch](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechRecognition/Jasper). Pretrained quartznet checkpoints were downloaded from [here](https://ngc.nvidia.com/catalog/models/nvidia:quartznet_15x5_ls_sp) and modified slightly to make them compatible with Jasper's PyTorch implementation.
  * [Power-ASR](https://github.com/NickRuiz/power-asr) is used for aligning ASR-generated transcripts and references for obtaining error labels.
  * Park et al's [Grapheme-to-Phoneme](https://github.com/Kyubyong/g2p) model is used for converting graphemes to phonemes as an input to the error model . 
