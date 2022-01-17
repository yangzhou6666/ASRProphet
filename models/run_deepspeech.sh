#!/bin/bash

cd deepspeech_asr

bash scripts/infer_transcriptions_on_seed_set.sh

cd ../error_model

bash scripts/deepspeech/train_error_model.sh

bash scripts/deepspeech/infer_error_model.sh

bash scripts/deepspeech/error_model_sampling.sh

cd ../deepspeech_asr
bash scripts/finetune_on_error_model_seleced_samples.sh
bash scripts/test_ASR_finetuned_on_error_model_sents.sh
