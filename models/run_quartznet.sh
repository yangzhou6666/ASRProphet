#!/bin/bash

cd quartznet_asr

bash scripts/infer_transcriptions_on_seed_set.sh

cd ../error_model

bash scripts/quartznet/train_error_model.sh

bash scripts/quartznet/infer_error_model.sh

bash scripts/quartznet/error_model_sampling.sh

cd ../quartznet_asr
bash scripts/finetune_on_error_model_seleced_samples.sh
bash scripts/test_ASR_finetuned_on_error_model_sents.sh
