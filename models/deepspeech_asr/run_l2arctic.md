# Scripts to run experiments using Deepspeech ASR on L2Arctic

The following code takes `ASI` in the L2Arctic dataset as an example. Please kindly change 

# 1. Infer Deepspeech ASR on the dataset

## 1.1 Infer on `seed_plus_dev.json`
Generate transcripts for the seed+dev set using the pre-trainded ASR (Transcripts are used while training error models)

```
DATA=$(cd ../../data/l2arctic/processed; pwd)
WAV_DIR=$(cd ../../data/l2arctic/; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd) 
declare -a accents=('ASI' 'RRBI')
for accent in "${accents[@]}"
do
  mkdir -p $DATA/$accent/manifests/deepspeech_outputs
  echo $accent
  echo $WAV_DIR/$accent/wav
  python3 -u inference.py \
  --wav_dir=$WAV_DIR \
  --output_file=$DATA/$accent/manifests/deepspeech_outputs/seed_plus_dev_out.txt \
  --val_manifest=$DATA/$accent/manifests/seed_plus_dev.json \
  --model=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.pbmm \
  --scorer=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.scorer \
  --model_tag=deepspeech-0.9.3 \
  > $DATA/$accent/manifests/deepspeech_outputs/seed_plus_dev_infer_log.txt
done
```

## 1.2 Infer on `seed.json` and `dev.json`

The results will be used for training word error predictor.

```
for accent in 'ASI' 'RRBI'
do
  mkdir -p $DATA/$accent/manifests/deepspeech_outputs
  echo $accent
  echo $WAV_DIR/$accent/wav 
  python3 -u inference.py \
  --wav_dir=$WAV_DIR \
  --output_file=$DATA/$accent/manifests/deepspeech_outputs/seed_out.txt \
  --val_manifest=$DATA/$accent/manifests/seed.json \
  --model=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.pbmm \
  --scorer=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.scorer \
  --model_tag=deepspeech-0.9.3 \
  > $DATA/$accent/manifests/deepspeech_outputs/seed_infer_log.txt

  python3 -u inference.py \
  --wav_dir=$WAV_DIR \
  --output_file=$DATA/$accent/manifests/deepspeech_outputs/dev_out.txt \
  --val_manifest=$DATA/$accent/manifests/dev.json \
  --model=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.pbmm \
  --scorer=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.scorer \
  --model_tag=deepspeech-0.9.3 \
  > $DATA/$accent/manifests/deepspeech_outputs/dev_infer_log.txt
done
```


## Infer on test dataset

Let's try on the test set.

```
for accent in "${accents[@]}"
do
  mkdir -p $DATA/$accent/manifests/deepspeech_outputs
  echo $accent
  echo $WAV_DIR/$accent/wav 
  python3 -u inference.py \
  --wav_dir=$WAV_DIR \
  --output_file=$DATA/$accent/manifests/deepspeech_outputs/original_test_out_out.txt \
  --val_manifest=$DATA/$accent/manifests/test.json \
  --model=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.pbmm \
  --scorer=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.scorer \
  --model_tag=deepspeech-0.9.3 \
  > $DATA/$accent/manifests/deepspeech_outputs/original_test_infer_log.txt
done
```

Let's also try on the synthetic test set.


```
for accent in "${accents[@]}"
do
  mkdir -p $DATA/$accent/manifests/deepspeech_outputs
  echo $accent
  echo $WAV_DIR/$accent/wav 
  python3 -u inference.py \
  --wav_dir=$WAV_DIR \
  --output_file=$DATA/$accent/manifests/deepspeech_outputs/original_test_out_tts.txt \
  --val_manifest=$DATA/$accent/manifests/test_tts.json \
  --model=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.pbmm \
  --scorer=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.scorer \
  --model_tag=deepspeech-0.9.3 \
  > $DATA/$accent/manifests/deepspeech_outputs/original_test_infer_log_tts.txt
done
```

```
for accent in "${accents[@]}"
do
  mkdir -p $DATA/$accent/manifests/deepspeech_outputs
  echo $accent
  echo $WAV_DIR/$accent/wav 
  python3 -u inference.py \
  --wav_dir=$WAV_DIR \
  --output_file=$DATA/$accent/manifests/deepspeech_outputs/seed_plus_dev_out_tts.txt \
  --val_manifest=$DATA/$accent/manifests/seed_plus_dev_tts.json \
  --model=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.pbmm \
  --scorer=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.scorer \
  --model_tag=deepspeech-0.9.3 \
  > $DATA/$accent/manifests/deepspeech_outputs/seed_plus_dev_infer_log_tts.txt
done
```

# 2. Train Error Model

## 2.1 Train word error predictor

```
for seed in 1 2 3
do
  for accent in "ASI" "RRBI"
  do
  mkdir -p $PRETRAINED_CKPTS/word_error_predictor/deepspeech/$accent/seed_"$seed"/best
  python word_error_predictor.py \
    --train_path=$DATA/$accent/manifests/deepspeech_outputs/seed_out.txt \
    --test_path=$DATA/$accent/manifests/deepspeech_outputs/dev_out.txt \
    --output_dir=$PRETRAINED_CKPTS/word_error_predictor/deepspeech/$accent/seed_"$seed"/best \
    > $PRETRAINED_CKPTS/word_error_predictor/deepspeech/$accent/seed_"$seed"/training.log
  done
done
```

## ICASSP Baseline
Please go to `/model/error_model` first.

Notice: this step needs the hypotheses file (i.e, `seed_plus_dev_out.txt`) infered in previous steps. 

```
for seed in 1 2 3
do
  for accent in "ASI" "RRBI"
  do
    LR=3e-4
    echo $accent seed $seed
    mkdir -p $PRETRAINED_CKPTS/error_models/deepspeech/$accent/seed_"$seed"/
    python3 -u train_error_model.py \
      --batch_size=10 \
      --num_epochs=200 \
      --train_freq=20 \
      --lr=$LR \
      --num_layers=4 \
      --hidden_size=64 \
      --input_size=64 \
      --weight_decay=0.001 \
      --train_portion=0.8 \
      --hypotheses_path=$DATA/$accent/manifests/deepspeech_outputs/seed_plus_dev_out.txt \
      --lr_decay=warmup \
      --seed=1 \
      --output_dir=$PRETRAINED_CKPTS/error_models/deepspeech/$accent/seed_"$seed"/recent \
      --best_dir=$PRETRAINED_CKPTS/error_models/deepspeech/$accent/seed_"$seed"/best \
    > $PRETRAINED_CKPTS/error_models/deepspeech/$accent/seed_"$seed"/train_log.txt
  done 
done 
```
Assuming that the seed is `1`, the model will be saved under `/models/pretrained_checkpoints/error_model/deepspeech/ASI/seed_1/`, and the train log is stored in the same folder. 

## Infer the error model on the dataset for selection

```
for seed in 1 2 3
do
  for accent in "${accents[@]}"
  do
    echo $accent seed $seed
    python3 -u infer_error_model.py \
      --batch_size=64 \
      --num_layers=4 \
      --hidden_size=64 \
      --input_size=64 \
      --json_path=$DATA/$accent/manifests/selection.json \
      --pretrained_ckpt=$PRETRAINED_CKPTS/error_models/deepspeech/$accent/seed_"$seed"/best/ErrorClassifierPhoneBiLSTM_V2.pt \
      --output_dir=$PRETRAINED_CKPTS/error_models/deepspeech/$accent/seed_"$seed"/best \
    > $PRETRAINED_CKPTS/error_models/deepspeech/$accent/seed_"$seed"/infer_log.txt
  echo
  done
done
```

## Training the common error predictor

```
echo 
for seed in 1 2 3
do
  for accent in 'ASI' 'RRBI'
  do
    LR=3e-4
    echo $accent seed $seed
    mkdir -p $PRETRAINED_CKPTS/matcher/deepspeech/$accent/seed_"$seed"/
    CUDA_VISIBLE_DEVICES=6 python3 -u train_style_matcher.py \
      --batch_size=10 \
      --num_epochs=200 \
      --train_freq=20 \
      --lr=$LR \
      --num_layers=4 \
      --hidden_size=64 \
      --input_size=64 \
      --weight_decay=0.001 \
      --train_portion=0.8 \
      --asr_hypotheses_path=$DATA/$accent/manifests/deepspeech_outputs/seed_plus_dev_out.txt \
      --tts_hypotheses_path=$DATA/$accent/manifests/deepspeech_outputs/seed_plus_dev_out_tts.txt \
      --lr_decay=warmup \
      --seed=1 \
      --output_dir=$PRETRAINED_CKPTS/matcher/deepspeech/$accent/seed_"$seed"/recent \
      --best_dir=$PRETRAINED_CKPTS/matcher/deepspeech/$accent/seed_"$seed"/best 
  echo
  done 
done 
```

```
for seed in 1 2 3
do
  for accent in 'ASI' 'RRBI'
  do
    echo $accent seed $seed
    CUDA_VISIBLE_DEVICES=2 python3 -u infer_matcher.py \
      --batch_size=64 \
      --num_layers=4 \
      --hidden_size=64 \
      --input_size=64 \
      --json_path=$DATA/$accent/manifests/selection.json \
      --pretrained_ckpt=$PRETRAINED_CKPTS/matcher/deepspeech/$accent/seed_"$seed"/best/ErrorClassifierPhoneBiLSTM_V2.pt \
      --output_dir=$PRETRAINED_CKPTS/matcher/deepspeech/$accent/seed_"$seed"/best \
    > $PRETRAINED_CKPTS/matcher/deepspeech/$accent/seed_"$seed"/infer_log.txt
  echo
  done
done
```

## Training the error estimator from ASREvolve


for seed in 1 2 3
do
  for accent in 'ASI' 'RRBI'
  do
    echo $accent seed $seed
    mkdir -p $PRETRAINED_CKPTS/asrevolve_error_models/deepspeech/$accent/seed_"$seed"
    CUDA_VISIBLE_DEVICES=2 python3 -u train_error_model_asrevolve.py \
      --train_path=$DATA/$accent/manifests/deepspeech_outputs/seed_out.txt \
      --test_path=$DATA/$accent/manifests/deepspeech_outputs/dev_out.txt \
      --seed=1 \
      --output_dir=$PRETRAINED_CKPTS/asrevolve_error_models/deepspeech/$accent/seed_"$seed"/best \
      --log_dir=$PRETRAINED_CKPTS/asrevolve_error_models/deepspeech/$accent/seed_"$seed"/train_log \
      > $PRETRAINED_CKPTS/asrevolve_error_models/deepspeech/$accent/seed_"$seed"/train_log.txt 
  done 
done 

## Using ASREvolve for sampling

```
for seed in 1 2 3
do
  for num_sample in 50 75 100 150 200 300 400 500
  do
    for accent in 'ASI' 'RRBI'
    do
      echo $accent seed $seed
      CUDA_VISIBLE_DEVICES=6 python3 -u error_model_sampling_asrevolve.py \
        --seed_json_file=$DATA/$accent/manifests/seed.json \
        --random_json_file=$DATA/$accent/manifests/train/random_tts/"$num_sample"/seed_"$seed"/train.json \
        --selection_json_file=$DATA/$accent/manifests/selection_tts.json \
        --finetuned_ckpt=$PRETRAINED_CKPTS/asrevolve_error_models/deepspeech/$accent/seed_"$seed"/best \
        --log_dir=$PRETRAINED_CKPTS/asrevolve_error_models/deepspeech/$accent/seed_"$seed"/train_log \
        --num_sample=$num_sample \
        --output_json_path=$DATA/$accent/manifests/train/deepspeech/asrevolve_error_model \
        --exp_id=$seed
    done
  done
done
```

### Measure original model performance on the selected audio

```
for seed in 1 2 3
do
  for size in 50 75 100 150 200 300 400 500
  do
    for accent in 'ASI' 'RRBI'
    do
      echo $accent $seed $size
      python3 -u inference.py \
      --wav_dir=$WAV_DIR \
      --val_manifest=$DATA/$accent/manifests/train/deepspeech/error_model/$size/seed_"$seed"/train.json \
      --output_file=$DATA/$accent/manifests/train/deepspeech/error_model/$size/seed_"$seed"/test_out_ori.txt \
      --model=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.pbmm \
      --scorer=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.scorer \
      --model_tag=deepspeech-0.9.3 \
      > $DATA/$accent/manifests/train/deepspeech/error_model/$size/seed_"$seed"/test_out_ori_log.txt
    done
  done
done
```

```
for seed in 1 2 3
do
  for size in 50 75 100 150 200 300 400 500
  do
    for accent in 'ASI' 'RRBI'
    do
      echo $accent $seed $size
      python3 -u inference.py \
      --wav_dir=$WAV_DIR \
      --val_manifest=$DATA/$accent/manifests/train/deepspeech/error_model_tts/$size/seed_"$seed"/train.json \
      --output_file=$DATA/$accent/manifests/train/deepspeech/error_model_tts/$size/seed_"$seed"/test_out_ori.txt \
      --model=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.pbmm \
      --scorer=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.scorer \
      --model_tag=deepspeech-0.9.3 \
      > $DATA/$accent/manifests/train/deepspeech/error_model_tts/$size/seed_"$seed"/test_out_ori_log.txt
    done
  done
done
```

```
for seed in 1 2 3
do
  for size in 50 75 100 150 200 300 400 500
  do
    for accent in 'ASI' 'RRBI'
    do
      echo $accent $seed $size
      python3 -u inference.py \
      --wav_dir=$WAV_DIR \
      --val_manifest=$DATA/$accent/manifests/train/deepspeech/asrevolve_error_model/$size/seed_"$seed"/train.json \
      --output_file=$DATA/$accent/manifests/train/deepspeech/asrevolve_error_model/$size/seed_"$seed"/test_out_ori.txt \
      --model=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.pbmm \
      --scorer=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.scorer \
      --model_tag=deepspeech-0.9.3 \
      > $DATA/$accent/manifests/train/deepspeech/asrevolve_error_model/$size/seed_"$seed"/test_out_ori_log.txt
    done
  done
done
```

## Train on ASREvolve data


Please ensure the `gpu_id` with the docker container name

```
for seed in 1 2 3
do 
  for size in 50 75 100 150 200 300 400 500
  do
    for accent in 'ASI' 'RRBI'
    do
      echo $accent $seed $size
      echo
      model_dir=$PRETRAINED_CKPTS/deepspeech/finetuned/$accent/$size/seed_"$seed"/asrevolve_tts
      mkdir -p $model_dir
      CUDA_VISIBLE_DEVICES=3 python3 -u finetune.py \
        --train_manifest=$DATA/$accent/manifests/train/deepspeech/asrevolve_error_model/$size/seed_"$seed"/train.json \
        --val_manifest=$DATA/$accent/manifests/dev.json \
        --wav_dir=$WAV_DIR \
        --output_dir=$model_dir/recent \
        --load_checkpoint_dir=$PRETRAINED_CKPTS/deepspeech/checkpoints/deepspeech-0.9.3-checkpoint/ \
        --save_checkpoint_dir=$model_dir \
        --model_scorer=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.scorer \
        --gpu_id=5 \
        --num_epochs=100 \
        --learning_rate=1e-4 \
      > $model_dir/train_log.txt
    done
  done
done
```


