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
for accent in 'ASI' 'RRBI'
do
  mkdir -p $DATA/$accent/manifests/deepspeech_outputs
  echo $accent
  echo $WAV_DIR/$accent/wav 
  python3 -u inference.py \
  --wav_dir=$WAV_DIR \
  --output_file=$DATA/$accent/manifests/deepspeech_outputs/original_test_out.txt \
  --val_manifest=$DATA/$accent/manifests/test.json \
  --model=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.pbmm \
  --scorer=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.scorer \
  --model_tag=deepspeech-0.9.3 \
  > $DATA/$accent/manifests/deepspeech_outputs/original_test_infer_log.txt
done
```


# 2. Train Error Model

## 2.1 Word error predictor

### 2.1.1 Train

```
for seed in 1 2 3
do
  for accent in 'ASI' 'RRBI'
  do
  mkdir -p $PRETRAINED_CKPTS/word_error_predictor/deepspeech/$accent/seed_"$seed"/best
  CUDA_VISIBLE_DEVICES=6 python word_error_predictor.py \
    --train_path=$DATA/$accent/manifests/deepspeech_outputs/seed_out.txt \
    --test_path=$DATA/$accent/manifests/deepspeech_outputs/dev_out.txt \
    --output_dir=$PRETRAINED_CKPTS/word_error_predictor/deepspeech/$accent/seed_"$seed"/best \
    > $PRETRAINED_CKPTS/word_error_predictor/deepspeech/$accent/seed_"$seed"/training.log
  done
done
```

Evaluate the model's performance, including accuracy and precision.

```
for seed in 1 2 3
do
  for accent in 'ASI' 'RRBI'
  do
  echo $accent seed $seed
  echo 
  CUDA_VISIBLE_DEVICES=6 python evaluate_word_err_model.py \
    --train_path=$DATA/$accent/manifests/deepspeech_outputs/seed_out.txt \
    --test_path=$DATA/$accent/manifests/deepspeech_outputs/dev_out.txt \
    --finetuned_ckpt=$PRETRAINED_CKPTS/word_error_predictor/deepspeech/$accent/seed_"$seed"/best
  done
done
```

### 2.1.2 Sample


```
for seed in 1 2 3
do
  for accent in 'ASI' 'RRBI'
  do
    echo $accent seed $seed
    CUDA_VISIBLE_DEVICES=6 python3 -u word_error_sampling.py \
      --seed_json_file=$DATA/$accent/manifests/seed.json \
      --data_folder=$DATA/$accent/manifests/train/random/ \
      --selection_json_file=$DATA/$accent/manifests/selection.json \
      --finetuned_ckpt=$PRETRAINED_CKPTS/word_error_predictor/deepspeech/$accent/seed_"$seed"/best \
      --output_json_path=$DATA/$accent/manifests/train/deepspeech/word_error_predictor_real \
      --seed=$seed &
  done
done
```

### 2.1.3 Evaluate

```
for seed in 1 2 3
do
  for size in 50 75 100 150 200 300 400 500
  do
    for accent in 'ASI' 'RRBI'
    do
      for method in "word_enhance"
      do
        echo $accent seed $seed size $size method $method
        python3 -u inference.py \
        --wav_dir=$WAV_DIR \
        --output_file=$DATA/$accent/manifests/train/deepspeech/word_error_predictor_real/$size/$method/seed_"$seed"/test_out_ori.txt \
        --val_manifest=$DATA/$accent/manifests/train/deepspeech/word_error_predictor_real/$size/$method/seed_"$seed"/train.json \
        --model=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.pbmm \
        --scorer=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.scorer \
        --model_tag=deepspeech-0.9.3 \
        > $DATA/$accent/manifests/train/deepspeech/word_error_predictor_real/$size/$method/seed_"$seed"/test_out_ori_log.txt
      done
    done
  done
done
```

## 2.2 ICASSP

## 2.2.1 Train

```
for seed in 1 2 3
do
  for accent in 'ASI' 'RRBI'
  do
    LR=3e-4
    echo $accent seed $seed
    mkdir -p $PRETRAINED_CKPTS/error_models/deepspeech/$accent/seed_"$seed"/
    CUDA_VISIBLE_DEVICES=3 python3 -u train_error_model.py \
      --batch_size=16 \
      --train_path=$DATA/$accent/manifests/deepspeech_outputs/seed_out.txt \
      --test_path=$DATA/$accent/manifests/deepspeech_outputs/dev_out.txt \
      --lr_decay=warmup \
      --seed=$seed \
      --output_dir=$PRETRAINED_CKPTS/error_models/deepspeech/$accent/seed_"$seed"/recent \
      --best_dir=$PRETRAINED_CKPTS/error_models/deepspeech/$accent/seed_"$seed"/best \
    > $PRETRAINED_CKPTS/error_models/deepspeech/$accent/seed_"$seed"/train_log.txt
  done 
done 
``` 


## 2.2.2 Sample

Before using the error model to select test cases, we need to first infer the model on all the texts and store the results.

```
for seed in 1 2 3
do
  for accent in 'ASI' 'RRBI'
  do
    echo $accent seed $seed
    CUDA_VISIBLE_DEVICES=3 python3 -u infer_error_model.py \
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

```
for seed in 1 2 3
do
  for accent in 'ASI' 'RRBI'
  do
    echo $accent seed $seed
    CUDA_VISIBLE_DEVICES=6 python3 -u error_model_sampling.py \
      --selection_json_file=$DATA/$accent/manifests/selection.json \
      --seed_json_file=$DATA/$accent/manifests/seed.json \
      --error_model_weights=$PRETRAINED_CKPTS/error_models/deepspeech/$accent/seed_"$seed"/best/weights.pkl \
      --random_json_path=$DATA/$accent/manifests/train/random \
      --output_json_path=$DATA/$accent/manifests/train/deepspeech/error_model \
      --exp_id=$seed
  done
done
```

## 2.2.2 Evaluate

```
for seed in 1 2 3
do
  for size in 50 75 100 150 200 300 400 500
  do
    for accent in 'ASI' 'RRBI'
    do
      echo $accent $seed $size
      echo 
      python3 -u inference.py \
        --wav_dir=$WAV_DIR \
        --output_file=$DATA/$accent/manifests/train/deepspeech/error_model/$size/seed_"$seed"/test_out_ori.txt \
        --val_manifest=$DATA/$accent/manifests/train/deepspeech/error_model/$size/seed_"$seed"/train.json \
        --model=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.pbmm \
        --scorer=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.scorer \
        --model_tag=deepspeech-0.9.3 \
        > $DATA/$accent/manifests/train/deepspeech/error_model/$size/seed_"$seed"/test_out_ori_log.txt
    done
  done 
done
```


# 3. Fine-Tune ASR Models

Please ensure the `gpu_id` is exaactly the same with the docker container name

## Train on ICASSP sampling (real)

```
for seed in 1 2 3
do 
  for size in 50 75 100 150 200 300 400 500
  do
    for accent in 'ASI' 'RRBI'
    do
      echo $accent $seed $size
      echo 
      echo
      model_dir=$PRETRAINED_CKPTS/deepspeech/finetuned/$accent/$size/seed_"$seed"/icassp_real_mix
      rm -r $model_dir/
      mkdir -p $model_dir/checkpoints/
      cp -r $PRETRAINED_CKPTS/deepspeech/checkpoints/deepspeech-0.9.3-checkpoint/* $model_dir/checkpoints/
      CUDA_VISIBLE_DEVICES=3 python3 -u finetune.py \
        --train_manifest=$DATA/$accent/manifests/train/deepspeech/error_model/$size/seed_"$seed"/train.json \
        --val_manifest=$DATA/$accent/manifests/dev.json \
        --wav_dir=$WAV_DIR \
        --checkpoint_dir=$model_dir/checkpoints/ \
        --export_dir=$model_dir/best \
        --model_scorer=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.scorer \
        --gpu_id=5 \
        --num_epochs=100 \
        --learning_rate=1e-4 \
      > $model_dir/train_log.txt
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
      echo 
      echo
      model_dir=$PRETRAINED_CKPTS/deepspeech/finetuned/$accent/$size/seed_"$seed"/icassp_real_mix
      mkdir -p $model_dir
      python3 -u inference.py \
      --wav_dir=$WAV_DIR \
      --val_manifest=$DATA/$accent/manifests/test.json \
      --output_file=$model_dir/test_out.txt \
      --model=$model_dir/best/output_graph.pbmm \
      --model_tag=deepspeech-finetuned-icassp_real_mix-size_"$size"-seed_"$seed" \
      --scorer=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.scorer \
      --overwrite \
      > $model_dir/test_infer_log.txt
    done 
  done
done
```


## Train on Word Error Data (real)

```
for seed in 1 2 3
do 
  for size in 50 75 100 150 200 300 400 500
  do
    for accent in 'ASI' 'RRBI'
    do
      echo $accent $seed $size
      echo 
      echo
      model_dir=$PRETRAINED_CKPTS/deepspeech/finetuned/$accent/$size/seed_"$seed"/word_error_real_mix/word_enhance
      rm -r $model_dir/
      mkdir -p $model_dir/checkpoints/
      cp -r $PRETRAINED_CKPTS/deepspeech/checkpoints/deepspeech-0.9.3-checkpoint/* $model_dir/checkpoints/
      CUDA_VISIBLE_DEVICES=3 python3 -u finetune.py \
        --train_manifest=$DATA/$accent/manifests/train/deepspeech/word_error_predictor_real/$size/word_enhance/seed_"$seed"/train.json \
        --val_manifest=$DATA/$accent/manifests/dev.json \
        --wav_dir=$WAV_DIR \
        --checkpoint_dir=$model_dir/checkpoints/ \
        --export_dir=$model_dir/best \
        --model_scorer=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.scorer \
        --gpu_id=5 \
        --num_epochs=100 \
        --learning_rate=1e-4 \
      > $model_dir/train_log.txt
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
      echo 
      echo
      model_dir=$PRETRAINED_CKPTS/deepspeech/finetuned/$accent/$size/seed_"$seed"/word_error_real_mix/word_enhance
      mkdir -p $model_dir
      python3 -u inference.py \
      --wav_dir=$WAV_DIR \
      --val_manifest=$DATA/$accent/manifests/test.json \
      --output_file=$model_dir/test_out.txt \
      --model=$model_dir/best/output_graph.pbmm \
      --model_tag=deepspeech-finetuned-word_error_real_mix-size_"$size"-seed_"$seed" \
      --scorer=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.scorer \
      --overwrite \
      > $model_dir/test_infer_log.txt
    done 
  done
done
```