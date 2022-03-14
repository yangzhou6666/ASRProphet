# Scripts to run experiments on L2Arctic

The following code takes `ASI` in the L2Arctic dataset as an example. Please kindly change 

# 1. Infer ASR on dataset

## 1.1 Infer on `seed_plus_dev.json`
Generate transcripts for the seed+dev set using the pre-trainded ASR (Transcripts are used while training error models)

```
DATA=$(cd ../../data/l2arctic/processed; pwd)
WAV_DIR=$(cd ../../data/l2arctic/; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd) 
declare -a accents=('ASI' 'RRBI')


for accent in "TNI" "SVBI" "ASI" "RRBI"
do
  mkdir -p $DATA/$accent/manifests/quartznet_outputs
  echo $accent
  echo $WAV_DIR/$accent/wav 
  CUDA_VISIBLE_DEVICES=1 python3 -u inference.py \
  --batch_size=16 \
  --output_file=$DATA/$accent/manifests/quartznet_outputs/seed_plus_dev_out.txt \
  --wav_dir=$WAV_DIR \
  --val_manifest=$DATA/$accent/manifests/seed_plus_dev.json \
  --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
  --ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
  > $DATA/$accent/manifests/quartznet_outputs/seed_plus_dev_infer_log.txt &
done
```


## 1.2 Infer on `seed.json` and `dev.json`

The results will be used for training word error predictor.

```
for accent in "TNI" "SVBI" "ASI" "RRBI"
do
  mkdir -p $DATA/$accent/manifests/quartznet_outputs
  echo $accent
  echo $WAV_DIR/$accent/wav 
  CUDA_VISIBLE_DEVICES=1 python3 -u inference.py \
  --batch_size=16 \
  --output_file=$DATA/$accent/manifests/quartznet_outputs/seed_out.txt \
  --wav_dir=$WAV_DIR \
  --val_manifest=$DATA/$accent/manifests/seed.json \
  --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
  --ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
  > $DATA/$accent/manifests/quartznet_outputs/seed_infer_log.txt 

  CUDA_VISIBLE_DEVICES=1 python3 -u inference.py \
  --batch_size=16 \
  --output_file=$DATA/$accent/manifests/quartznet_outputs/dev_out.txt \
  --wav_dir=$WAV_DIR \
  --val_manifest=$DATA/$accent/manifests/dev.json \
  --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
  --ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
  > $DATA/$accent/manifests/quartznet_outputs/dev_infer_log.txt 
done &
```



## Infer on test dataset

Let's try on the test set.


```
for accent in "TNI" "SVBI" "ASI" "RRBI"
do
  mkdir -p $DATA/$accent/manifests/quartznet_outputs
  echo $accent
  echo $WAV_DIR/$accent/wav 
  CUDA_VISIBLE_DEVICES=1 python3 -u inference.py \
  --batch_size=16 \
  --output_file=$DATA/$accent/manifests/quartznet_outputs/original_test_out.txt \
  --wav_dir=$WAV_DIR \
  --val_manifest=$DATA/$accent/manifests/test.json \
  --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
  --ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
  > $DATA/$accent/manifests/quartznet_outputs/original_test_infer_log.txt 
done &
```

# 2. Train Error Model

## 2.1 Word error predictor

### 2.1.1 Training



```
for seed in 1 2 3
do
  for accent in "THV"
  do
  mkdir -p $PRETRAINED_CKPTS/word_error_predictor/quartznet/$accent/seed_"$seed"/best
  echo $accent seed $seed
  echo 
  CUDA_VISIBLE_DEVICES=7 python word_error_predictor.py \
    --train_path=$DATA/$accent/manifests/quartznet_outputs/seed_out.txt \
    --test_path=$DATA/$accent/manifests/quartznet_outputs/dev_out.txt \
    --output_dir=$PRETRAINED_CKPTS/word_error_predictor/quartznet/$accent/seed_"$seed"/best \
    --seed=$seed \
     > $PRETRAINED_CKPTS/word_error_predictor/quartznet/$accent/seed_"$seed"/training.log &
  done
done 
```



### 2.1.2 Sampling `real` audio and evaluation

```
for seed in 1 2 3
do
  for accent in "THV"
  do
    echo $accent seed $seed
    CUDA_VISIBLE_DEVICES=5 python3 -u word_error_sampling.py \
      --seed_json_file=$DATA/$accent/manifests/seed.json \
      --data_folder=$DATA/$accent/manifests/train/random/ \
      --selection_json_file=$DATA/$accent/manifests/selection.json \
      --finetuned_ckpt=$PRETRAINED_CKPTS/word_error_predictor/quartznet/$accent/seed_"$seed"/best \
      --output_json_path=$DATA/$accent/manifests/train/quartznet/word_error_predictor_real \
      --seed=$seed &
  done
done
```


"ABA" "BWC" "EBVS" "HJK" "HKK" "HQTV" "LXC" "NJS" "SKA" "THV"
```
for size in 50 75 100 150 200 300 400 500
do
  for seed in 2 3
  do
    for accent in "HJK"
    do
      for method in "word_enhance"
      do
        echo $accent seed $seed size $size method $method
        echo 
        echo
        CUDA_VISIBLE_DEVICES=4 python3 -u inference.py \
        --batch_size=32 \
        --output_file=$DATA/$accent/manifests/train/quartznet/word_error_predictor_real/$size/$method/seed_"$seed"/test_out_ori.txt \
        --wav_dir=$WAV_DIR \
        --val_manifest=$DATA/$accent/manifests/train/quartznet/word_error_predictor_real/$size/$method/seed_"$seed"/train.json \
        --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
        --ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
        > $DATA/$accent/manifests/train/quartznet/word_error_predictor_real/$size/$method/seed_"$seed"/test_out_ori_log.txt 
      done
    done 
  done & 
done
```




## 2.3 ICASSP

### 2.3.1 Train

"ABA" "BWC" "EBVS" "HJK" "HKK" "HQTV" "LXC" "NJS" "SKA" "THV"
```
for accent in "THV"
do
  for seed in 1 2 3
  do
    LR=3e-4
    echo $accent seed $seed
    mkdir -p $PRETRAINED_CKPTS/error_models/quartznet/$accent/seed_"$seed"/
    CUDA_VISIBLE_DEVICES=5 python3 -u train_error_model.py \
      --batch_size=1 \
      --train_path=$DATA/$accent/manifests/quartznet_outputs/seed_out.txt \
      --test_path=$DATA/$accent/manifests/quartznet_outputs/dev_out.txt \
      --lr_decay=warmup \
      --seed=$seed \
      --output_dir=$PRETRAINED_CKPTS/error_models/quartznet/$accent/seed_"$seed"/recent \
      --best_dir=$PRETRAINED_CKPTS/error_models/quartznet/$accent/seed_"$seed"/best \
    > $PRETRAINED_CKPTS/error_models/quartznet/$accent/seed_"$seed"/train_log.txt & 
  echo
  done 
done 
```

### 2.3.2 Sample

Before using the error model to select test cases, we need to first infer the model on all the texts and store the results.

"ABA" "BWC" "EBVS" "HJK" "HKK" "HQTV" "LXC" "NJS" "SKA" "THV"
```
for accent in "EBVS" "HJK" "HKK" "HQTV" "LXC" "NJS" "SKA" "THV"
do
  for seed in 1 2 3
  do
    echo $accent seed $seed
    CUDA_VISIBLE_DEVICES=4 python3 -u infer_error_model.py \
      --batch_size=64 \
      --json_path=$DATA/$accent/manifests/selection.json \
      --pretrained_ckpt=$PRETRAINED_CKPTS/error_models/quartznet/$accent/seed_"$seed"/best/ErrorClassifierPhoneBiLSTM_V2.pt \
      --output_dir=$PRETRAINED_CKPTS/error_models/quartznet/$accent/seed_"$seed"/best \
    > $PRETRAINED_CKPTS/error_models/quartznet/$accent/seed_"$seed"/infer_log.txt &
  echo
  done 
done 
```


```
for accent in "ABA" "BWC" "EBVS" "HJK" "HKK" "HQTV" "LXC" "NJS" "SKA" "THV"
do
  for seed in 1 2 3
  do
    echo $accent seed $seed
    CUDA_VISIBLE_DEVICES=4 python3 -u error_model_sampling.py \
      --selection_json_file=$DATA/$accent/manifests/selection.json \
      --seed_json_file=$DATA/$accent/manifests/seed.json \
      --error_model_weights=$PRETRAINED_CKPTS/error_models/quartznet/$accent/seed_"$seed"/best/weights.pkl \
      --random_json_path=$DATA/$accent/manifests/train/random \
      --output_json_path=$DATA/$accent/manifests/train/quartznet/error_model \
      --exp_id=$seed &
  echo
  done 
done 
```


### 2.3.3 Evaluate

"ABA" "BWC" "EBVS" "HJK" "HKK" "HQTV" "LXC" "NJS" "SKA" "THV"

```
for size in 50 75 100 150 200 300 400 500
do
  for seed in 1 2 3
  do
    for accent in "NJS"
    do
      echo $accent $seed $size
      echo 
      echo
      CUDA_VISIBLE_DEVICES=4 python3 -u inference.py \
      --batch_size=32 \
      --output_file=$DATA/$accent/manifests/train/quartznet/error_model/$size/seed_"$seed"/test_out_ori.txt \
      --wav_dir=$WAV_DIR \
      --val_manifest=$DATA/$accent/manifests/train/quartznet/error_model/$size/seed_"$seed"/train.json \
      --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
      --ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
      > $DATA/$accent/manifests/train/quartznet/error_model/$size/seed_"$seed"/test_out_ori_log.txt
    done 
  done &
done 
```

# 3. Fine-Tune ASR Models

## Train on ICASSP sampling (real)

"" ""  ""  ""  ""  ""
"ABA" "BWC" "EBVS" "HJK" "HKK" "HQTV" "LXC" "NJS" "SKA" "THV"
```
for seed in 1 2 3
do
  for size in 50 75 100 150 200 300 400 500
  do
    for accent in "THV" 
    do
      echo $accent $seed $size
      echo 
      echo
      model_dir=$PRETRAINED_CKPTS/quartznet/finetuned/$accent/$size/seed_"$seed"/icassp_real_mix
      mkdir -p $model_dir
      CUDA_VISIBLE_DEVICES=7 python3 -u finetune.py \
        --batch_size=16 \
        --num_epochs=100 \
        --eval_freq=1 \
        --train_freq=30 \
        --lr=1e-5 \
        --wav_dir=$WAV_DIR \
        --train_manifest=$DATA/$accent/manifests/train/quartznet/error_model/$size/seed_"$seed"/train.json \
        --val_manifest=$DATA/$accent/manifests/dev.json \
        --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
        --output_dir=$model_dir/recent \
        --best_dir=$model_dir/best \
        --early_stop_patience=10 \
        --zero_infinity \
        --save_after_each_epoch \
        --turn_bn_eval \
        --ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
        --lr_decay=warmup \
        --seed=$seed \
        --optimizer=novograd \
      > $model_dir/train_log.txt


      CUDA_VISIBLE_DEVICES=7 python3 -u inference.py \
      --batch_size=16 \
      --output_file=$model_dir/test_out.txt \
      --wav_dir=$WAV_DIR \
      --val_manifest=$DATA/$accent/manifests/test.json \
      --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
      --ckpt=$model_dir/best/Jasper.pt \
      > $model_dir/test_infer_log.txt
    done 
  done 
done  &
```

## Train on ICASSP no seed.


ASI RRBI TNI SVBI
```
for seed in 1 2 3
do
  for size in 50 75 100 150 200 300 400 500
  do
    for accent in "SVBI" 
    do
      echo $accent $seed $size
      echo 
      echo
      model_dir=$PRETRAINED_CKPTS/quartznet/finetuned/$accent/$size/seed_"$seed"/icassp_real_no_seed
      mkdir -p $model_dir
      CUDA_VISIBLE_DEVICES=4 python3 -u finetune.py \
        --batch_size=16 \
        --num_epochs=100 \
        --eval_freq=1 \
        --train_freq=30 \
        --lr=1e-5 \
        --wav_dir=$WAV_DIR \
        --train_manifest=$DATA/$accent/manifests/train/quartznet/error_model/$size/seed_"$seed"/train_no_seed.json \
        --val_manifest=$DATA/$accent/manifests/dev.json \
        --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
        --output_dir=$model_dir/recent \
        --best_dir=$model_dir/best \
        --early_stop_patience=10 \
        --zero_infinity \
        --save_after_each_epoch \
        --turn_bn_eval \
        --ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
        --lr_decay=warmup \
        --seed=$seed \
        --optimizer=novograd \
      > $model_dir/train_log.txt


      CUDA_VISIBLE_DEVICES=4 python3 -u inference.py \
      --batch_size=32 \
      --output_file=$model_dir/test_out.txt \
      --wav_dir=$WAV_DIR \
      --val_manifest=$DATA/$accent/manifests/test.json \
      --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
      --ckpt=$model_dir/best/Jasper.pt \
      > $model_dir/test_infer_log.txt
    done 
  done 
done  &
```


## Train on Word Error Data (real)

"" ""  ""  "" ""  ""  ""  ""  ""
ASI RRBI TNI SVBI
```
for accent in "SVBI"
do
  for seed in 1 2 3
  do
    for size in 50 75 100 150 200 300 400 500
    do
      echo $accent $seed $size
      echo 
      model_dir=$PRETRAINED_CKPTS/quartznet/finetuned/$accent/$size/seed_"$seed"/word_error_real_mix/word_enhance
      mkdir -p $model_dir
      CUDA_VISIBLE_DEVICES=7 python3 -u finetune.py \
        --batch_size=16 \
        --num_epochs=100 \
        --eval_freq=1 \
        --train_freq=30 \
        --lr=2e-5 \
        --wav_dir=$WAV_DIR \
        --train_manifest=$DATA/$accent/manifests/train/quartznet/word_error_predictor_real/$size/word_enhance/seed_"$seed"/train.json \
        --val_manifest=$DATA/$accent/manifests/dev.json \
        --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
        --output_dir=$model_dir/recent \
        --best_dir=$model_dir/best \
        --early_stop_patience=10 \
        --zero_infinity \
        --save_after_each_epoch \
        --turn_bn_eval \
        --ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
        --lr_decay=warmup \
        --seed=$seed \
        --optimizer=novograd \
      > $model_dir/train_log.txt

      echo $accent $seed $size
      echo
      CUDA_VISIBLE_DEVICES=7 python3 -u inference.py \
      --batch_size=64 \
      --output_file=$model_dir/test_out.txt \
      --wav_dir=$WAV_DIR \
      --val_manifest=$DATA/$accent/manifests/test.json \
      --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
      --ckpt=$model_dir/best/Jasper.pt \
      > $model_dir/test_infer_log.txt  
    done 
  done
done  &
```


## Train on Word Error Data (no seed)

"" ""  ""  "" ""  ""  ""  ""  ""
ASI RRBI TNI SVBI
```
for accent in "SVBI"
do
  for seed in 1 2 3
  do
    for size in 50 75 100 150 200 300 400 500
    do
      echo $accent $seed $size
      echo 
      model_dir=$PRETRAINED_CKPTS/quartznet/finetuned/$accent/$size/seed_"$seed"/word_error_real_mix/word_enhance_no_seed
      mkdir -p $model_dir
      CUDA_VISIBLE_DEVICES=3 python3 -u finetune.py \
        --batch_size=16 \
        --num_epochs=100 \
        --eval_freq=1 \
        --train_freq=30 \
        --lr=2e-5 \
        --wav_dir=$WAV_DIR \
        --train_manifest=$DATA/$accent/manifests/train/quartznet/word_error_predictor_real/$size/word_enhance/seed_"$seed"/train_no_seed.json \
        --val_manifest=$DATA/$accent/manifests/dev.json \
        --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
        --output_dir=$model_dir/recent \
        --best_dir=$model_dir/best \
        --early_stop_patience=10 \
        --zero_infinity \
        --save_after_each_epoch \
        --turn_bn_eval \
        --ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
        --lr_decay=warmup \
        --seed=$seed \
        --optimizer=novograd \
      > $model_dir/train_log.txt

      echo $accent $seed $size
      echo
      CUDA_VISIBLE_DEVICES=3 python3 -u inference.py \
      --batch_size=64 \
      --output_file=$model_dir/test_out.txt \
      --wav_dir=$WAV_DIR \
      --val_manifest=$DATA/$accent/manifests/test.json \
      --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
      --ckpt=$model_dir/best/Jasper.pt \
      > $model_dir/test_infer_log.txt  
    done 
  done
done  &
```
