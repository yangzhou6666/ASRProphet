# Scripts to run experiments on L2Arctic

The following code takes `ASI` in the L2Arctic dataset as an example. Please kindly change 

# 1. Infer ASR on dataset

## 1.1 Infer on `seed_plus_dev.json`
Generate transcripts for the seed+dev set using the pre-trainded ASR (Transcripts are used while training error models)


"YBAA" "ZHAA" "ASI" "TNI" "NCC" "TXHC" "EBVS" "ERMS" "YDCK" "YKWK" "THV" "TLV"

```
DATA=$(cd ../../data/l2arctic/processed; pwd)
WAV_DIR=$(cd ../../data/l2arctic/; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd) 
declare -a accents=('ASI' 'RRBI')
declare -a models=('wavlm')

"YBAA" "ZHAA" "ASI" "TNI" "NCC" "TXHC"
"EBVS" "ERMS" "YDCK" "YKWK" "THV" "TLV"

for model in "wavlm"
do
  for accent in "EBVS" "ERMS" "YDCK" "YKWK" "THV" "TLV"
  do
    echo $model $accent
    mkdir -p $DATA/$accent/manifests/"$model"_outputs
    echo $WAV_DIR/$accent/wav 
    CUDA_VISIBLE_DEVICES=5 python3 -u inference.py \
      --wav_dir=$WAV_DIR \
      --val_manifest=$DATA/$accent/manifests/seed_plus_dev.json \
      --model $model \
      --output_file=$DATA/$accent/manifests/"$model"_outputs/seed_plus_dev_out.txt \
      > $DATA/$accent/manifests/"$model"_outputs/seed_plus_dev_infer_log.txt
  done
done &
```

## 1.2 Infer on `seed.json` and `dev.json`

The results will be used for training word error predictor.

"YBAA" "ZHAA" "ASI" "TNI" "NCC" "TXHC"
"EBVS" "ERMS" "YDCK" "YKWK" "THV" "TLV"
```
for model in "wavlm"
do
  for accent in "EBVS" "ERMS" "YDCK" "YKWK" "THV" "TLV"
  do
    echo $model $accent
    mkdir -p $DATA/$accent/manifests/"$model"_outputs
    echo $WAV_DIR/$accent/wav 
    cuda_devices=1

    CUDA_VISIBLE_DEVICES=$cuda_devices python3 -u inference.py \
      --wav_dir=$WAV_DIR \
      --val_manifest=$DATA/$accent/manifests/seed.json \
      --model $model \
      --output_file=$DATA/$accent/manifests/"$model"_outputs/seed_out.txt \
      > $DATA/$accent/manifests/"$model"_outputs/seed_infer_log.txt
    
    CUDA_VISIBLE_DEVICES=$cuda_devices python3 -u inference.py \
      --wav_dir=$WAV_DIR \
      --val_manifest=$DATA/$accent/manifests/dev.json \
      --model $model \
      --output_file=$DATA/$accent/manifests/"$model"_outputs/dev_out.txt \
      > $DATA/$accent/manifests/"$model"_outputs/dev_infer_log.txt
  done
done &
```



## Infer on test dataset

Let's try on the test set.

"YBAA" "ZHAA" "ASI" "TNI" "NCC" "TXHC" "EBVS" "ERMS" "YDCK" "YKWK" "THV" "TLV"
"YBAA" "ZHAA" "ASI" "TNI" "NCC" "TXHC"
"EBVS" "ERMS" "YDCK" "YKWK" "THV" "TLV"

```
for model in "wavlm"
do
  for accent in "EBVS" "ERMS" "YDCK" "YKWK" "THV" "TLV"
  do
    echo $model $accent
    mkdir -p $DATA/$accent/manifests/"$model"_outputs
    echo $WAV_DIR/$accent/wav 
    CUDA_VISIBLE_DEVICES=6 python3 -u inference.py \
      --wav_dir=$WAV_DIR \
      --val_manifest=$DATA/$accent/manifests/test.json \
      --model $model \
      --output_file=$DATA/$accent/manifests/"$model"_outputs/original_test_out.txt \
      > $DATA/$accent/manifests/"$model"_outputs/original_test_infer_log.txt
  done
done &
```

# 2. Train Error Model

## 2.1 Word error predictor

### 2.1.1 Training

"YBAA" "ZHAA" "ASI" "TNI"
"NCC" "TXHC" "EBVS" "ERMS"
"YDCK" "YKWK" "THV" "TLV"
```
for model in "wavlm"
do
  for seed in 1 2 3
  do
    for accent in "YDCK" "YKWK" "THV" "TLV"
    do
    mkdir -p $PRETRAINED_CKPTS/word_error_predictor/"$model"/$accent/seed_"$seed"/best
    echo $accent seed $seed
    echo 
    CUDA_VISIBLE_DEVICES=5 python word_error_predictor.py \
      --train_path=$DATA/$accent/manifests/"$model"_outputs/seed_out.txt \
      --test_path=$DATA/$accent/manifests/"$model"_outputs/dev_out.txt \
      --output_dir=$PRETRAINED_CKPTS/word_error_predictor/"$model"/$accent/seed_"$seed"/best \
      --seed=$seed \
      > $PRETRAINED_CKPTS/word_error_predictor/"$model"/$accent/seed_"$seed"/training.log
    done
  done &
done &
```


### 2.1.2 Sampling `real` audio and evaluation



"YBAA" "ZHAA" "ASI" "TNI"
"NCC" "TXHC" "EBVS" "ERMS"
"YDCK" "YKWK" "THV" "TLV"

"YBAA" "ZHAA" "ASI"
"TNI" "NCC" "TXHC"
"EBVS" "ERMS" "YDCK"
"YKWK" "THV" "TLV"

```
for model in "wavlm"
do
  for seed in 1 2 3
  do
    for accent in "YKWK" "THV" "TLV"
    do
      echo $accent seed $seed
      CUDA_VISIBLE_DEVICES=5 python3 -u word_error_sampling.py \
        --seed_json_file=$DATA/$accent/manifests/seed.json \
        --data_folder=$DATA/$accent/manifests/train/random/ \
        --selection_json_file=$DATA/$accent/manifests/selection.json \
        --finetuned_ckpt=$PRETRAINED_CKPTS/word_error_predictor/"$model"/$accent/seed_"$seed"/best \
        --output_json_path=$DATA/$accent/manifests/train/"$model"/word_error_predictor_real \
        --seed=$seed 
    done 
  done
done &
```


"YBAA" "ZHAA" "ASI" "TNI" "NCC" "TXHC" "EBVS" "ERMS" "YDCK" "YKWK" "THV" "TLV"

"YBAA" "ZHAA"
"ASI" "TNI"
"NCC" "TXHC"
"EBVS" "ERMS"
"YDCK" "YKWK"
"THV" "TLV"


```
for model in "wavlm"
do
  for seed in 1 2 3
  do
    for accent in "THV" "TLV"
    do
      for size in 100 200 300 400
      do
        for method in "word_enhance"
        do
          echo $accent seed $seed size $size method $method
          echo 
          echo
          CUDA_VISIBLE_DEVICES=6 python3 -u inference.py \
          --wav_dir=$WAV_DIR \
          --val_manifest=$DATA/$accent/manifests/train/"$model"/word_error_predictor_real/$size/$method/seed_"$seed"/train.json \
          --model $model \
          --batch_size 8 \
          --output_file=$DATA/$accent/manifests/train/"$model"/word_error_predictor_real/$size/$method/seed_"$seed"/test_out_ori.txt \
          > $DATA/$accent/manifests/train/"$model"/word_error_predictor_real/$size/$method/seed_"$seed"/test_out_ori_log.txt
        done 
      done
    done
  done
done &
```




## 2.3 ICASSP

### 2.3.1 Train


"YBAA" "ZHAA" "ASI" "TNI"
"NCC" "TXHC" "EBVS" "ERMS"
"YDCK" "YKWK" "THV" "TLV"
```
for model in "wavlm"
do
  for seed in 1 2 3
  do
    for accent in "YDCK" "YKWK" "THV" "TLV"
    do
      LR=3e-4
      echo $accent seed $seed
      mkdir -p $PRETRAINED_CKPTS/error_models/"$model"/$accent/seed_"$seed"/
      CUDA_VISIBLE_DEVICES=6 python3 -u train_error_model.py \
        --batch_size=64 \
        --train_path=$DATA/$accent/manifests/"$model"_outputs/seed_out.txt \
        --test_path=$DATA/$accent/manifests/"$model"_outputs/dev_out.txt \
        --lr_decay=warmup \
        --seed=$seed \
        --output_dir=$PRETRAINED_CKPTS/error_models/"$model"/$accent/seed_"$seed"/recent \
        --best_dir=$PRETRAINED_CKPTS/error_models/"$model"/$accent/seed_"$seed"/best \
      > $PRETRAINED_CKPTS/error_models/"$model"/$accent/seed_"$seed"/train_log.txt
    done 
  done &
done &
```


### 2.3.2 Sample

Before using the error model to select test cases, we need to first infer the model on all the texts and store the results.


"YBAA" "ZHAA" "ASI" "TNI"
"NCC" "TXHC" "EBVS" "ERMS"
"YDCK" "YKWK" "THV" "TLV"
```
for model in "wavlm"
do
  for seed in 1 2 3
  do
    for accent in "YDCK" "YKWK" "THV" "TLV"
    do
      cuda_devices=6
      echo $accent seed $seed cuda $cuda_devices
      CUDA_VISIBLE_DEVICES=$cuda_devices python3 -u infer_error_model.py \
        --batch_size=64 \
        --json_path=$DATA/$accent/manifests/selection.json \
        --pretrained_ckpt=$PRETRAINED_CKPTS/error_models/"$model"/$accent/seed_"$seed"/best/ErrorClassifierPhoneBiLSTM_V2.pt \
        --output_dir=$PRETRAINED_CKPTS/error_models/"$model"/$accent/seed_"$seed"/best \
      > $PRETRAINED_CKPTS/error_models/"$model"/$accent/seed_"$seed"/infer_log.txt

      CUDA_VISIBLE_DEVICES=$cuda_devices python3 -u error_model_sampling.py \
        --selection_json_file=$DATA/$accent/manifests/selection.json \
        --seed_json_file=$DATA/$accent/manifests/seed.json \
        --error_model_weights=$PRETRAINED_CKPTS/error_models/"$model"/$accent/seed_"$seed"/best/weights.pkl \
        --random_json_path=$DATA/$accent/manifests/train/random \
        --output_json_path=$DATA/$accent/manifests/train/"$model"/error_model \
        --exp_id=$seed 
    echo
    done
  done &
done &
```


### 2.3.3 Evaluate

"YBAA" "ZHAA" "ASI" "TNI"
"NCC" "TXHC" "EBVS" "ERMS"
"YDCK" "YKWK" "THV" "TLV"
```
for model in "wavlm"
do
  for seed in 1 2 3
  do
    for accent in "YDCK" "YKWK" "THV" "TLV"
    do
      for size in 100 200 300 400
      do
        echo $accent $seed $size
        echo 
        echo
        CUDA_VISIBLE_DEVICES=7 python3 -u inference.py \
          --wav_dir=$WAV_DIR \
          --val_manifest=$DATA/$accent/manifests/train/"$model"/error_model/$size/seed_"$seed"/train.json \
          --model $model \
          --batch_size 8 \
          --output_file=$DATA/$accent/manifests/train/"$model"/error_model/$size/seed_"$seed"/test_out_ori.txt \
          > $DATA/$accent/manifests/train/"$model"/error_model/$size/seed_"$seed"/test_out_ori_log.txt
      done 
    done 
  done 
done &
```

# 3. Fine-Tune ASR Models

## Train on ICASSP sampling (real)

"YBAA" "ZHAA" "ASI" "TNI" 
"NCC" "TXHC" "EBVS" "ERMS"
"YDCK" "YKWK" "THV" "TLV"
```
for model in "wavlm"
do
  for accent in "YDCK" "YKWK" "THV" "TLV"
  do
    for seed in 1 2 3
    do
      for size in 100 200 300 400
      do
        echo $accent seed $seed $model
        model_dir=$PRETRAINED_CKPTS/"$model"/finetuned/$accent/$size/seed_"$seed"/icassp_real_mix
        mkdir -p $model_dir
        cuda_devices=3
        CUDA_VISIBLE_DEVICES=$cuda_devices python3 -u finetune.py \
          --wav_dir=$WAV_DIR \
          --train_manifest=$DATA/$accent/manifests/train/"$model"/error_model/$size/seed_"$seed"/train.json \
          --val_manifest=$DATA/$accent/manifests/dev.json \
          --output_dir=$model_dir/best \
          --model=$model \
          --seed=$seed \
          --lr=1e-5 \
          --batch_size=6 \
          > $model_dir/train_log.txt
        
        rm -rf $model_dir/best/tmp_checkpoints/

        CUDA_VISIBLE_DEVICES=$cuda_devices python3 -u inference.py \
          --wav_dir=$WAV_DIR \
          --val_manifest=$DATA/$accent/manifests/test.json \
          --output_file=$model_dir/test_out.txt \
          --model=$model \
          --seed=$seed \
          --checkpoint=$model_dir/best \
          > $model_dir/test_infer_log.txt
      done
    done
  done
done &
```


## Train on Word Error Data (real)

"YBAA" "ZHAA" "ASI" "TNI" 
"NCC" "TXHC" "EBVS" "ERMS"
"YDCK" "YKWK" "THV" "TLV"

```
for model in "wavlm"
do
  for accent in "YDCK" "YKWK" "THV" "TLV"
  do
    for seed in 1 2 3
    do
      for size in 100 200 300 400
      do
        echo $accent seed $seed $model
        model_dir=$PRETRAINED_CKPTS/"$model"/finetuned/$accent/$size/seed_"$seed"/word_error_real_mix/word_enhance/
        mkdir -p $model_dir
        cuda_devices=6
        CUDA_VISIBLE_DEVICES=$cuda_devices python3 -u finetune.py \
          --wav_dir=$WAV_DIR \
          --train_manifest=$DATA/$accent/manifests/train/"$model"/word_error_predictor_real/$size/word_enhance/seed_"$seed"/train.json \
          --val_manifest=$DATA/$accent/manifests/dev.json \
          --output_dir=$model_dir/best \
          --model=$model \
          --seed=$seed \
          --lr=1e-5 \
          --batch_size=6 \
          > $model_dir/train_log.txt
        
        rm -rf $model_dir/best/tmp_checkpoints/

        CUDA_VISIBLE_DEVICES=$cuda_devices python3 -u inference.py \
          --wav_dir=$WAV_DIR \
          --val_manifest=$DATA/$accent/manifests/test.json \
          --output_file=$model_dir/test_out.txt \
          --model=$model \
          --seed=$seed \
          --checkpoint=$model_dir/best \
          > $model_dir/test_infer_log.txt
      done 
    done
  done
done &
```