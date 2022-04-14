# Scripts to run experiments on L2Arctic

Prepare some environment variables
```
DATA=$(cd ../../data/l2arctic/processed; pwd)
WAV_DIR=$(cd ../../data/l2arctic/; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd) 
mkdir -p "$(pwd)"/../../huggingface_cache/
CACHE_DIR=$(cd ../../huggingface_cache/; pwd)
```

# 1. Infer ASR on dataset


## Infer on `seed.json` and `dev.json`

The results will be used for training word error predictor.

"YBAA" "ZHAA" "ASI" "TNI" "NCC" "TXHC" "EBVS" "ERMS" "YDCK" "YKWK" "THV" "TLV"
```
for model in "hubert"
do
  for accent in "SKA" "ZHAA" "HJK" "HKK" "ASI" "RRBI" "SVBI" "TNI" "THV"
  do
    echo $model $accent
    mkdir -p $DATA/$accent/manifests/"$model"_outputs
    echo $WAV_DIR/$accent/wav 

    CUDA_VISIBLE_DEVICES=6 python3 -u inference.py \
      --cache_dir=$CACHE_DIR \
      --wav_dir=$WAV_DIR \
      --val_manifest=$DATA/$accent/manifests/seed.json \
      --model $model \
      --output_file=$DATA/$accent/manifests/"$model"_outputs/seed_out.txt \
      > $DATA/$accent/manifests/"$model"_outputs/seed_infer_log.txt
    
    CUDA_VISIBLE_DEVICES=6 python3 -u inference.py \
      --cache_dir=$CACHE_DIR \
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
```
for model in "hubert"
do
  for accent in "ASI"
  do
    echo $model $accent
    mkdir -p $DATA/$accent/manifests/"$model"_outputs
    echo $WAV_DIR/$accent/wav 
    CUDA_VISIBLE_DEVICES=7 python3 -u inference.py \
      --cache_dir=$CACHE_DIR \
      --wav_dir=$WAV_DIR \
      --val_manifest=$DATA/$accent/manifests/test.json \
      --model $model \
      --output_file=$DATA/$accent/manifests/"$model"_outputs/original_test_out.txt \
      > $DATA/$accent/manifests/"$model"_outputs/original_test_infer_log.txt
  done
done &
```

# 2. Train Error Model

## 2.1 ICASSP

### 2.1.1 Train

"YBAA" "ZHAA" "ASI" "TNI" "NCC" "TXHC" "EBVS" "ERMS" "YDCK" "YKWK" "THV" "TLV"
```
for model in "hubert"
do
  for accent in "EBVS" "ERMS" "YDCK" "YKWK" "THV" "TLV"
  do
    for seed in 1 2 3
    do
      LR=3e-4
      echo $accent seed $seed
      mkdir -p $PRETRAINED_CKPTS/error_models/"$model"/$accent/seed_"$seed"/
      CUDA_VISIBLE_DEVICES=3 python3 -u train_error_model.py \
        --batch_size=64 \
        --train_path=$DATA/$accent/manifests/"$model"_outputs/seed_out.txt \
        --test_path=$DATA/$accent/manifests/"$model"_outputs/dev_out.txt \
        --lr_decay=warmup \
        --seed=$seed \
        --output_dir=$PRETRAINED_CKPTS/error_models/"$model"/$accent/seed_"$seed"/recent \
        --best_dir=$PRETRAINED_CKPTS/error_models/"$model"/$accent/seed_"$seed"/best \
      > $PRETRAINED_CKPTS/error_models/"$model"/$accent/seed_"$seed"/train_log.txt
    done & 
  done 
done &
```

### 2.1.2 Infer error model
Before using the error model to select test cases, we need to first infer the model on all the texts and store the results.


"YBAA" "ZHAA" "ASI" "TNI" "NCC" "TXHC" "EBVS" "ERMS" "YDCK" "YKWK" "THV" "TLV"
```
for model in "hubert"
do
  for seed in 1 2 3
  do
    for accent in "YBAA" "ZHAA" "ASI" "TNI"
    do
      cuda_devices=0
      echo $accent seed $seed cuda $cuda_devices
      CUDA_VISIBLE_DEVICES=$cuda_devices python3 -u infer_error_model.py \
        --batch_size=64 \
        --json_path=$DATA/$accent/manifests/selection.json \
        --pretrained_ckpt=$PRETRAINED_CKPTS/error_models/"$model"/$accent/seed_"$seed"/best/ErrorClassifierPhoneBiLSTM_V2.pt \
        --output_dir=$PRETRAINED_CKPTS/error_models/"$model"/$accent/seed_"$seed"/best \
      > $PRETRAINED_CKPTS/error_models/"$model"/$accent/seed_"$seed"/infer_log.txt
    done &
  done 
done &
```


### 2.1.2 Sample 

**diversity_enhancing**

"YBAA" "ZHAA" "ASI" "TNI" "NCC" "TXHC" "EBVS" "ERMS" "YDCK" "YKWK" "THV" "TLV"
```
for model in "hubert"
do
  for seed in 1 2 3
  do
    for accent in "RRBI" "SVBI" "TNI" "THV"
    do
      sampling_method=diversity_enhancing
      cuda_devices=0
      echo $accent seed $seed cuda $cuda_devices
      CUDA_VISIBLE_DEVICES=$cuda_devices python3 -u error_model_sampling.py \
        --sampling_method=$sampling_method \
        --selection_json_file=$DATA/$accent/manifests/selection.json \
        --seed_json_file=$DATA/$accent/manifests/seed.json \
        --error_model_weights=$PRETRAINED_CKPTS/error_models/"$model"/$accent/seed_"$seed"/best/weights.pkl \
        --random_json_path=$DATA/$accent/manifests/train/random \
        --output_json_path=$DATA/$accent/manifests/train/"$model"/error_model \
        --exp_id=$seed 
    done &
  done 
done &
```

**without_diversity_enhancing**

"YBAA" "ZHAA" "ASI" "TNI" "NCC" "TXHC" "EBVS" "ERMS" "YDCK" "YKWK" "THV" "TLV"
```
for model in "hubert"
do
  for seed in 1 2 3
  do
    for accent in "YBAA" "ZHAA" "ASI" "TNI" "NCC" "TXHC" "EBVS" "ERMS" "YDCK" "YKWK" "THV" "TLV"
    do
      cuda_devices=3
      sampling_method=without_diversity_enhancing
      echo $accent seed $seed cuda $cuda_devices
      CUDA_VISIBLE_DEVICES=$cuda_devices python3 -u error_model_sampling.py \
        --sampling_method=$sampling_method \
        --selection_json_file=$DATA/$accent/manifests/selection.json \
        --seed_json_file=$DATA/$accent/manifests/seed.json \
        --error_model_weights=$PRETRAINED_CKPTS/error_models/"$model"/$accent/seed_"$seed"/best/weights.pkl \
        --random_json_path=$DATA/$accent/manifests/train/random \
        --output_json_path=$DATA/$accent/manifests/train/"$model"/error_model_"$sampling_method" \
        --exp_id=$seed 
    done &
  done 
done &
```

**pure_diversity**

"YBAA" "ZHAA" "ASI" "TNI" "NCC" "TXHC"
"EBVS" "ERMS" "YDCK" "YKWK" "THV" "TLV"
```
for model in "hubert"
do
  for seed in 1 2 3
  do
    for accent in "EBVS" "ERMS" "YDCK" "YKWK" "THV" "TLV"
    do
      cuda_devices=5
      sampling_method=pure_diversity
      echo $accent seed $seed cuda $cuda_devices
      CUDA_VISIBLE_DEVICES=$cuda_devices python3 -u error_model_sampling.py \
        --sampling_method=$sampling_method \
        --selection_json_file=$DATA/$accent/manifests/selection.json \
        --seed_json_file=$DATA/$accent/manifests/seed.json \
        --error_model_weights=$PRETRAINED_CKPTS/error_models/"$model"/$accent/seed_"$seed"/best/weights.pkl \
        --random_json_path=$DATA/$accent/manifests/train/random \
        --output_json_path=$DATA/$accent/manifests/train/"$model"/error_model_"$sampling_method" \
        --exp_id=$seed 
    done &
  done 
done &
```


### 2.1.3 Evaluate

**diversity_enhancing**
"YBAA" "ZHAA" "ASI" "TNI" "NCC" "TXHC" "EBVS" "ERMS" "YDCK" "YKWK" "THV" "TLV"
```
for model in "hubert"
do
  for seed in 1 2 3
  do
    for size in 100 200 300 400
    do
      for accent in "TXHC"
      do
        echo $accent $seed $size
        echo 
        echo
        CUDA_VISIBLE_DEVICES=7 python3 -u inference.py \
          --cache_dir=$CACHE_DIR \
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

**without_diversity_enhancing**
"YBAA" "ZHAA" "ASI" "TNI" "NCC" "TXHC" "EBVS" "ERMS" "YDCK" "YKWK" "THV" "TLV"
```
for model in "hubert" "wav2vec-base"
do
  for seed in 1 2 3
  do
    for size in 100 200 300 400
    do
      for accent in "EBVS" "ERMS" "YDCK" "YKWK" "THV" "TLV"
      do
        sampling_method=without_diversity_enhancing
        echo $accent $seed $size $sampling_method
        echo 
        echo
        CUDA_VISIBLE_DEVICES=7 python3 -u inference.py \
          --cache_dir=$CACHE_DIR \
          --wav_dir=$WAV_DIR \
          --val_manifest=$DATA/$accent/manifests/train/"$model"/error_model_"$sampling_method"/$size/seed_"$seed"/train.json \
          --model $model \
          --batch_size 8 \
          --output_file=$DATA/$accent/manifests/train/"$model"/error_model_"$sampling_method"/$size/seed_"$seed"/test_out_ori.txt \
          > $DATA/$accent/manifests/train/"$model"/error_model_"$sampling_method"/$size/seed_"$seed"/test_out_ori_log.txt
      done 
    done 
  done &
done &
```


**pure_diversity**
"YBAA" "ZHAA" "ASI" "TNI" "NCC" "TXHC" "EBVS" "ERMS" "YDCK" "YKWK" "THV" "TLV"
```
for model in "hubert"
do
  for seed in 2
  do
    for size in 200 300
    do
      for accent in "NCC"
      do
        sampling_method=pure_diversity
        echo $accent $seed $size $sampling_method
        echo 
        echo
        CUDA_VISIBLE_DEVICES=0 python3 -u inference.py \
          --cache_dir=$CACHE_DIR \
          --wav_dir=$WAV_DIR \
          --val_manifest=$DATA/$accent/manifests/train/"$model"/error_model_"$sampling_method"/$size/seed_"$seed"/train.json \
          --model $model \
          --batch_size 8 \
          --output_file=$DATA/$accent/manifests/train/"$model"/error_model_"$sampling_method"/$size/seed_"$seed"/test_out_ori.txt \
          > $DATA/$accent/manifests/train/"$model"/error_model_"$sampling_method"/$size/seed_"$seed"/test_out_ori_log.txt
      done 
    done 
  done &
done &
```


## 2.2 ASREvolve

### 2.2.1 Training

"YBAA" "ZHAA"
"ASI" "TNI"
"NCC" "TXHC"
"EBVS" "ERMS"
"YDCK" "YKWK" 
"THV" "TLV"
```
for model in "hubert"
do  
  for seed in 1 2 3
  do
    for accent in "YBAA" "ZHAA"
    do
      cuda_devices=1
      echo $accent seed $seed
      mkdir -p $PRETRAINED_CKPTS/asrevolve_error_models/"$model"/$accent/seed_"$seed"
      CUDA_VISIBLE_DEVICES=$cuda_devices python3 -u train_error_model_asrevolve.py \
        --train_path=$DATA/$accent/manifests/"$model"_outputs/seed_out.txt \
        --test_path=$DATA/$accent/manifests/"$model"_outputs/dev_out.txt \
        --seed=$seed \
        --output_dir=$PRETRAINED_CKPTS/asrevolve_error_models/"$model"/$accent/seed_"$seed"/best \
        --log_dir=$PRETRAINED_CKPTS/asrevolve_error_models/"$model"/$accent/seed_"$seed"/train_log \
        > $PRETRAINED_CKPTS/asrevolve_error_models/"$model"/$accent/seed_"$seed"/train_log.txt 
    done & 
  done 
done &
```


### 2.2.2 Using ASREvolve for sampling

"YBAA" "ZHAA" "ASI"
"TNI" "NCC" "TXHC"
"EBVS" "ERMS" "YDCK" "YKWK" "THV" "TLV"

"ZHAA"
"TNI"
"TXHC"
"ERMS"
"YKWK" 
"TLV"

```
for model in "hubert" "wav2vec-base"
do
  for seed in 1 2 3
  do
    for num_sample in 50 75 500
    do
      for accent in "YBAA" "ZHAA" "ASI"
      do
        echo $accent seed $seed
        CUDA_VISIBLE_DEVICES=6 python3 -u error_model_sampling_asrevolve.py \
          --seed_json_file=$DATA/$accent/manifests/seed.json \
          --random_json_file=$DATA/$accent/manifests/train/random/"$num_sample"/seed_"$seed"/train.json \
          --selection_json_file=$DATA/$accent/manifests/selection.json \
          --finetuned_ckpt=$PRETRAINED_CKPTS/asrevolve_error_models/"$model"/$accent/seed_"$seed"/best \
          --log_dir=$PRETRAINED_CKPTS/asrevolve_error_models/"$model"/$accent/seed_"$seed"/train_log \
          --num_sample=$num_sample \
          --output_json_path=$DATA/$accent/manifests/train/"$model"/asrevolve_error_model_real \
          --exp_id=$seed
      done
    done &
  done  
done &
```

### 2.2.3 Measure original model performance on the selected audio

"YBAA" "ZHAA" "ASI" "TNI" "NCC" "TXHC" "EBVS" "ERMS" "YDCK" "YKWK" "THV" "TLV"

```
for model in "hubert"
do
  for seed in 1 2 3
  do
    for size in 100 200 300 400
    do
      for accent in "YBAA" "ZHAA" "ASI" "TNI" "NCC" "TXHC" "EBVS" "ERMS" "YDCK" "YKWK" "THV" "TLV"
      do
        echo $accent seed $seed size $size method $method
        echo 
        echo
        CUDA_VISIBLE_DEVICES=0 python3 -u inference.py \
        --cache_dir=$CACHE_DIR \
        --wav_dir=$WAV_DIR \
        --val_manifest=$DATA/$accent/manifests/train/"$model"/asrevolve_error_model_real/$size/seed_"$seed"/train.json \
        --model $model \
        --batch_size 8 \
        --output_file=$DATA/$accent/manifests/train/"$model"/asrevolve_error_model_real/$size/seed_"$seed"/test_out_ori.txt \
        > $DATA/$accent/manifests/train/"$model"/asrevolve_error_model_real/$size/$method/seed_"$seed"/test_out_ori_log.txt
      done 
    done
  done
done &
```

## 2.3 Word error predictor

### 2.3.1 Training

"YBAA" "ZHAA"
"ASI" "TNI" 
"NCC" "TXHC"
"EBVS" "ERMS"
"YDCK" "YKWK"
"THV" "TLV"
```
for model in "hubert"
do
  for seed in 1 2 3
  do
    for accent in "THV" "TLV"
    do
    mkdir -p $PRETRAINED_CKPTS/word_error_predictor/"$model"/$accent/seed_"$seed"/best
    echo $accent seed $seed
    echo 
    CUDA_VISIBLE_DEVICES=6 python word_error_predictor.py \
      --train_path=$DATA/$accent/manifests/"$model"_outputs/seed_out.txt \
      --test_path=$DATA/$accent/manifests/"$model"_outputs/dev_out.txt \
      --output_dir=$PRETRAINED_CKPTS/word_error_predictor/"$model"/$accent/seed_"$seed"/best \
      --seed=$seed \
      > $PRETRAINED_CKPTS/word_error_predictor/"$model"/$accent/seed_"$seed"/training.log
    done &
  done 
done &
```

### 2.3.2 Sampling `real` audio and evaluation

"YBAA" "ZHAA" "ASI" "TNI" "NCC" "TXHC"
"EBVS" "ERMS" "YDCK" "YKWK" "THV" "TLV"
**word_enhance**
```
for model in "hubert"
do
  for seed in 1 2 3
  do
    for accent in "EBVS" "ERMS" "YDCK" "YKWK" "THV" "TLV"
    do
      echo $accent seed $seed
      CUDA_VISIBLE_DEVICES=6 python3 -u word_error_sampling.py \
        --sampling_method=word_enhance \
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


"YBAA" "ZHAA"
"ASI" "TNI" 
"NCC" "TXHC"
"EBVS" "ERMS"
"YDCK" "YKWK"
"THV" "TLV"
**no_word_enhance**
```
for model in "hubert"
do
  for seed in 1 2 3
  do
    for accent in "THV" "TLV"
    do
      echo $accent seed $seed
      CUDA_VISIBLE_DEVICES=6 python3 -u word_error_sampling.py \
        --sampling_method=no_word_enhance \
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

### 2.3.3 Evaluate

"YBAA" "ZHAA" "ASI" "TNI" "NCC" "TXHC"
"EBVS" "ERMS" "YDCK" "YKWK" "THV" "TLV"

for method in "word_enhance" "no_word_enhance"
```
for model in "hubert"
do
  for seed in 1 2 3
  do
    for size in 100 200 300 400
    do
      for accent in "EBVS" "ERMS" "YDCK" "YKWK" "THV" "TLV"
      do
        for method in "no_word_enhance"
        do
          echo $accent seed $seed size $size method $method
          echo 
          echo
          CUDA_VISIBLE_DEVICES=7 python3 -u inference.py \
          --cache_dir=$CACHE_DIR \
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


# 3. Fine-Tune ASR Models

## Train using random sampling

"YBAA" "ZHAA" "ASI"
"TNI" "NCC" "TXHC"
"EBVS" "ERMS" "YDCK"
"YKWK" "THV" "TLV"
```
for model in "hubert"
do
  for accent in "YBAA" "ZHAA" "ASI"
  do
    for seed in 1 2 3
    do
      for size in 100 200
      do
        cuda_devices=4
        echo $accent seed $seed $model
        model_dir=$PRETRAINED_CKPTS/"$model"/finetuned/random/$accent/$size/seed_"$seed"/
        mkdir -p $model_dir
        CUDA_VISIBLE_DEVICES=$cuda_devices python3 -u finetune.py \
          --cache_dir=$CACHE_DIR \
          --wav_dir=$WAV_DIR \
          --train_manifest=$DATA/$accent/manifests/train/random/$size/seed_"$seed"/train.json \
          --val_manifest=$DATA/$accent/manifests/dev.json \
          --output_dir=$model_dir/best \
          --model=$model \
          --seed=$seed \
          --lr=2e-5 \
          --batch_size=6 \
          > $model_dir/train_log.txt

        rm -rf $model_dir/best/tmp_checkpoints/

        CUDA_VISIBLE_DEVICES=$cuda_devices python3 -u inference.py \
          --cache_dir=$CACHE_DIR \
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


## Train on ICASSP sampling (real)

**diversity_enhancing**

"YBAA" "ZHAA" "ASI"
"TNI" "NCC" "TXHC"
"EBVS" "ERMS" "YDCK"
"YKWK" "THV" "TLV"
```
for model in "hubert"
do
  for accent in "YBAA" "ZHAA" "ASI"
  do
    for seed in 1 2 3
    do
      for size in 100 200 300 400
      do
        echo $accent seed $seed $model
        model_dir=$PRETRAINED_CKPTS/"$model"/finetuned/icassp_real_mix/$accent/$size/seed_"$seed"/
        mkdir -p $model_dir
        cuda_devices=0
        CUDA_VISIBLE_DEVICES=$cuda_devices python3 -u finetune.py \
          --cache_dir=$CACHE_DIR \
          --wav_dir=$WAV_DIR \
          --train_manifest=$DATA/$accent/manifests/train/"$model"/error_model/$size/seed_"$seed"/train.json \
          --val_manifest=$DATA/$accent/manifests/dev.json \
          --output_dir=$model_dir/best \
          --model=$model \
          --seed=$seed \
          --lr=2e-5 \
          --batch_size=6 \
          > $model_dir/train_log.txt

        rm -rf $model_dir/best/tmp_checkpoints/

        CUDA_VISIBLE_DEVICES=$cuda_devices python3 -u inference.py \
          --cache_dir=$CACHE_DIR \
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

**without_diversity_enhancing**
"YBAA" "ZHAA"
"ASI" "TNI"
"NCC" "TXHC"
"EBVS" "ERMS"
"YDCK" "YKWK"
"THV" "TLV"
```
for model in "hubert"
do
  for accent in "THV" "TLV"
  do
    for seed in 1 2 3
    do
      for size in 100 200 300 400
      do
        cuda_devices=6
        sampling_method=without_diversity_enhancing
        echo $accent seed $seed $model icassp $sampling_method 
        echo cuda $cuda_devices
        model_dir=$PRETRAINED_CKPTS/"$model"/finetuned/icassp_"$sampling_method"_real_mix/$accent/$size/seed_"$seed"/
        mkdir -p $model_dir
        CUDA_VISIBLE_DEVICES=$cuda_devices python3 -u finetune.py \
          --cache_dir=$CACHE_DIR \
          --wav_dir=$WAV_DIR \
          --train_manifest=$DATA/$accent/manifests/train/"$model"/error_model_"$sampling_method"/$size/seed_"$seed"/train.json \
          --val_manifest=$DATA/$accent/manifests/dev.json \
          --output_dir=$model_dir/best \
          --model=$model \
          --seed=$seed \
          --lr=2e-5 \
          --batch_size=6 \
          > $model_dir/train_log.txt

        rm -rf $model_dir/best/tmp_checkpoints/

        CUDA_VISIBLE_DEVICES=$cuda_devices python3 -u inference.py \
          --cache_dir=$CACHE_DIR \
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

**pure_diversity**
"YBAA" "ZHAA"
"ASI" "TNI"
"NCC" "TXHC"
"EBVS" "ERMS"
"YDCK" "YKWK"
"THV" "TLV"
```
for model in "hubert"
do
  for seed in 2
  do
    for size in 300
    do
      for accent in "NCC"
      do
        cuda_devices=7
        sampling_method=pure_diversity
        echo $accent seed $seed $model icassp $sampling_method 
        echo cuda $cuda_devices
        model_dir=$PRETRAINED_CKPTS/"$model"/finetuned/"$sampling_method"/$accent/$size/seed_"$seed"/
        mkdir -p $model_dir
        CUDA_VISIBLE_DEVICES=$cuda_devices python3 -u finetune.py \
          --cache_dir=$CACHE_DIR \
          --wav_dir=$WAV_DIR \
          --train_manifest=$DATA/$accent/manifests/train/"$model"/error_model_"$sampling_method"/$size/seed_"$seed"/train.json \
          --val_manifest=$DATA/$accent/manifests/dev.json \
          --output_dir=$model_dir/best \
          --model=$model \
          --seed=$seed \
          --lr=2e-5 \
          --batch_size=6 \
          > $model_dir/train_log.txt

        rm -rf $model_dir/best/tmp_checkpoints/

        CUDA_VISIBLE_DEVICES=$cuda_devices python3 -u inference.py \
          --cache_dir=$CACHE_DIR \
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


#### Train ASREvolve Failure Estimator

"YBAA" "ZHAA"
"ASI" "TNI"
"NCC" "TXHC"
"EBVS" "ERMS"
"YDCK" "YKWK"
"THV" "TLV"

```
for model in "hubert"
do
  for accent in "YDCK"
  do
    for seed in 1 2 3
    do
      for size in 100 200 300 400
      do
        cuda_devices=5
        echo $accent seed $seed $model
        model_dir=$PRETRAINED_CKPTS/"$model"/finetuned/asrevolve_error_model_real/$accent/$size/seed_"$seed"
        mkdir -p $model_dir
        CUDA_VISIBLE_DEVICES=$cuda_devices python3 -u finetune.py \
          --cache_dir=$CACHE_DIR \
          --wav_dir=$WAV_DIR \
          --train_manifest=$DATA/$accent/manifests/train/"$model"/asrevolve_error_model_real/$size/seed_"$seed"/train.json \
          --val_manifest=$DATA/$accent/manifests/dev.json \
          --output_dir=$model_dir/best \
          --model=$model \
          --seed=$seed \
          --lr=2e-5 \
          --batch_size=6 \
          > $model_dir/train_log.txt
        
        rm -rf $model_dir/best/tmp_checkpoints/

        CUDA_VISIBLE_DEVICES=$cuda_devices python3 -u inference.py \
          --cache_dir=$CACHE_DIR \
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

"YBAA" "ZHAA" "ASI" "TNI" "NCC" "TXHC" "EBVS" "ERMS" "YDCK" "YKWK" "THV" "TLV"
**word_enhance**
```
for model in "hubert"
do
  for accent in "YDCK"
  do
    for seed in 3
    do
      for size in 400
      do
        cuda_devices=5
        sampling_method=word_enhance
        echo $accent seed $seed $model
        model_dir=$PRETRAINED_CKPTS/"$model"/finetuned/word_error_real_mix/"$sampling_method"/$accent/$size/seed_"$seed"
        mkdir -p $model_dir
        CUDA_VISIBLE_DEVICES=$cuda_devices python3 -u finetune.py \
          --cache_dir=$CACHE_DIR \
          --wav_dir=$WAV_DIR \
          --train_manifest=$DATA/$accent/manifests/train/"$model"/word_error_predictor_real/$size/"$sampling_method"/seed_"$seed"/train.json \
          --val_manifest=$DATA/$accent/manifests/dev.json \
          --output_dir=$model_dir/best \
          --model=$model \
          --seed=$seed \
          --lr=2e-5 \
          --batch_size=6 \
          > $model_dir/train_log.txt
        
        rm -rf $model_dir/best/tmp_checkpoints/

        CUDA_VISIBLE_DEVICES=$cuda_devices python3 -u inference.py \
          --cache_dir=$CACHE_DIR \
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


"YBAA" "ZHAA"
"ASI" "TNI"
"NCC" "TXHC"
"EBVS" "ERMS"
"YDCK" "YKWK"
"THV" "TLV"
**no_word_enhance**
```
for model in "hubert"
do
  for accent in "THV" "TLV"
  do
    for seed in 1 2 3
    do
      for size in 100 200 300 400
      do
        cuda_devices=6
        sampling_method=no_word_enhance
        echo $accent seed $seed $model
        model_dir=$PRETRAINED_CKPTS/"$model"/finetuned/word_error_real_mix/"$sampling_method"/$accent/$size/seed_"$seed"
        mkdir -p $model_dir
        CUDA_VISIBLE_DEVICES=$cuda_devices python3 -u finetune.py \
          --cache_dir=$CACHE_DIR \
          --wav_dir=$WAV_DIR \
          --train_manifest=$DATA/$accent/manifests/train/"$model"/word_error_predictor_real/$size/"$sampling_method"/seed_"$seed"/train.json \
          --val_manifest=$DATA/$accent/manifests/dev.json \
          --output_dir=$model_dir/best \
          --model=$model \
          --seed=$seed \
          --lr=2e-5 \
          --batch_size=6 \
          > $model_dir/train_log.txt
        
        rm -rf $model_dir/best/tmp_checkpoints/

        CUDA_VISIBLE_DEVICES=$cuda_devices python3 -u inference.py \
          --cache_dir=$CACHE_DIR \
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