# Scripts to run experiments on L2Arctic

The following code takes `ASI` in the L2Arctic dataset as an example. Please kindly change 

## Infer on seed+dev dataset

Generate transcripts for the seed+dev set using the pre-trainded ASR (Transcripts are used while training error models)

```
DATA=$(cd ../../data/l2arctic/processed; pwd)
WAV_DIR=$(cd ../../data/l2arctic/; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd) 
declare -a accents=('ASI' 'RRBI')
for accent in "${accents[@]}"
do
  mkdir -p $DATA/$accent/manifests/quartznet_outputs
  echo $accent
  echo $WAV_DIR/$accent/wav 
  python3 -u inference.py \
  --batch_size=16 \
  --output_file=$DATA/$accent/manifests/quartznet_outputs/seed_plus_dev_out.txt \
  --wav_dir=$WAV_DIR \
  --val_manifest=$DATA/$accent/manifests/seed_plus_dev.json \
  --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
  --ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
  > $DATA/$accent/manifests/quartznet_outputs/seed_plus_dev_infer_log.txt 
done
```

We can also evaluate the perform on synthetic seed and dev dataset.

```
for accent in 'ASI' 'RRBI'
do
  mkdir -p $DATA/$accent/manifests/quartznet_outputs
  echo $accent
  echo $WAV_DIR/$accent/wav 
  python3 -u inference.py \
  --batch_size=16 \
  --output_file=$DATA/$accent/manifests/quartznet_outputs/seed_plus_dev_out_tts.txt \
  --wav_dir=$WAV_DIR \
  --val_manifest=$DATA/$accent/manifests/seed_plus_dev_tts.json \
  --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
  --ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
  > $DATA/$accent/manifests/quartznet_outputs/seed_plus_dev_infer_log_tts.txt 
done
```

## Infer on test dataset

Let's try on the test set.


```
for accent in "${accents[@]}"
do
  mkdir -p $DATA/$accent/manifests/quartznet_outputs
  echo $accent
  echo $WAV_DIR/$accent/wav 
  python3 -u inference.py \
  --batch_size=16 \
  --output_file=$DATA/$accent/manifests/quartznet_outputs/original_test_out.txt \
  --wav_dir=$WAV_DIR \
  --val_manifest=$DATA/$accent/manifests/test.json \
  --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
  --ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
  > $DATA/$accent/manifests/quartznet_outputs/original_test_infer_log.txt 
done
```

Let's also try on the synthetic test set.

```
for accent in "${accents[@]}"
do
  mkdir -p $DATA/$accent/manifests/quartznet_outputs
  echo $accent
  echo $WAV_DIR/$accent/wav 
  python3 -u inference.py \
  --batch_size=16 \
  --output_file=$DATA/$accent/manifests/quartznet_outputs/original_test_out_tts.txt \
  --wav_dir=$WAV_DIR \
  --val_manifest=$DATA/$accent/manifests/test_tts.json \
  --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
  --ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
  > $DATA/$accent/manifests/quartznet_outputs/original_test_infer_log_tts.txt 
done
```


## Train Error Model
Please go to `/model/error_model` first.

Notice: this step needs the hypotheses file (i.e, `seed_plus_dev_out.txt`) infered in previous steps. 

```
for seed in 1 2 3
do
  for accent in "${accents[@]}"
  do
    LR=3e-4
    echo $accent seed $seed
    mkdir -p $PRETRAINED_CKPTS/error_models/quartznet/$accent/seed_"$seed"/
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
      --hypotheses_path=$DATA/$accent/manifests/quartznet_outputs/seed_plus_dev_out.txt \
      --lr_decay=warmup \
      --seed=1 \
      --output_dir=$PRETRAINED_CKPTS/error_models/quartznet/$accent/seed_"$seed"/recent \
      --best_dir=$PRETRAINED_CKPTS/error_models/quartznet/$accent/seed_"$seed"/best \
    > $PRETRAINED_CKPTS/error_models/quartznet/$accent/seed_"$seed"/train_log.txt 
  echo
  done 
done 
```
Assuming that the seed is `1`, the model will be saved under `/models/pretrained_checkpoints/error_model/quartznet/ASI/seed_1/`, and the train log is stored in the same folder. 


## Infer the error model on the dataset for selection

```
for seed in {1..3}
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
      --pretrained_ckpt=$PRETRAINED_CKPTS/error_models/quartznet/$accent/seed_"$seed"/best/ErrorClassifierPhoneBiLSTM_V2.pt \
      --output_dir=$PRETRAINED_CKPTS/error_models/quartznet/$accent/seed_"$seed"/best \
    > $PRETRAINED_CKPTS/error_models/quartznet/$accent/seed_"$seed"/infer_log.txt
  echo
  done
done
```

## Training the common error predictor

```
echo 
for seed in {1..3}
do
  for accent in 'ASI' 'RRBI'
  do
    LR=3e-4
    echo $accent seed $seed
    mkdir -p $PRETRAINED_CKPTS/matcher/quartznet/$accent/seed_"$seed"/
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
      --asr_hypotheses_path=$DATA/$accent/manifests/quartznet_outputs/seed_plus_dev_out.txt \
      --tts_hypotheses_path=$DATA/$accent/manifests/quartznet_outputs/seed_plus_dev_out_tts.txt \
      --lr_decay=warmup \
      --seed=1 \
      --output_dir=$PRETRAINED_CKPTS/matcher/quartznet/$accent/seed_"$seed"/recent \
      --best_dir=$PRETRAINED_CKPTS/matcher/quartznet/$accent/seed_"$seed"/best 
  echo
  done 
done 
```

```
for seed in {1..3}
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
      --pretrained_ckpt=$PRETRAINED_CKPTS/matcher/quartznet/$accent/seed_"$seed"/best/ErrorClassifierPhoneBiLSTM_V2.pt \
      --output_dir=$PRETRAINED_CKPTS/matcher/quartznet/$accent/seed_"$seed"/best \
    > $PRETRAINED_CKPTS/matcher/quartznet/$accent/seed_"$seed"/infer_log.txt
  echo
  done
done
```

## Training the error estimator from ASREvolve


for seed in {1..3}
do
  for accent in 'ASI' 'RRBI'
  do
    LR=3e-4
    echo $accent seed $seed
    mkdir -p $PRETRAINED_CKPTS/asrevolve_error_models/quartznet/$accent/seed_"$seed"
    CUDA_VISIBLE_DEVICES=2 python3 -u train_error_model_asrevolve.py \
      --batch_size=10 \
      --num_epochs=200 \
      --train_freq=20 \
      --lr=$LR \
      --num_layers=4 \
      --hidden_size=64 \
      --input_size=64 \
      --weight_decay=0.001 \
      --train_portion=0.8 \
      --hypotheses_path=$DATA/$accent/manifests/quartznet_outputs/seed_plus_dev_out.txt \
      --lr_decay=warmup \
      --seed=1 \
      --output_dir=$PRETRAINED_CKPTS/asrevolve_error_models/quartznet/$accent/seed_"$seed"/best \
      --log_dir=$PRETRAINED_CKPTS/asrevolve_error_models/quartznet/$accent/seed_"$seed"/train_log \
      > $PRETRAINED_CKPTS/asrevolve_error_models/quartznet/$accent/seed_"$seed"/train_log.txt 
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
        --finetuned_ckpt=$PRETRAINED_CKPTS/asrevolve_error_models/quartznet/$accent/seed_"$seed"/best \
        --log_dir=$PRETRAINED_CKPTS/asrevolve_error_models/quartznet/$accent/seed_"$seed"/train_log \
        --num_sample=$num_sample \
        --output_json_path=$DATA/$accent/manifests/train/quartznet/asrevolve_error_model \
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
      echo 
      echo
      python3 -u inference.py \
      --batch_size=64 \
      --output_file=$DATA/$accent/manifests/train/quartznet/error_model_tts/$size/seed_"$seed"/test_out_ori.txt \
      --wav_dir=$WAV_DIR \
      --val_manifest=$DATA/$accent/manifests/train/quartznet/error_model_tts/$size/seed_"$seed"/train.json \
      --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
      --ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
      > $DATA/$accent/manifests/train/quartznet/error_model_tts/$size/seed_"$seed"/test_out_ori_log.txt
    done
  done
done
```




## Using error model to select valuable data examples

This paper aims to select synthetic data. Please use the following commands.

```
for seed in {1..3}
do
  for accent in 'ASI' 'RRBI'
  do
    LR=3e-4
    echo $accent seed $seed
    mkdir -p $PRETRAINED_CKPTS/asrevolve_error_models/quartznet/$accent/seed_"$seed"
    python3 -u train_error_model_asrevolve.py \
      --batch_size=10 \
      --num_epochs=200 \
      --train_freq=20 \
      --lr=$LR \
      --num_layers=4 \
      --hidden_size=64 \
      --input_size=64 \
      --weight_decay=0.001 \
      --train_portion=0.8 \
      --hypotheses_path=$DATA/$accent/manifests/quartznet_outputs/seed_plus_dev_out.txt \
      --lr_decay=warmup \
      --seed=1 \
      --output_dir=$PRETRAINED_CKPTS/asrevolve_error_models/quartznet/$accent/seed_"$seed"/best \
      --log_dir=$PRETRAINED_CKPTS/asrevolve_error_models/quartznet/$accent/seed_"$seed"/train_log \
      > $PRETRAINED_CKPTS/asrevolve_error_models/quartznet/$accent/seed_"$seed"/train_log.txt 
  done 
done 
```


### Error Model Only

```
for seed in 1 2 3
do
  for accent in 'ASI' 'RRBI'
  do
    echo $accent seed $seed
    python3 -u error_model_sampling.py \
      --selection_json_file=$DATA/$accent/manifests/selection_tts.json \
      --seed_json_file=$DATA/$accent/manifests/seed.json \
      --error_model_weights=$PRETRAINED_CKPTS/error_models/quartznet/$accent/seed_"$seed"/best/weights.pkl \
      --random_json_path=$DATA/$accent/manifests/train/random_tts \
      --output_json_path=$DATA/$accent/manifests/train/quartznet/error_model_tts \
      --exp_id=$seed
  echo
  done
done
```

Evaluation

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
      python3 -u inference.py \
      --batch_size=64 \
      --output_file=$DATA/$accent/manifests/train/quartznet/error_model_tts/$size/seed_"$seed"/test_out_ori.txt \
      --wav_dir=$WAV_DIR \
      --val_manifest=$DATA/$accent/manifests/train/quartznet/error_model_tts/$size/seed_"$seed"/train.json \
      --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
      --ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
      > $DATA/$accent/manifests/train/quartznet/error_model_tts/$size/seed_"$seed"/test_out_ori_log.txt
    done
  done
done
```




### Error Model + Matcher

```
for seed in {1..3}
do
  for accent in 'ASI' 'RRBI'
  do
    echo $accent seed $seed
    python3 -u combined_metric_sampling.py \
      --selection_json_file=$DATA/$accent/manifests/selection_tts.json \
      --seed_json_file=$DATA/$accent/manifests/seed.json \
      --error_model_weights=$PRETRAINED_CKPTS/error_models/quartznet/$accent/seed_"$seed"/best/weights.pkl \
      --matcher_model_weights=$PRETRAINED_CKPTS/matcher/quartznet/$accent/seed_"$seed"/best/weights.pkl \
      --random_json_path=$DATA/$accent/manifests/train/random_tts \
      --output_json_path=$DATA/$accent/manifests/train/quartznet/matcher \
      --exp_id=$seed
  echo
  done
done
```



Evaluation

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
      python3 -u inference.py \
      --batch_size=64 \
      --output_file=$DATA/$accent/manifests/train/quartznet/matcher/$size/seed_"$seed"/test_out_ori.txt \
      --wav_dir=$WAV_DIR \
      --val_manifest=$DATA/$accent/manifests/train/quartznet/matcher/$size/seed_"$seed"/train.json \
      --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
      --ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
      > $DATA/$accent/manifests/train/quartznet/matcher/$size/seed_"$seed"/test_out_ori_log.txt
    done
  done
done
```


## Using error-model selected audio to fine-tune and evaluation

Let's go back to `model/quartznet_asr` folder to fine-tune ASR models. We first fine-tune on synthetic data selected by the error model obtained in the previous step. The selected data is stored in `manifests/train/quartznet/error_model_tts/$size/seed_"$seed"/train.json`.

```
for seed in {1..3}
do 
  for size in 50 75 100 150 200 300 400 500
  do
    for accent in 'ASI' 'RRBI'
    do
      echo $accent $seed $size
      echo 
      echo
      model_dir=$PRETRAINED_CKPTS/quartznet/finetuned/$accent/$size/seed_"$seed"/error_model_tts
      mkdir -p $model_dir
      python3 -u finetune.py \
        --batch_size=16 \
        --num_epochs=100 \
        --eval_freq=1 \
        --train_freq=30 \
        --lr=1e-5 \
        --wav_dir=$WAV_DIR \
        --train_manifest=$DATA/$accent/manifests/train/quartznet/error_model_tts/$size/seed_"$seed"/train.json \
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
        --seed=42 \
        --optimizer=novograd \
      > $model_dir/train_log.txt

      echo $accent $seed $size
      echo 
      echo
      model_dir=$PRETRAINED_CKPTS/quartznet/finetuned/$accent/$size/seed_"$seed"/error_model_tts
      python3 -u inference.py \
      --batch_size=64 \
      --output_file=$model_dir/test_out.txt \
      --wav_dir=$WAV_DIR \
      --val_manifest=$DATA/$accent/manifests/test.json \
      --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
      --ckpt=$model_dir/best/Jasper.pt \
      > $model_dir/test_infer_log.txt
    done
  done
done
```



## Using error model + matcher selected audio to fine-tune and evaluation


```
for seed in {1..3}
do 
  for size in 50 75 100 150 200 300 400 500
  do
    for accent in 'ASI' 'RRBI'
    do
      echo $accent $seed $size
      echo 
      echo
      model_dir=$PRETRAINED_CKPTS/quartznet/finetuned/$accent/$size/seed_"$seed"/matcher
      mkdir -p $model_dir
      CUDA_VISIBLE_DEVICES=6 python3 -u finetune.py \
        --batch_size=16 \
        --num_epochs=100 \
        --eval_freq=1 \
        --train_freq=30 \
        --lr=1e-5 \
        --wav_dir=$WAV_DIR \
        --train_manifest=$DATA/$accent/manifests/train/quartznet/matcher/$size/seed_"$seed"/train.json \
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
        --seed=42 \
        --optimizer=novograd \
      > $model_dir/train_log.txt

      echo $accent $seed $size
      echo 
      echo
      model_dir=$PRETRAINED_CKPTS/quartznet/finetuned/$accent/$size/seed_"$seed"/matcher
      CUDA_VISIBLE_DEVICES=6 python3 -u inference.py \
      --batch_size=64 \
      --output_file=$model_dir/test_out.txt \
      --wav_dir=$WAV_DIR \
      --val_manifest=$DATA/$accent/manifests/test.json \
      --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
      --ckpt=$model_dir/best/Jasper.pt \
      > $model_dir/test_infer_log.txt
    done
  done
done
```


## Using randomly selected audio to fine-tune and evaluation


Evaluating original performance on randomly selected data.

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
      python3 -u inference.py \
      --batch_size=64 \
      --output_file=$DATA/$accent/manifests/train/random_tts/$size/seed_"$seed"/test_out_ori.txt \
      --wav_dir=$WAV_DIR \
      --val_manifest=$DATA/$accent/manifests/train/random_tts/$size/seed_"$seed"/train.json \
      --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
      --ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
      > $DATA/$accent/manifests/train/random_tts/$size/seed_"$seed"/test_out_ori_log.txt
    done
  done
done
```

We also fine-tune the model on randomly selected data.

```
for seed in {1..3}
do 
  for size in 50 75 100 150 200 300 400 500
  do
    for accent in 'ASI' 'RRBI'
    do
      echo $accent $seed $size
      echo 
      echo
      model_dir=$PRETRAINED_CKPTS/quartznet/finetuned/$accent/$size/seed_"$seed"/random_tts
      mkdir -p $model_dir
      CUDA_VISIBLE_DEVICES=2 python3 -u finetune.py \
        --batch_size=16 \
        --num_epochs=100 \
        --eval_freq=1 \
        --train_freq=30 \
        --lr=1e-5 \
        --wav_dir=$WAV_DIR \
        --train_manifest=$DATA/$accent/manifests/train/random_tts/$size/seed_"$seed"/train.json \
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
        --seed=42 \
        --optimizer=novograd \
      > $model_dir/train_log.txt

      echo $accent $seed $size
      echo 
      echo
      model_dir=$PRETRAINED_CKPTS/quartznet/finetuned/$accent/$size/seed_"$seed"/random_tts
      CUDA_VISIBLE_DEVICES=2 python3 -u inference.py \
      --batch_size=64 \
      --output_file=$model_dir/test_out.txt \
      --wav_dir=$WAV_DIR \
      --val_manifest=$DATA/$accent/manifests/test.json \
      --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
      --ckpt=$model_dir/best/Jasper.pt \
      > $model_dir/test_infer_log.txt
    done
  done
done
```

## Train on ASREvolve data

```
for seed in {1..3}
do 
  for size in 50 75 100 150 200 300 400 500
  do
    for accent in 'ASI' 'RRBI'
    do
      echo $accent $seed $size
      echo 
      echo
      model_dir=$PRETRAINED_CKPTS/quartznet/finetuned/$accent/$size/seed_"$seed"/asrevolve_tts
      mkdir -p $model_dir
      CUDA_VISIBLE_DEVICES=5 python3 -u finetune.py \
        --batch_size=16 \
        --num_epochs=100 \
        --eval_freq=1 \
        --train_freq=30 \
        --lr=1e-5 \
        --wav_dir=$WAV_DIR \
        --train_manifest=$DATA/$accent/manifests/train/quartznet/asrevolve_error_model/$size/seed_"$seed"/train.json \
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
        --seed=42 \
        --optimizer=novograd \
      > $model_dir/train_log.txt
    done
  done
done
```


for seed in {1..3}
do 
  for size in 50 75 100 150 200 300 400 500
  do
    for accent in 'ASI' 'RRBI'
    do
      echo $accent $seed $size
      echo 
      echo
      model_dir=$PRETRAINED_CKPTS/quartznet/finetuned/$accent/$size/seed_"$seed"/asrevolve_tts
      CUDA_VISIBLE_DEVICES=1 python3 -u inference.py \
      --batch_size=64 \
      --output_file=$model_dir/test_out.txt \
      --wav_dir=$WAV_DIR \
      --val_manifest=$DATA/$accent/manifests/test.json \
      --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
      --ckpt=$model_dir/best/Jasper.pt \
      > $model_dir/test_infer_log.txt
    done
  done
done