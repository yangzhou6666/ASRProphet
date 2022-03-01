DATA=$(cd ../../data/l2arctic/processed; pwd)
WAV_DIR=$(cd ../../data/l2arctic; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd) 
declare -a accents=('ASI')

for seed in 1
do 
  for size in 1500
  do
    for accent in "${accents[@]}"
    do
      echo $accent $seed $size
      echo 
      echo
      model_dir=$PRETRAINED_CKPTS/deepspeech/finetuned/$accent/$size/seed_"$seed"/random
      mkdir -p $model_dir
      python3 -u finetune.py \
        --train_manifest=$DATA/$accent/manifests/train/deepspeech/random/$size/seed_"$seed"/train.json \
        --val_manifest=$DATA/$accent/manifests/dev.json \
        --wav_dir=$WAV_DIR \
        --load_checkpoint_dir=$PRETRAINED_CKPTS/deepspeech/checkpoints/deepspeech-0.9.3-checkpoint/ \
        --save_checkpoint_dir=$PRETRAINED_CKPTS/deepspeech/checkpoints/deepspeech-random-checkpoint/ \
        --model_scorer=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.scorer \
        --output_dir=$model_dir/recent \
        --gpu_id=0 \
      > $model_dir/train_log.txt
    done
  done
done