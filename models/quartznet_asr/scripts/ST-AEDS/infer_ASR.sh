DATA=$(cd ../../data; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd) 

declare -a accents=('ST-AEDS')
for seed in 1 2 3
do 
  for size in 50 75 100 150 200 300 400 500
  do
    for accent in "${accents[@]}"
    do
      mkdir -p $PRETRAINED_CKPTS/quartznet/finetuned/$accent/$size/seed_"$seed"/word_error/real
      echo $accent $seed $size
        CUDA_VISIBLE_DEVICES=7 python3 -u inference.py \
        --batch_size=32 \
        --wav_dir $DATA \
        --output_file=$PRETRAINED_CKPTS/quartznet/finetuned/$accent/$size/seed_"$seed"/word_error/real/train.txt \
        --val_manifest=$DATA/$accent/manifests/train/quartznet/word_error_predictor/$size/real/seed_"$seed"/train.json \
        --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
        --ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
        > $PRETRAINED_CKPTS/quartznet/finetuned/$accent/$size/seed_"$seed"/word_error/real/train_log.txt
    done
  done
done