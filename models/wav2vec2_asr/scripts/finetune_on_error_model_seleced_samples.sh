DATA=$(cd ../../data; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd) 

# declare -a accents=('LibriSpeech')

declare -a accents=('ST-AEDS')

lr=1e-4
ep=100

for seed in 1
do 
  for size in 100
  do
    for accent in "${accents[@]}"
    do
      echo $accent $seed $size
      echo $ep, $lr
      model_dir=$PRETRAINED_CKPTS/wav2vec2/finetuned/$accent/$size/seed_$seed/lr_$lr/epoch_$ep/error_model
      mkdir -p $model_dir
      python3 -u finetune.py \
        --train_manifest=$DATA/$accent/manifests/train/wav2vec2/error_model/$size/seed_$seed/train.json \
        --val_manifest=$DATA/$accent/manifests/dev.json \
        --output_dir=$model_dir/recent \
        --vocab=$model_dir/recent/vocab.json \
        --num_epochs=$ep \
        --learning_rate=$lr 
      > $model_dir/train_log.txt
    done
  done
done