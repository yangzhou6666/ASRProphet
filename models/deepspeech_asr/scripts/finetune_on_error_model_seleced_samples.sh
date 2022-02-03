DATA=$(cd ../../data; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd) 
#declare -a accents=('kannada_male_english' 'rajasthani_male_english' 'gujarati_female_english' 'hindi_male_english' 'assamese_female_english' 'malayalam_male_english' 'manipuri_female_english' 'tamil_male_english')
declare -a accents=('LibriSpeech')
lr=1e-4
ep=100

for seed in 1
do 
  for size in 1500
  do
    for accent in "${accents[@]}"
    do
      echo $accent $seed $size
      echo $ep, $lr
      echo
      echo
      model_dir=$PRETRAINED_CKPTS/deepspeech/finetuned/$accent/$size/seed_"$seed"/lr_"$lr"/epoch_$epoch/error_model
      mkdir -p $model_dir
      python3 -u finetune.py \
        --train_manifest=$DATA/$accent/manifests/train/deepspeech/error_model/$size/seed_"$seed"/train.json \
        --val_manifest=$DATA/$accent/manifests/dev.json \
        --load_checkpoint_dir=$PRETRAINED_CKPTS/deepspeech/checkpoints/deepspeech-0.9.3-checkpoint/ \
        --model_scorer=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.scorer \
        --output_dir=$model_dir/recent \
        --gpu_id=1 \
        --num_epochs=$ep \
        --learning_rate=$lr 
      > $model_dir/train_log.txt
    done
  done
done