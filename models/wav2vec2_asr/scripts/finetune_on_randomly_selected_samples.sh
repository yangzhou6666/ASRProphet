DATA=$(cd ../../data; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd) 
#declare -a accents=('kannada_male_english' 'rajasthani_male_english' 'gujarati_female_english' 'hindi_male_english' 'assamese_female_english' 'malayalam_male_english' 'manipuri_female_english' 'tamil_male_english')
# declare -a accents=('LibriSpeech')

lr=1e-4
ep=100


declare -a accents=('ST-AEDS')

for seed in 1
do 
  for size in 100
  do
    for accent in "${accents[@]}"
    do
      echo $accent $seed $size
      echo 
      echo
      model_dir=$PRETRAINED_CKPTS/wav2vec2/finetuned/$accent/$size/seed_"$seed"/random
      mkdir -p $model_dir
      python3 -u finetune.py \
        --train_manifest=$DATA/$accent/manifests/train/random/$size/seed_"$seed"/train.json \
        --val_manifest=$DATA/$accent/manifests/dev.json \
        --output_dir=$model_dir/recent \
        --vocab=$model_dir/recent/vocab.json \
        --num_epochs=$ep \
        --learning_rate=$lr \
        --batch_size=16 
      > $model_dir/train_log.txt
    done
  done
done