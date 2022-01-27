DATA=$(cd ../../data; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd) 
#declare -a accents=('kannada_male_english' 'rajasthani_male_english' 'gujarati_female_english' 'hindi_male_english' 'assamese_female_english' 'malayalam_male_english' 'manipuri_female_english' 'tamil_male_english')

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
      model_dir=$PRETRAINED_CKPTS/wav2vec2/finetuned/$accent/$size/seed_$seed/lr_$lr/epoch_$ep/error_model
      python3 -u inference.py \
      --output_file=$model_dir/test_out.txt \
      --val_manifest=$DATA/$accent/manifests/test.json \
      --model_name=$model_dir/recent/checkpoint-500 \
      --vocab=$model_dir/recent/vocab.json \
      --model_tag=wav2vec2-finetuned-seed$seed-size$size \
      --overwrite \
      > $model_dir/test_infer_log.txt
    done
  done
done