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
      echo 
      echo
      model_dir=$PRETRAINED_CKPTS/deepspeech/finetuned/$accent/$size/seed_"$seed"/lr_$lr/epoch_$epoch/error_model
      python3 -u inference.py \
      --val_manifest=$DATA/$accent/manifests/test.json \
      --model=$model_dir/recent/output_graph.pbmm \
      --model_tag=deepspeech-finetuned-seed$seed-size$size \
      --scorer=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.scorer \
      --output_file=$model_dir/test_out.txt \
      > $model_dir/test_infer_log.txt
    done
  done
done