DATA=$(cd ../../data; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd)

#declare -a accents=('kannada_male_english' 'rajasthani_male_english' 'gujarati_female_english' 'hindi_male_english' 'malayalam_male_english' 'assamese_female_english' 'manipuri_female_english' 'tamil_male_english')

declare -a accents=('ST-AEDS')
for seed in 2
do
  for accent in "${accents[@]}"
  do
    echo $accent seed $seed
    python3 -u infer_error_model.py \
      --batch_size=64 \
      --num_layers=4 \
      --hidden_size=64 \
      --input_size=64 \
      --json_path=$DATA/$accent/manifests/quartznet_outputs/selection_out.txt \
      --pretrained_ckpt=$PRETRAINED_CKPTS/error_models/quartznet/$accent/seed_"$seed"/best/ErrorClassifierTTS.pt \
      --output_dir=$PRETRAINED_CKPTS/error_models/quartznet/$accent/seed_"$seed"/best \
    > $PRETRAINED_CKPTS/error_models/quartznet/$accent/seed_"$seed"/infer_log.txt
  echo
  done
done