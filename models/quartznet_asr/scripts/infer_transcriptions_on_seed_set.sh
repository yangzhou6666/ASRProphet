DATA=$(cd ../../data; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd) 
# declare -a accents=('kannada_male_english' 'rajasthani_male_english' 'gujarati_female_english' 'hindi_male_english' 'assamese_female_english' 'malayalam_male_english' 'manipuri_female_english' 'tamil_male_english')
# declare -a accents=('kannada_male_english')
declare -a accents=('ST-AEDS')
for accent in "${accents[@]}"
do
  mkdir -p $DATA/$accent/manifests/quartznet_outputs
  echo $accent
  python3 -u inference.py \
  --batch_size=16 \
  --output_file=$DATA/$accent/manifests/quartznet_outputs/selection_out.txt \
  --wav_dir=$DATA/$accent/data \
  --val_manifest=$DATA/$accent/manifests/selection.json \
  --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
  --ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
  > $DATA/$accent/manifests/quartznet_outputs/selection_log.txt
done