DATA=$(cd ../../data; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd) 
declare -a accents=('LibriSpeech')
for accent in "${accents[@]}"
do
  mkdir -p $DATA/$accent/manifests/deepspeech_outputs
  echo $accent
  python3 -u deepspeech_inference.py \
  --output_file=$DATA/$accent/manifests/deepspeech_outputs/seed_plus_dev_out.txt \
  --val_manifest=$DATA/$accent/manifests/seed_plus_dev.json \
  --model=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.pbmm \
  --scorer=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.scorer \
  > $DATA/$accent/manifests/deepspeech_outputs/seed_plus_dev_infer_log.txt
done