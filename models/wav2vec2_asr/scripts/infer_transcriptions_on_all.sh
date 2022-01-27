DATA=$(cd ../../data; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd) 
# declare -a accents=('LibriSpeech')
declare -a accents=('ST-AEDS')
for accent in "${accents[@]}"
do
  mkdir -p $DATA/$accent/manifests/wav2vec2_outputs
  echo $accent
  python3 -u inference.py \
  --output_file=$DATA/$accent/manifests/wav2vec2_outputs/all_out.txt \
  --val_manifest=$DATA/$accent/manifests/all.json \
  --model_name=facebook/wav2vec2-base-960h \
  --vocab=facebook/wav2vec2-base-960h \
  --model_tag=wav2vec2 \
  > $DATA/$accent/manifests/wav2vec2_outputs/all_infer_log.txt
done