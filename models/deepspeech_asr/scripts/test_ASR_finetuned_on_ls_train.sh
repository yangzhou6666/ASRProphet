DATA=$(cd ../../data; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd) 
#declare -a accents=('kannada_male_english' 'rajasthani_male_english' 'gujarati_female_english' 'hindi_male_english' 'assamese_female_english' 'malayalam_male_english' 'manipuri_female_english' 'tamil_male_english')
declare -a accents=('LibriSpeechTrain')

lr=1e-4
ep=100

for accent in "${accents[@]}"
do
  echo $accent
  echo 
  echo
  model_dir=$PRETRAINED_CKPTS/deepspeech/finetuned/$accent/lr_$lr/epoch_$ep
  python3 -u inference.py \
  --val_manifest=$DATA/LibriSpeech/manifests/all.json \
  --model=$model_dir/recent/output_graph.pbmm \
  --model_tag=deepspeech-finetuned-ls-train \
  --scorer=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.scorer \
  --output_file=$model_dir/test_out.txt \
  > $model_dir/test_infer_log.txt
done