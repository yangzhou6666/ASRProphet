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
  mkdir -p $model_dir
  python3 -u finetune.py \
    --train_manifest=$DATA/$accent/manifests/all.json \
    --val_manifest=$DATA/LibriSpeech/manifests/all.json \
    --load_checkpoint_dir=$PRETRAINED_CKPTS/deepspeech/checkpoints/deepspeech-0.9.3-checkpoint/ \
    --save_checkpoint_dir=$PRETRAINED_CKPTS/deepspeech/checkpoints/deepspeech-ls-train-finetuned-checkpoint/ \
    --model_scorer=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.scorer \
    --output_dir=$model_dir/recent \
    --gpu_id=0 \
  > $model_dir/train_log.txt
done
