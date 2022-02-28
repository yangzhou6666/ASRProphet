DATA=$(cd ../../data/l2arctic/processed; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd)
declare -a accents=('ASI')
declare -a num_samples=(50 75 100 150)

WAV_DIR=$(cd ../../data/l2arctic/; pwd)

for seed in 1
do 
  for size in "${num_samples[@]}"
  do
    for accent in "${accents[@]}"
    do
      echo $accent $seed $size
      echo 
      echo
      model_dir=$PRETRAINED_CKPTS/quartznet/finetuned/$accent/$size/seed_"$seed"/asrevolve_tts
      mkdir -p $model_dir
      python3 -u inference.py \
      --batch_size=64 \
      --output_file=$model_dir/test_out.txt \
      --wav_dir=$WAV_DIR \
      --val_manifest=$DATA/$accent/manifests/train/quartznet/asrevolve_error_model/$size/seed_"$seed"/train.json \
      --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
      --ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
      > $model_dir/test_infer_log_ori.txt
    done
  done
done