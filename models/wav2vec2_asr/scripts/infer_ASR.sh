DATA=$(cd ../../data/l2arctic/processed; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd)
WAV_DIR=$(cd ../../data/l2arctic; pwd)
declare -a accents=('ASI' 'RRBI' 'ABA' 'BWC' 'EBVS' 'HJK' 'HKK' 'HQTV' 'LXC' 'NJS' 'SKA' 'THV')
declare -a models=('hubert' 'wav2vec')

for accent in "${accents[@]}"
do
  for model in "${models[@]}"
  do
    mkdir -p $DATA/$accent/manifests/"$model"_outputs
    echo $accent
    CUDA_VISIBLE_DEVICES=1 python3 -u inference.py \
    --wav_dir=$WAV_DIR \
    --val_manifest=$DATA/$accent/manifests/seed.json \
    --output_file=$DATA/$accent/manifests/"$model"_outputs/seed_out.txt \
    --model $model \
    > $DATA/$accent/manifests/"$model"_outputs/seed_infer_log.txt
  done
done &