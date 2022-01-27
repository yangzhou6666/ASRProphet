DATA=$(cd ../../data/l2arctic/processed; pwd)
WAV_DIR=$(cd ../../data/l2arctic/; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd) 
declare -a accents=('ABA')
for accent in "${accents[@]}"
do
  mkdir -p $DATA/$accent/manifests/quartznet_outputs
  echo $accent
  echo $WAV_DIR/$accent/wav 
  CUDA_VISIBLE_DEVICES=0 python3 -u inference.py \
  --batch_size=16 \
  --output_file=$DATA/$accent/manifests/quartznet_outputs/seed_plus_dev_out.txt \
  --wav_dir=$WAV_DIR \
  --val_manifest=$DATA/$accent/manifests/seed_plus_dev.json \
  --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
  --ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
  > $DATA/$accent/manifests/quartznet_outputs/seed_plus_dev_infer_log.txt &
done
