DATA=$(cd ../../data/l2arctic/processed; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd)
declare -a accents=('ABA' 'ASI' 'BWC' 'EBVS' 'ERMS' 'HJK' 'HKK' 'HQTV' 'LXC' 'MBMPS' 'NCC' 'NJS' 'PNV' 'RRBI' 'SKA' 'SVBI' 'THV' 'TLV' 'TNI' 'TXHC' 'YBAA' 'YDCK' 'YKWK' 'ZHAA' 'suitcase_corpus')
for seed in 1
do
  for accent in "${accents[@]}"
  do
    echo $accent seed $seed
    python3 -u infer_error_model.py \
      --batch_size=64 \
      --num_layers=4 \
      --hidden_size=64 \
      --input_size=64 \
      --json_path=$DATA/$accent/manifests/selection.json \
      --pretrained_ckpt=$PRETRAINED_CKPTS/error_models/quartznet/$accent/seed_"$seed"/best/ErrorClassifierPhoneBiLSTM_V2.pt \
      --output_dir=$PRETRAINED_CKPTS/error_models/quartznet/$accent/seed_"$seed"/best \
    > $PRETRAINED_CKPTS/error_models/quartznet/$accent/seed_"$seed"/infer_log.txt
  echo
  done
done