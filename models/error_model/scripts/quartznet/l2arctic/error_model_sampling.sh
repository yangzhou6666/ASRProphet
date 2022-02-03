DATA=$(cd ../../data/l2arctic/processed; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd)
declare -a accents=('ASI')
for seed in 1
do
  for accent in "${accents[@]}"
  do
    echo $accent seed $seed
    python3 -u error_model_sampling.py \
      --selection_json_file=$DATA/$accent/manifests/quartznet_outputs/selection_tts.txt \
      --seed_json_file=$DATA/$accent/manifests/seed.json \
      --error_model_weights=$PRETRAINED_CKPTS/error_models/quartznet/$accent/seed_"$seed"/best/weights.pkl \
      --random_json_path=$DATA/$accent/manifests/train/random \
      --output_json_path=$DATA/$accent/manifests/train/quartznet/error_model \
      --exp_id=$seed
  echo
  done
done