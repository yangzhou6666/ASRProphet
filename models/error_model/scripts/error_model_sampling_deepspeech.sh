DATA=$(cd ../../data; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd)

declare -a accents=('LibriSpeech')
for seed in 1
do
  for accent in "${accents[@]}"
  do
    echo $accent seed $seed
    python3 -u error_model_sampling.py \
      --selection_json_file=$DATA/$accent/manifests/selection.json \
      --seed_json_file=$DATA/$accent/manifests/seed.json \
      --error_model_weights=$PRETRAINED_CKPTS/error_models/deepspeech/$accent/seed_"$seed"/best/weights.pkl \
      --random_json_path=$DATA/$accent/manifests/train/random \
      --output_json_path=$DATA/$accent/manifests/train/error_model \
      --exp_id=$seed
  echo
  done
done