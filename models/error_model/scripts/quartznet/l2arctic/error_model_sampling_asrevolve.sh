DATA=$(cd ../../data/l2arctic/processed; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd)
declare -a accents=('ASI')
declare -a num_samples=(50 100 200 300 500)
for seed in 1
do
  for num_sample in "${num_samples[@]}"
  do
    for accent in "${accents[@]}"
    do
      echo $accent seed $seed
      python3 -u error_model_sampling_asrevolve.py \
        --seed_json_file=$DATA/$accent/manifests/seed.json \
        --selection_json_file=$DATA/$accent/manifests/selection.json \
        --finetuned_ckpt=$PRETRAINED_CKPTS/asrevolve_error_models/quartznet/$accent/seed_"$seed"/best \
        --log_dir=$PRETRAINED_CKPTS/asrevolve_error_models/quartznet/$accent/seed_"$seed"/train_log \
        --num_sample=$num_sample \
        --output_json_path=$DATA/$accent/manifests/train/quartznet/asrevolve_error_model \
        --exp_id=$seed
    done
  done
done