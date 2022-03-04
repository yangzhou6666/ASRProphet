DATA=$(cd ../../data; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd)

for seed in 1 2 3
do
  for num_sample in 50 75 100 150 200 300 400 500
  do
    for accent in 'ST-AEDS'
    do
      echo $accent seed $seed
      CUDA_VISIBLE_DEVICES=6 python3 -u error_model_sampling_asrevolve.py \
        --seed_json_file=$DATA/$accent/manifests/seed.json \
        --random_json_file=$DATA/$accent/manifests/train/random_tts/"$num_sample"/seed_"$seed"/train.json \
        --selection_json_file=$DATA/$accent/manifests/selection_tts.json \
        --finetuned_ckpt=$PRETRAINED_CKPTS/asrevolve_error_models/quartznet/$accent/seed_"$seed"/best \
        --log_dir=$PRETRAINED_CKPTS/asrevolve_error_models/quartznet/$accent/seed_"$seed"/train_log \
        --num_sample=$num_sample \
        --output_json_path=$DATA/$accent/manifests/train/quartznet/asrevolve_error_model \
        --exp_id=$seed
    done
  done
done