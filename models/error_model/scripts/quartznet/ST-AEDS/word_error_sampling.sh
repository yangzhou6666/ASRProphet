DATA=$(cd ../../data; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd)
declare -a accents=('ST-AEDS')
declare -a num_samples=(50 75 100 150)

for seed in 1 2 3
do
  for accent in "${accents[@]}"
  do
    echo $accent seed $seed num_sample $num_sample method $method
    CUDA_VISIBLE_DEVICES=7 python3 -u word_error_sampling.py \
      --seed_json_file=$DATA/$accent/manifests/seed.json \
      --selection_json_file=$DATA/$accent/manifests/selection.json \
      --finetuned_ckpt=$PRETRAINED_CKPTS/word_error_predictor/quartznet/$accent/seed_"$seed"/best \
      --data_folder=$DATA/$accent/manifests/train/random \
      --output_json_path=$DATA/$accent/manifests/train/quartznet/word_error_predictor \
      --seed=$seed
  done
done