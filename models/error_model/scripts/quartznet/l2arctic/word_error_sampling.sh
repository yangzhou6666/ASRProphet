DATA=$(cd ../../data/l2arctic/processed; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd)
declare -a accents=('ASI')
declare -a num_samples=(50)
for seed in 1 
do
  for num_sample in "${num_samples[@]}"
  do
    for accent in "${accents[@]}"
    do
      echo $accent seed $seed
      CUDA_VISIBLE_DEVICES=6 python3 -u word_error_sampling.py \
        --seed_json_file=$DATA/$accent/manifests/seed.json \
        --random_json_file=$DATA/$accent/manifests/train/random_tts/"$num_sample"/seed_"$seed"/train.json \
        --selection_json_file=$DATA/$accent/manifests/selection_tts.json \
        --finetuned_ckpt=$PRETRAINED_CKPTS/word_error_predictor/quartznet/$accent/seed_"$seed"/best \
        --log_dir=$PRETRAINED_CKPTS/word_error_predictor/quartznet/$accent/seed_"$seed"/train_log \
        --num_sample=$num_sample \
        --output_json_path=$DATA/$accent/manifests/train/quartznet/word_error_predictor \
        --seed=$seed
    done
  done
done