DATA=$(cd ../../data/l2arctic/processed; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd)
declare -a accents=( 'LXC' 'NJS' 'SKA' 'THV') # 
declare -a models=('hubert' 'wav2vec')
for seed in 1 2 3
do
 for accent in "${accents[@]}"
  do
    for model in "${models[@]}"
    do 
      echo $accent seed $seed $model
      
      CUDA_VISIBLE_DEVICES=0 python3 -u word_error_sampling.py \
      --seed_json_file=$DATA/$accent/manifests/seed.json \
      --data_folder=$DATA/$accent/manifests/train/random/ \
      --selection_json_file=$DATA/$accent/manifests/selection.json \
      --finetuned_ckpt=$PRETRAINED_CKPTS/word_error_predictor/$model/$accent/seed_"$seed"/best \
      --output_json_path=$DATA/$accent/manifests/train/$model/word_error_predictor_real \
      --seed=$seed
    done 
  done &
done