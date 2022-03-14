DATA=$(cd ../../data/l2arctic/processed; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd)
declare -a accents=('SVBI' 'TNI')
declare -a models=('hubert' 'wav2vec')

for seed in 1 2 3
do
  for accent in "${accents[@]}"
  do
    for model in "${models[@]}"
    do 
      echo $accent seed $seed $model
      
      CUDA_VISIBLE_DEVICES=1 python3 -u infer_error_model.py \
        --batch_size=64 \
        --json_path=$DATA/$accent/manifests/selection.json \
        --pretrained_ckpt=$PRETRAINED_CKPTS/error_models/$model/$accent/seed_"$seed"/best/ErrorClassifierPhoneBiLSTM_V2.pt \
        --output_dir=$PRETRAINED_CKPTS/error_models/$model/$accent/seed_"$seed"/best \
        > $PRETRAINED_CKPTS/error_models/$model/$accent/seed_"$seed"/infer_log.txt

      CUDA_VISIBLE_DEVICES=1 python3 -u error_model_sampling.py \
        --selection_json_file=$DATA/$accent/manifests/selection.json \
        --seed_json_file=$DATA/$accent/manifests/seed.json \
        --error_model_weights=$PRETRAINED_CKPTS/error_models/$model/$accent/seed_"$seed"/best/weights.pkl \
        --random_json_path=$DATA/$accent/manifests/train/random \
        --output_json_path=$DATA/$accent/manifests/train/$model/error_model \
        --exp_id=$seed
    done
  done &
done