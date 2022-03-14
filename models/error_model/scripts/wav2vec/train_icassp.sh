DATA=$(cd ../../data/l2arctic/processed; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd)
declare -a accents=('SVBI' 'TNI')
declare -a models=('wav2vec' 'hubert')

for seed in 1 2 3
do
  for accent in "${accents[@]}"
  do
    for model in "${models[@]}"
    do
      echo $accent seed $seed $model
      mkdir -p $PRETRAINED_CKPTS/error_models/$model/$accent/seed_"$seed"/
      CUDA_VISIBLE_DEVICES=1 python3 -u train_error_model.py \
        --batch_size=16 \
        --train_path=$DATA/$accent/manifests/"$model"_outputs/seed_out.txt \
        --test_path=$DATA/$accent/manifests/"$model"_outputs/dev_out.txt \
        --lr_decay=warmup \
        --seed=$seed \
        --output_dir=$PRETRAINED_CKPTS/error_models/$model/$accent/seed_"$seed"/recent \
        --best_dir=$PRETRAINED_CKPTS/error_models/$model/$accent/seed_"$seed"/best \
      > $PRETRAINED_CKPTS/error_models/$model/$accent/seed_"$seed"/train_log.txt
    done
  done &
done 