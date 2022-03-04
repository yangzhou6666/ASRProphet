
DATA=$(cd ../../data; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd)
declare -a accents=('ST-AEDS')
#     
for seed in 1 2 3
do
  for accent in "${accents[@]}"
  do
    echo $accent seed $seed
    mkdir -p $PRETRAINED_CKPTS/error_models/quartznet/$accent/seed_"$seed"/
    CUDA_VISIBLE_DEVICES=4 python3 -u train_error_model.py \
      --batch_size=16 \
      --train_path=$DATA/$accent/manifests/quartznet_outputs/seed_out.txt \
      --test_path=$DATA/$accent/manifests/quartznet_outputs/dev_out.txt \
      --lr_decay=warmup \
      --seed=$seed \
      --output_dir=$PRETRAINED_CKPTS/error_models/quartznet/$accent/seed_"$seed"/recent \
      --best_dir=$PRETRAINED_CKPTS/error_models/quartznet/$accent/seed_"$seed"/best \
    > $PRETRAINED_CKPTS/error_models/quartznet/$accent/seed_"$seed"/train_log.txt
  done 
done 