
DATA=$(cd ../../data; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd)
declare -a accents=('ST-AEDS')
#     
for seed in 1 2 3
do
  for accent in "${accents[@]}"
  do
    LR=3e-4
    echo $accent seed $seed
    mkdir -p $PRETRAINED_CKPTS/asrevolve_error_models/quartznet/$accent/seed_"$seed"
    CUDA_VISIBLE_DEVICES=6 python3 -u train_error_model_asrevolve.py \
      --train_path=$DATA/$accent/manifests/quartznet_outputs/seed_out.txt \
      --test_path=$DATA/$accent/manifests/quartznet_outputs/dev_out.txt \
      --seed=$seed \
      --output_dir=$PRETRAINED_CKPTS/asrevolve_error_models/quartznet/$accent/seed_"$seed"/best \
      --log_dir=$PRETRAINED_CKPTS/asrevolve_error_models/quartznet/$accent/seed_"$seed"/train_log \
      > $PRETRAINED_CKPTS/asrevolve_error_models/quartznet/$accent/seed_"$seed"/train_log.txt 
  done 
done 