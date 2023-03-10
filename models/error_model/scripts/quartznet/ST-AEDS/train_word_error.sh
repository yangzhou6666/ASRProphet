DATA=$(cd ../../data; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd)
declare -a accents=('ST-AEDS')

for seed in 1 2 3
do
  for accent in "${accents[@]}"
  do
  echo $accent seed $seed
  mkdir -p $PRETRAINED_CKPTS/word_error_predictor/quartznet/$accent/seed_"$seed"/best
  CUDA_VISIBLE_DEVICES=5 python word_error_predictor.py \
    --train_path=$DATA/$accent/manifests/quartznet_outputs/seed_out.txt \
    --test_path=$DATA/$accent/manifests/quartznet_outputs/dev_out.txt \
    --seed=$seed \
    --output_dir=$PRETRAINED_CKPTS/word_error_predictor/quartznet/$accent/seed_"$seed"/best \
    > $PRETRAINED_CKPTS/word_error_predictor/quartznet/$accent/seed_"$seed"/training.log
  done
done