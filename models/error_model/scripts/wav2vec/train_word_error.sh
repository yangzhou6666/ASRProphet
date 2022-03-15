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
      mkdir -p $PRETRAINED_CKPTS/word_error_predictor/$model/$accent/seed_"$seed"/best
      CUDA_VISIBLE_DEVICES=3 python3 word_error_predictor.py \
        --train_path=$DATA/$accent/manifests/"$model"_outputs/seed_out.txt \
        --test_path=$DATA/$accent/manifests/"$model"_outputs/dev_out.txt \
        --asrmodel=$model \
        --output_dir=$PRETRAINED_CKPTS/word_error_predictor/$model/$accent/seed_"$seed"/best \
        --seed=$seed \
        > $PRETRAINED_CKPTS/word_error_predictor/$model/$accent/seed_"$seed"/train_log.txt
    done 
  done &
done