DATA=$(cd ../../data/l2arctic/processed; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd)
declare -a accents=('ASI')
  
for seed in 1
do
  for accent in "${accents[@]}"
  do
    LR=3e-4
    echo $accent seed $seed
    mkdir -p $PRETRAINED_CKPTS/error_models/deepspeech/$accent/seed_"$seed"
    CUDA_VISIBLE_DEVICES=6 python3 -u train_error_model.py \
      --batch_size=10 \
      --num_epochs=200 \
      --train_freq=20 \
      --lr=$LR \
      --num_layers=4 \
      --hidden_size=64 \
      --input_size=64 \
      --weight_decay=0.001 \
      --train_portion=0.8 \
      --hypotheses_path=$DATA/$accent/manifests/deepspeech_outputs/seed_plus_dev_out.txt \
      --lr_decay=warmup \
      --seed=1 \
      --output_dir=$PRETRAINED_CKPTS/error_models/deepspeech/$accent/seed_"$seed"/recent \
      --best_dir=$PRETRAINED_CKPTS/error_models/deepspeech/$accent/seed_"$seed"/best 
       > $PRETRAINED_CKPTS/error_models/deepspeech/$accent/seed_"$seed"/train_log.txt 
  echo
  done 
done 