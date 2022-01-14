DATA=$(cd ../../data; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd)

declare -a accents=('LibriSpeech')
for seed in 1
do
  for accent in "${accents[@]}"
  do
    LR=1e-4
    echo $accent seed $seed
    CUDA_VISIBLE_DEVICES=1 python3 -u train_error_model.py \
      --batch_size=32 \
      --num_epochs=200 \
      --train_freq=20 \
      --lr=$LR \
      --num_layers=2 \
      --hidden_size=128 \
      --input_size=128 \
      --weight_decay=0.001 \
      --train_portion=0.8 \
      --hypotheses_path=$DATA/$accent/manifests/deepspeech_outputs/seed_plus_dev_out.txt \
      --lr_decay=warmup \
      --seed=$seed \
      --output_dir=$PRETRAINED_CKPTS/error_models/deepspeech/$accent/seed_"$seed"/recent \
      --best_dir=$PRETRAINED_CKPTS/error_models/deepspeech/$accent/seed_"$seed"/best 
      # --pretrained_ckpt=$PRETRAINED_CKPTS/error_models/librispeech/seed_"$seed"/best/ErrorClassifierPhoneBiLSTM_V2.pt
  echo
  done
done