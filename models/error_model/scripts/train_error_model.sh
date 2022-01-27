DATA=$(cd ../../data; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd)

# declare -a accents=('LibriSpeech')
declare -a accents=('ST-AEDS')
declare -a asrs=('deepspeech' 'quartznet' 'wav2vec2')

for seed in 1
do
  for accent in "${accents[@]}"
  do
    for asr in "${asrs[@]}"
    do
      LR=3e-4
      echo $accent seed $seed
      python3 -u train_error_model.py \
        --batch_size=100 \
        --num_epochs=200 \
        --train_freq=20 \
        --lr=$LR \
        --num_layers=4 \
        --hidden_size=64 \
        --input_size=64 \
        --weight_decay=0.001 \
        --train_portion=0.8 \
        --hypotheses_path=$DATA/$accent/manifests/"$asr"_outputs/seed_plus_dev_out.txt \
        --lr_decay=warmup \
        --seed=$seed \
        --output_dir=$PRETRAINED_CKPTS/error_models/"$asr"/$accent/seed_"$seed"/recent \
        --best_dir=$PRETRAINED_CKPTS/error_models/"$asr"/$accent/seed_"$seed"/best \
        --pretrained_ckpt=$PRETRAINED_CKPTS/error_models/librispeech/seed_"$seed"/best/ErrorClassifierPhoneBiLSTM_V2.pt
      echo
    done
  done
done