
DATA=$(cd ../../data; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd)

#declare -a accents=('kannada_male_english' 'rajasthani_male_english' 'gujarati_female_english' 'hindi_male_english' 'malayalam_male_english' 'assamese_female_english' 'manipuri_female_english' 'tamil_male_english')
declare -a accents=('ST-AEDS')
for seed in 2
do
  for accent in "${accents[@]}"
  do
    LR=3e-4
    echo $accent seed $seed
    python3 -u train_error_model.py \
      --batch_size=50 \
      --num_epochs=300 \
      --train_freq=20 \
      --lr=$LR \
      --num_layers=4 \
      --hidden_size=64 \
      --input_size=64 \
      --weight_decay=0.001 \
      --train_portion=0.8 \
      --hypotheses_path=$DATA/$accent/manifests/quartznet_outputs/seed_plus_dev_out.txt \
      --lr_decay=warmup \
      --seed=1 \
      --output_dir=$PRETRAINED_CKPTS/error_models/quartznet/$accent/seed_"$seed"/recent \
      --best_dir=$PRETRAINED_CKPTS/error_models/quartznet/$accent/seed_"$seed"/best \
      # --pretrained_ckpt=$PRETRAINED_CKPTS/error_models/librispeech/seed_"$seed"/best/ErrorClassifierPhoneBiLSTM_V2.pt
  echo
  done
done