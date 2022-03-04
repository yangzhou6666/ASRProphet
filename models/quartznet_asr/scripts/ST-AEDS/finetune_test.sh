DATA=$(cd ../../data; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd) 

for seed in 1
do 
  for size in 50 75 100 200
  do
    for accent in 'ST-AEDS'
    do
      echo $accent $seed $size
      model_dir=$PRETRAINED_CKPTS/quartznet/finetuned/$accent/$size/seed_"$seed"/word_error/real
    #   mkdir -p $model_dir
      # CUDA_VISIBLE_DEVICES=4 python3 -u finetune.py \
      #   --batch_size=16 \
      #   --num_epochs=100 \
      #   --eval_freq=1 \
      #   --train_freq=30 \
      #   --lr=1e-5 \
      #   --wav_dir=$WAV_DIR \
      #   --train_manifest=$DATA/$accent/manifests/train/quartznet/word_error_predictor/$size/normal/seed_"$seed"/train.json \
      #   --val_manifest=$DATA/$accent/manifests/dev.json \
      #   --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
      #   --output_dir=$model_dir/recent \
      #   --best_dir=$model_dir/best \
      #   --early_stop_patience=10 \
      #   --zero_infinity \
      #   --save_after_each_epoch \
      #   --turn_bn_eval \
      #   --ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
      #   --lr_decay=warmup \
      #   --seed=42 \
      #   --optimizer=novograd \
      # > $model_dir/training_log.txt

      echo $accent $seed $size
      CUDA_VISIBLE_DEVICES=5 python3 -u inference.py \
      --batch_size=16 \
      --output_file=$model_dir/test_out.txt \
      --wav_dir=$WAV_DIR \
      --val_manifest=$DATA/$accent/manifests/test.json \
      --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
      --ckpt=$model_dir/best/Jasper.pt \
      > $model_dir/test_infer_log.txt
    done
  done
done