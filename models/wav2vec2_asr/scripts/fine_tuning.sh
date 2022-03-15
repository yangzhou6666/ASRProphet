DATA=$(cd ../../data/l2arctic/processed; pwd)
WAV_DIR=$(cd ../../data/l2arctic; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd)
declare -a accents=('ABA' 'BWC' 'EBVS' 'HJK' 'HKK' 'HQTV' 'LXC' 'NJS' 'SKA' 'THV') # 'ASI' 'RRBI' 'ABA' 'BWC' 'EBVS' 'HJK' 'HKK' 'HQTV' 'LXC' 'NJS' 'SKA' 'THV'
declare -a sizes=(50 75 100 150 200 300 400 500) # 50 75 100 150 200 300 400 500
declare -a models=('wav2vec')

for accent in "${accents[@]}"
do
  for size in "${sizes[@]}"
  do
    for seed in 1 2 3
    do
      for model in "${models[@]}"
      do
        echo $accent seed $seed $model
        model_dir=$PRETRAINED_CKPTS/$model/word_error_predictor/$accent/$size/seed_"$seed"
        mkdir -p $model_dir
        CUDA_VISIBLE_DEVICES=3 python3 -u finetune.py \
          --wav_dir=$WAV_DIR \
          --train_manifest=$DATA/$accent/manifests/train/$model/word_error_predictor_real/$size/word_enhance/seed_"$seed"/train.json \
          --val_manifest=$DATA/$accent/manifests/dev.json \
          --output_dir=$model_dir/best \
          --model=$model \
          --seed=$seed \
          --lr=1e-5 \
          > $model_dir/train_1e5_log.txt

        CUDA_VISIBLE_DEVICES=3 python3 -u inference.py \
          --wav_dir=$WAV_DIR \
          --val_manifest=$DATA/$accent/manifests/test.json \
          --output_file=$model_dir/test_1e5_out.txt \
          --model=$model \
          --seed=$seed \
          --checkpoint=$model_dir/best \
          > $model_dir/test_1e5_infer_log.txt
      done
    done
  done
done