DATA=$(cd ../../data/l2arctic/processed; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd)
WAV_DIR=$(cd ../../data/l2arctic; pwd)
declare -a accents=('ABA' 'BWC' 'EBVS' 'HJK' 'HKK' 'HQTV' 'LXC' 'NJS' 'SKA' 'THV') # 'RRBI' 'ABA' 'BWC' 'EBVS' 'HJK' 'HKK' 'HQTV' 'LXC' 'NJS' 'SKA' 'THV')
declare -a sizes=(50 75 100 150 200 300 400 500)
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
        mkdir -p $PRETRAINED_CKPTS/$model/word_error_predictor/$accent/$size/seed_"$seed"
        CUDA_VISIBLE_DEVICES=0 python3 -u inference.py \
        --wav_dir=$WAV_DIR \
        --val_manifest=$DATA/$accent/manifests/train/$model/word_error_predictor_real/$size/word_enhance/seed_"$seed"/train.json \
        --output_file=$PRETRAINED_CKPTS/$model/word_error_predictor/$accent/$size/seed_"$seed"/RQ1_out.txt \
        --model $model \
        --seed=$seed \
        > $PRETRAINED_CKPTS/$model/word_error_predictor/$accent/$size/seed_"$seed"/RQ1_log.txt
      done
    done
  done
done