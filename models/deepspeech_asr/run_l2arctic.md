# Scripts to run experiments using Deepspeech ASR on L2Arctic

The following code takes `ASI` in the L2Arctic dataset as an example. Please kindly change 

# 1. Infer ASR on dataset

## 1.1 Infer on `seed_plus_dev.json`
Generate transcripts for the seed+dev set using the pre-trainded ASR (Transcripts are used while training error models)

```
DATA=$(cd ../../data/l2arctic/processed; pwd)
WAV_DIR=$(cd ../../data/l2arctic/; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd) 
declare -a accents=('ASI' 'RRBI')
for accent in "${accents[@]}"
do
  mkdir -p $DATA/$accent/manifests/deepspeech_outputs
  echo $accent
  echo $WAV_DIR/$accent/wav
  python3 -u inference.py \
  --wav_dir=$WAV_DIR \
  --output_file=$DATA/$accent/manifests/deepspeech_outputs/seed_plus_dev_out.txt \
  --val_manifest=$DATA/$accent/manifests/seed_plus_dev.json \
  --model=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.pbmm \
  --scorer=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.scorer \
  --model_tag=deepspeech-0.9.3 \
  > $DATA/$accent/manifests/deepspeech_outputs/seed_plus_dev_infer_log.txt
done
```

## 1.2 Infer on `seed.json` and `dev.json`

The results will be used for training word error predictor.

```
for accent in 'ASI' 'RRBI'
do
  mkdir -p $DATA/$accent/manifests/deepspeech_outputs
  echo $accent
  echo $WAV_DIR/$accent/wav 
  python3 -u inference.py \
  --wav_dir=$WAV_DIR \
  --output_file=$DATA/$accent/manifests/deepspeech_outputs/seed_out.txt \
  --val_manifest=$DATA/$accent/manifests/seed.json \
  --model=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.pbmm \
  --scorer=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.scorer \
  --model_tag=deepspeech-0.9.3 \
  > $DATA/$accent/manifests/deepspeech_outputs/seed_infer_log.txt

  python3 -u inference.py \
  --wav_dir=$WAV_DIR \
  --output_file=$DATA/$accent/manifests/deepspeech_outputs/dev_out.txt \
  --val_manifest=$DATA/$accent/manifests/dev.json \
  --model=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.pbmm \
  --scorer=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.scorer \
  --model_tag=deepspeech-0.9.3 \
  > $DATA/$accent/manifests/deepspeech_outputs/dev_infer_log.txt
done
```


## Infer on test dataset

Let's try on the test set.

```
for accent in "${accents[@]}"
do
  mkdir -p $DATA/$accent/manifests/deepspeech_outputs
  echo $accent
  echo $WAV_DIR/$accent/wav 
  python3 -u inference.py \
  --wav_dir=$WAV_DIR \
  --output_file=$DATA/$accent/manifests/deepspeech_outputs/original_test_out_out.txt \
  --val_manifest=$DATA/$accent/manifests/test.json \
  --model=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.pbmm \
  --scorer=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.scorer \
  --model_tag=deepspeech-0.9.3 \
  > $DATA/$accent/manifests/deepspeech_outputs/original_test_infer_log.txt
done
```

Let's also try on the synthetic test set.

```
for accent in "${accents[@]}"
do
  mkdir -p $DATA/$accent/manifests/deepspeech_outputs
  echo $accent
  echo $WAV_DIR/$accent/wav 
  python3 -u inference.py \
  --wav_dir=$WAV_DIR \
  --output_file=$DATA/$accent/manifests/deepspeech_outputs/original_test_out_out_tts.txt \
  --val_manifest=$DATA/$accent/manifests/test_tts.json \
  --model=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.pbmm \
  --scorer=$PRETRAINED_CKPTS/deepspeech/deepspeech-0.9.3-models.scorer \
  --model_tag=deepspeech-0.9.3 \
  > $DATA/$accent/manifests/deepspeech_outputs/original_test_infer_log_tts.txt
done
```
