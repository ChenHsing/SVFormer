python setup.py build develop
python run_net_emamix.py \
  --cfg configs/Kinetics/TimeSformer_base_ssl.yaml \
  DATA.PATH_TO_DATA_DIR ./dataset/list_k400_1/ \
  OUTPUT_DIR ./output/list_k400_1/k400_1/ \
  NUM_GPUS 8 \
  TRAIN.BATCH_SIZE 8 \
  TEST.BATCH_SIZE 64 \
  TRAIN.ENABLE False
  TRAIN.FINETUNE False \
  TRAIN.CHECKPOINT_FILE_PATH ./checkpoints/checkpoint.pyth

