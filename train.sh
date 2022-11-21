python setup.py build develop

python run_net_emamix.py \
  --cfg configs/Kinetics/TimeSformer_base_ssl.yaml \
  DATA.PATH_TO_DATA_DIR ./dataset/list_ucf_10/ \
  OUTPUT_DIR /mnt/blob/output/list_ucf_10/ucf_small_10_joint/ \
  NUM_GPUS 8 \
  TRAIN.BATCH_SIZE 8 \
  TEST.BATCH_SIZE 64 \
  TRAIN.ENABLE False
  TRAIN.FINETUNE False \
  TRAIN.CHECKPOINT_FILE_PATH ./timesformer_10/checkpoints/checkpoint_epoch_00030.pyth

