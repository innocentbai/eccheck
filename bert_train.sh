#!/bin/sh

export MASTER_ADDR=localhost
export MASTER_PORT=29500
# export NCCL_IB_DISABLE=1 

# sudo mount -t tmpfs -o size=200G tmpfs /home/lff/eccheck/ckpt  挂载并指定大小为 20GB
# sudo umount /mnt/my_tmpfs
# Training parameters
DATASET=wikitext-103
MODEL=bert-large-uncased
EPOCHS=1
BATCH_SIZE=32
FREQ=2
RESUME=0
CPU_HOME=/home/lff/eccheck/ckpt/
# Save_DIR=/data/lff/eccheck

# Distributed training with DeepSpeed
# deepspeed --hostfil=hostfile cv.py
# pkill -f python
# stop:0.2 0.6
deepspeed bert_train.py \
  --ckpt_run \
  --dataset $DATASET \
  --model $MODEL \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --freq $FREQ \
  --resume $RESUME \
  --cpu_home $CPU_HOME \
  --ec_run
