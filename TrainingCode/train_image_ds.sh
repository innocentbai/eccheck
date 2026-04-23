#!/bin/bash
# export PYTHONPATH=/home/lff/RCKPT/DeepSpeed:$PYTHONPATH
# python test_ds.py
deepspeed train_image_ds.py -a resnet50 --deepspeed --deepspeed_config ds_config.json --multiprocessing_distributed --dataset /data/dataset/cv/imagenet_0908
