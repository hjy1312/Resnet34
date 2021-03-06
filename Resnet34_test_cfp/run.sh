#!/usr/bin/env sh
if [ ! -d "./log" ]; then
  mkdir ./log
fi
LOG=./log/log-`date +%Y-%m-%d-%H:%M:%S`.log
PYDIR=/home/junyang/anaconda2/bin
nohup $PYDIR/python -B resnet34_cal_fea.py --batchSize 200 \
 --cuda --ngpu 1 --Resnet34 /data/hjy1312/download/model_arcface/Resnet34_epoch_25.pth \
 --gallery_list /data/dataset/CFP/aligned_Pair_list_P.txt \
 --outf ./fea_resnet34 2>&1 | tee $LOG&
