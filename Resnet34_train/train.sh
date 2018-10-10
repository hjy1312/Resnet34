#!/usr/bin/env sh
if [ ! -d "./log" ]; then
  mkdir ./log
fi
LOG=./log/log-`date +%Y-%m-%d-%H:%M:%S`.log
PYDIR=/home/junyanghuang/anaconda2/bin
nohup $PYDIR/python -B resnet34_main.py --train_list /home/junyanghuang/download/training_list_without_deduplication.txt --batchSize 320 \
 --cuda --ngpu 2  \
 --outf ./train_resnet34_`date +%Y-%m-%d-%H:%M:%S` 2>&1 | tee $LOG&
#--Resnet34 /home/junyanghuang/experiment/ResNet34/ArcFace/train_resnet34_2018-06-05-15:52:34/Resnet34_epoch_8.pth
