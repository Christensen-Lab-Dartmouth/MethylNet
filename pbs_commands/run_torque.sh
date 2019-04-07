#!/bin/bash -l

#PBS -N run_torque
#PBS -q QUEUE
#PBS -l gpus=NGPU
#Specify whether use PBS -l feature=gpu
USE_GPU
#PBS -l walltime=TIME:00:00
#PBS -j oe

cd $PBS_O_WORKDIR
# gpuNum=`cat $PBS_GPUFILE | sed -e 's/.*-gpu//g'`
# unset CUDA_VISIBLE_DEVICES
# export CUDA_DEVICE=$gpuNum
USE_GPU_COMMANDS
module load python/3-Anaconda
module load cuda
echo $gpuNum
source activate methylnet_pro2
CUDA_VISIBLE_DEVICES="$gpuNum" COMMAND
