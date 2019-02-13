#!/bin/bash -l

# declare a name for this job to be sample_job
#PBS -N methyl_embed_hyperparameters
# Specify the gpuq queue
#PBS -q gpuq
#PBS -l gpus=1
#PBS -l feature=gpu
# request 4 hours and 30 minutes of cpu time
#PBS -l walltime=02:00:00
# mail is sent to you when the job starts and when it terminates or aborts

# Join error and standard output into one file
#PBS -j oe
# By default, PBS scripts execute in your home directory, not the
# directory from which they were submitted. The following line
# places you in the directory from which the job was submitted.
cd $PBS_O_WORKDIR
gpuNum=`cat $PBS_GPUFILE | sed -e 's/.*-gpu//g'`
unset CUDA_VISIBLE_DEVICES
export CUDA_DEVICE=$gpuNum
# run the program
module load python/3-Anaconda
module load cuda
source activate methylnet_pro2
COMMAND
exit 0
