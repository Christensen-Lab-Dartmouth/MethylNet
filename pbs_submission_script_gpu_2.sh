#!/bin/bash -l

# declare a name for this job to be sample_job
#PBS -N methyl_gpu_2
# Specify the gpuq queue
#PBS -q gpuq
# Specify the number of gpus
#PBS -l nodes=1:ppn=10
#PBS -l gpus=1
# gpus ppn was 4 and 4, figure out in future
#PBS -l hostlist=g03
# Specify the gpu feature
#PBS -l feature=gpu
#PBS -l mem=50GB
# request 4 hours and 30 minutes of cpu time
#PBS -l walltime=01:00:00
# mail is sent to you when the job starts and when it terminates or aborts

# Join error and standard output into one file
#PBS -j oe
# By default, PBS scripts execute in your home directory, not the
# directory from which they were submitted. The following line
# places you in the directory from which the job was submitted.
cd $PBS_O_WORKDIR
# run the program
gpuNum=`cat $PBS_GPUFILE | sed -e 's/.*-gpu//g'`
unset CUDA_VISIBLE_DEVICES
export CUDA_DEVICE=$gpuNum
module load python/3-Anaconda
module load cuda
echo $gpuNum
source activate py36
CUDA_VISIBLE_DEVICES="$gpuNum" python predictions.py make_prediction -cat -hlt 100 -s warm_restarts -bs 256 -c -v -p 0.4 -e 5
exit 0