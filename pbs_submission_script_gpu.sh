#!/bin/bash -l

# declare a name for this job to be sample_job
#PBS -N methyl_gpu
# Specify the gpuq queue
#PBS -q gpuq
# Specify the number of gpus
#PBS -l nodes=1:ppn=10
#PBS -l gpus=1
#PBS -l hostlist=g04
# Specify the gpu feature
#PBS -l feature=gpu
#PBS -l mem=50GB
# request 4 hours and 30 minutes of cpu time
#PBS -l walltime=20:00:00
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
CUDA_VISIBLE_DEVICES="$gpuNum" python embedding.py perform_embedding -n 100 -bs 512 -hlt 500 -kl 20 --t_max 50 --eta_min 5e-4 --t_mult 2 -b 5. -s warm_restarts -lr 1e-2 -bce -e 1500 -c
exit 0
