#!/bin/bash -l

# declare a name for this job to be sample_job
#PBS -N methyl_zombie
# Specify the gpuq queue
#PBS -l gpus=4
#PBS -l feature=gpu
# request 4 hours and 30 minutes of cpu time
#PBS -l walltime=10:00:00
# mail is sent to you when the job starts and when it terminates or aborts

# Join error and standard output into one file
#PBS -j oe
# By default, PBS scripts execute in your home directory, not the
# directory from which they were submitted. The following line
# places you in the directory from which the job was submitted.
sleep 10800
exit 0
