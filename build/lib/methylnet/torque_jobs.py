"""
torque_jobs.py
=======================
Wraps and runs your commands through torque.
"""

import os

def assemble_replace_dict(command, use_gpu, additions, queue, time, ngpu):
    """Create dictionary to update BASH submission script for torque.

    Parameters
    ----------
    command : type
        Command to executer through torque.
    use_gpu : type
        GPUs needed?
    additions : type
        Additional commands to add (eg. module loads).
    queue : type
        Queue to place job in.
    time : type
        How many hours to run job for.
    ngpu : type
        Number of GPU to use.

    Returns
    -------
    Dict
        Dictionary used to update Torque Script.

    """
    replace_dict = {'COMMAND':"{} {}".format('CUDA_VISIBLE_DEVICES="$gpuNum"' if use_gpu else '',command),
                'GPU_SETUP':"""gpuNum=`cat $PBS_GPUFILE | sed -e 's/.*-gpu//g'`
unset CUDA_VISIBLE_DEVICES
export CUDA_DEVICE=$gpuNum""" if use_gpu else '',
                'NGPU':'#PBS -l gpus={}'.format(ngpu) if (use_gpu and ngpu) else '',
                'USE_GPU':"#PBS -l feature=gpu" if use_gpu else '',
                'TIME':str(time),'QUEUE':queue,'ADDITIONS':additions}
    return replace_dict

def run_torque_job_(replace_dict, additional_options=""):
    """Run torque job after creating submission script.

    Parameters
    ----------
    replace_dict : type
        Dictionary used to replace information in bash script to run torque job.
    additional_options : type
        Additional options to pass scheduler.

    Returns
    -------
    str
        Custom torque job name.

    """
    txt="""#!/bin/bash -l
#PBS -N run_torque
#PBS -q QUEUE
NGPU
USE_GPU
#PBS -l walltime=TIME:00:00
#PBS -j oe
cd $PBS_O_WORKDIR
GPU_SETUP
ADDITIONS
COMMAND"""
    for k,v in replace_dict.items():
        txt = txt.replace(k,v)
    with open('torque_job.sh','w') as f:
        f.write(txt)
    job=os.popen("mksub {} {}".format('torque_job.sh',additional_options)).read().strip('\n')
    return job

def assemble_run_torque(command, use_gpu, additions, queue, time, ngpu, additional_options=""):
    """Runs torque job after passing commands to setup bash file.

    Parameters
    ----------
    command : type
        Command to executer through torque.
    use_gpu : type
        GPUs needed?
    additions : type
        Additional commands to add (eg. module loads).
    queue : type
        Queue to place job in.
    time : type
        How many hours to run job for.
    ngpu : type
        Number of GPU to use.
    additional_options : type
        Additional options to pass to Torque scheduler.

    Returns
    -------
    job
        Custom job name.

    """
    job = run_torque_job_(assemble_replace_dict(command, use_gpu, additions, queue, time, ngpu),additional_options)
    return job
