"""
torque_job_runner.py
=======================
Wraps and runs your commands through torque.
"""

import click, os

CONTEXT_SETTINGS = dict(help_option_names=['-h','--help'], max_content_width=90)

@click.group(context_settings= CONTEXT_SETTINGS)
@click.version_option(version='0.1')
def torque():
    pass

def assemble_replace_dict(command, use_gpu, additions, queue, time, ngpu):
    """Short summary.

    Parameters
    ----------
    command : type
        Description of parameter `command`.
    use_gpu : type
        Description of parameter `use_gpu`.
    additions : type
        Description of parameter `additions`.
    queue : type
        Description of parameter `queue`.
    time : type
        Description of parameter `time`.
    ngpu : type
        Description of parameter `ngpu`.

    Returns
    -------
    type
        Description of returned object.

    """
    replace_dict = {'COMMAND':"{} {}".format('CUDA_VISIBLE_DEVICES="$gpuNum"' if use_gpu else '',command),
                'GPU_SETUP':"""gpuNum=`cat $PBS_GPUFILE | sed -e 's/.*-gpu//g'`
unset CUDA_VISIBLE_DEVICES
export CUDA_DEVICE=$gpuNum""" if use_gpu else '',
                'NGPU':'#PBS -l gpus={}'.format(ngpu) if ngpu else '',
                'USE_GPU':"#PBS -l feature=gpu" if use_gpu else '',
                'TIME':str(time),'QUEUE':queue,'ADDITIONS':additions}
    return replace_dict

def run_torque_job_(replace_dict, additional_options=""):
    """Short summary.

    Parameters
    ----------
    replace_dict : type
        Description of parameter `replace_dict`.
    additional_options : type
        Description of parameter `additional_options`.

    Returns
    -------
    type
        Description of returned object.

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
    """Short summary.

    Parameters
    ----------
    command : type
        Description of parameter `command`.
    use_gpu : type
        Description of parameter `use_gpu`.
    additions : type
        Description of parameter `additions`.
    queue : type
        Description of parameter `queue`.
    time : type
        Description of parameter `time`.
    ngpu : type
        Description of parameter `ngpu`.
    additional_options : type
        Description of parameter `additional_options`.

    Returns
    -------
    type
        Description of returned object.

    """
    job = run_torque_job_(assemble_replace_dict(command, use_gpu, additions, queue, time, ngpu),additional_options)
    return job

@torque.command()
@click.option('-c', '--command', default='', help='Submit torque job.', show_default=True)
@click.option('-gpu', '--use_gpu', is_flag=True, help='Specify whether to use GPUs.', show_default=True)
@click.option('-a', '--additions', default='', help='Additional commands to add, one liner for now.', show_default=True)
@click.option('-q', '--queue', default='default', help='Queue for torque submission, gpuq also a good one if using GPUs.', show_default=True)
@click.option('-t', '--time', default=1, help='Walltime in hours for job.', show_default=True)
@click.option('-n', '--ngpu', default=0, help='Number of gpus to request.', show_default=True)
def run_torque_job(command, use_gpu, additions, queue, time, ngpu):
    replace_dict = assemble_replace_dict(command, use_gpu, additions, queue, time, ngpu)
    run_torque_job_(replace_dict)

if __name__ == '__main__':
    torque()
