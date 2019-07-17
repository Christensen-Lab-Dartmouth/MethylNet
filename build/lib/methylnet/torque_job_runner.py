import click, os
from methylnet.torque_jobs import *

CONTEXT_SETTINGS = dict(help_option_names=['-h','--help'], max_content_width=90)

@click.group(context_settings= CONTEXT_SETTINGS)
@click.version_option(version='0.1')
def torque():
    pass

@torque.command()
@click.option('-c', '--command', default='', help='Command to execute through torque.', show_default=True)
@click.option('-gpu', '--use_gpu', is_flag=True, help='Specify whether to use GPUs.', show_default=True)
@click.option('-a', '--additions', default='', help='Additional commands to add, one liner for now.', show_default=True)
@click.option('-q', '--queue', default='default', help='Queue for torque submission, gpuq also a good one if using GPUs.', show_default=True)
@click.option('-t', '--time', default=1, help='Walltime in hours for job.', show_default=True)
@click.option('-n', '--ngpu', default=0, help='Number of gpus to request.', show_default=True)
@click.option('-ao', '--additional_options', default='', help='Additional options to add for torque run.', type=click.Path(exists=False))
def run_torque_job(command, use_gpu, additions, queue, time, ngpu, additional_options):
    """Run torque job."""
    replace_dict = assemble_replace_dict(command, use_gpu, additions, queue, time, ngpu)
    run_torque_job_(replace_dict, additional_options)

if __name__ == '__main__':
    torque()
