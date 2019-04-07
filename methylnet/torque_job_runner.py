import click

CONTEXT_SETTINGS = dict(help_option_names=['-h','--help'], max_content_width=90)

@click.group(context_settings= CONTEXT_SETTINGS)
@click.version_option(version='0.1')
def torque():
    pass

# FINISH add GPU and others

def run_torque_job_(replace_dict):
    import subprocess
    with open('pbs_commands/run_torque.sh','r') as f:
        txt=f1.read()
    for k,v in replace_dict.items():
        txt = txt.replace(k,v)
    with open('torque_job.sh','w') as f:
        f.write(txt)
    subprocess.call("mksub {}".format('torque_job.sh'),shell=True)

@torque.command()
@click.option('-c', '--command', default='', help='Submit torque job.', show_default=True)
def run_torque_job(command):
    replace_dict = {'COMMAND':command}
    run_torque_job_(replace_dict)

if __name__ == '__main__':
    torque()
