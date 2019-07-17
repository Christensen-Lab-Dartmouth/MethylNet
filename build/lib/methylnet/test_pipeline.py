import click
import os, copy
from os.path import join
import subprocess

CONTEXT_SETTINGS = dict(help_option_names=['-h','--help'], max_content_width=90)

@click.group(context_settings= CONTEXT_SETTINGS)
@click.version_option(version='0.1')
def test():
    pass

@test.command()
def test_pipeline():
    import torch
    print("WARNING: Make sure to download test data from https://github.com/Christensen-Lab-Dartmouth/MethylNet, unzip files/folder and make sure they're located in train_val_test_sets.")
    cuda_str=''
    if torch.cuda.is_available():
        cuda_str='-cu'
    commands = """mkdir visualizations results
methylnet-embed launch_hyperparameter_scan -gpu 0 {0} --hyperparameter_yaml example_embedding_hyperparameter_grid.yaml -sc Age -mc 0.84 -b 1. -g -j 20
methylnet-embed launch_hyperparameter_scan -gpu 0 {0} --hyperparameter_yaml example_embedding_hyperparameter_grid.yaml -sc Age -g -n 1 -b 1.
pymethyl-visualize transform_plot -i embeddings/vae_methyl_arr.pkl -nn 8 -c Age -o results/vae_embedding_plot.html
methylnet-predict launch_hyperparameter_scan -gpu 0 {0} --hyperparameter_yaml example_prediction_hyperparameter_grid.yaml -ic Age -mc 0.84 -g -j 20
methylnet-predict launch_hyperparameter_scan -gpu 0 {0} --hyperparameter_yaml example_prediction_hyperparameter_grid.yaml -ic Age -g -n 1
pymethyl-visualize transform_plot -i predictions/vae_mlp_methyl_arr.pkl -nn 8 -c Age -o results/mlp_embedding_plot.html
methylnet-predict regression_report
methylnet-visualize plot_training_curve -t embeddings/training_val_curve.p -vae -o results/embed_training_curve.png -thr 2e8
methylnet-visualize plot_training_curve -thr 2e6""".format(cuda_str).splitlines()
    for command in commands:
        subprocess.call(command,shell=True)
    print("Check results in embeddings, predictions, visualizations, results.")

#################

if __name__ == '__main__':
    test()
