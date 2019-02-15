from umap import UMAP
import plotly.graph_objs as go
import plotly.offline as py
import pandas as pd
import numpy as np
import networkx as nx
import click
import pickle
from sklearn.preprocessing import LabelEncoder

CONTEXT_SETTINGS = dict(help_option_names=['-h','--help'], max_content_width=90)

@click.group(context_settings= CONTEXT_SETTINGS)
@click.version_option(version='0.1')
def visualize():
    pass

@visualize.command()
@click.option('-t', '--training_curve_file', default='predictions/training_val_curve.p', show_default=True, help='Training and validation loss and learning curves.', type=click.Path(exists=False))
@click.option('-o', '--outputfilename', default='results/training_curve.png', show_default=True, help='Output image.', type=click.Path(exists=False))
@click.option('-vae', '--vae_train', is_flag=True, help='Plot VAE Training curves.', type=click.Path(exists=False))
def plot_training_curve(training_curve_file, outputfilename, vae_train):
    import os
    os.makedirs(outputfilename[:outputfilename.rfind('/')],exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="whitegrid")
    df=pd.DataFrame(pickle.load(open(training_curve_file,'rb')))
    plt.subplots(1,3 if vae_train else 2,figsize=(15 if vae_train else 10,5))
    plt.subplot(131 if vae_train else 121)
    sns.lineplot(data=df[['loss','val_loss']+(['recon_loss','val_recon_loss'] if vae_train else [])], palette="tab10", linewidth=2.5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.subplot(132 if vae_train else 122)
    sns.lineplot(data=df[['lr_vae', 'lr_mlp']], palette="tab10", linewidth=2.5)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    if vae_train:
        plt.subplot(133)
        sns.lineplot(data=df[['kl_loss','val_kl_loss']], palette="tab10", linewidth=2.5)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(outputfilename)

#################

if __name__ == '__main__':
    visualize()
