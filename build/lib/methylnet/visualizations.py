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
@click.option('-vae', '--vae_train', is_flag=True, help='Plot VAE Training curves.')
@click.option('-thr', '--threshold', default=1e10, help='Loss values get rid of greater than this.')
def plot_training_curve(training_curve_file, outputfilename, vae_train, threshold):
    """Plot training curves as output from either the VAE training or VAE MLP."""
    import os
    os.makedirs(outputfilename[:outputfilename.rfind('/')],exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    #from scipy.signal import find_peaks_cwt
    #from impyute import moving_window
    def impute_peaks(signal):
        signal=signal.values
        idxs=signal>=threshold
        signal[idxs]=np.nan
        """for idx in idxs:
            signal[idx]=(signal[idx-1]+signal[idx+1])/2 if idx else signal[idx+1]"""
        return signal
    def impute_df(df):
        df=df.apply(impute_peaks,axis=0).fillna(method='bfill',axis=0)
        print(df)
        return df
    sns.set(style="whitegrid")
    df=pd.DataFrame(pickle.load(open(training_curve_file,'rb')))
    plt.subplots(1,3 if vae_train else 2,figsize=(15 if vae_train else 10,5))
    plt.subplot(131 if vae_train else 121)
    if vae_train:
        df['loss']=df['recon_loss']+df['kl_loss']
        df['val_loss']=df['val_recon_loss']+df['val_kl_loss']
    sns.lineplot(data=impute_df(df[['loss','val_loss']+(['recon_loss','val_recon_loss'] if vae_train else [])]), palette="tab10", linewidth=2.5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.subplot(132 if vae_train else 122)
    sns.lineplot(data=df[['lr_vae', 'lr_mlp'] if not vae_train else ['lr']], palette="tab10", linewidth=2.5)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    if vae_train:
        plt.subplot(133)
        sns.lineplot(data=impute_df(df[['kl_loss','val_kl_loss']]), palette="tab10", linewidth=2.5)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(outputfilename)

@visualize.command()
@click.option('-r', '--roc_curve_csv', default='results/Weighted_ROC.csv', show_default=True, help='Weighted ROC Curve.', type=click.Path(exists=False))
@click.option('-o', '--outputfilename', default='results/roc_curve.png', show_default=True, help='Output image.', type=click.Path(exists=False))
def plot_roc_curve(roc_curve_csv, outputfilename):
    """Plot ROC Curves from classification tasks; requires classification_report be run first."""
    import os
    from rpy2.robjects.packages import importr
    import rpy2.robjects as robjects
    os.makedirs(outputfilename[:outputfilename.rfind('/')],exist_ok=True)
    tidyverse=importr('tidyverse')
    robjects.r("""function (in.csv, out.file.name) {
        df<-read.csv(in.csv)
        df %>%
            ggplot() +
            geom_line(aes(x = fpr, y = tpr,color=Legend)) +
            ggtitle('ROC Curve') +
            xlab('1-Specificity') +
            ylab('Sensitivity')
        ggsave(out.file.name)
        }""")(roc_curve_csv,outputfilename)

#################

if __name__ == '__main__':
    visualize()
