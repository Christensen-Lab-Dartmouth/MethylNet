from preprocess import MethylationArray, extract_pheno_beta_df_from_pickle_dict
from models import AutoEncoder, TybaltTitusVAE, CVAE
from datasets import get_methylation_dataset
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss, BCELoss
import pickle
import pandas as pd, numpy as np
import click
import os
from os.path import join

CONTEXT_SETTINGS = dict(help_option_names=['-h','--help'], max_content_width=90)

@click.group(context_settings= CONTEXT_SETTINGS)
@click.version_option(version='0.1')
def embed():
    pass

def embed_vae(input_pkl,output_dir,cuda,n_latent,lr,weight_decay,n_epochs,hidden_layer_encoder_topology, convolutional = False, kl_warm_up=0, beta=1., scheduler='null', decay=0.5, t_max=10, eta_min=1e-6, t_mult=2, bce_loss=False, batch_size=50, train_percent=0.8):
    os.makedirs(output_dir,exist_ok=True)

    output_file = join(output_dir,'output_latent.csv')
    output_model = join(output_dir,'output_model.p')
    outcome_dict_file = join(output_dir,'output_outcomes.p')
    output_pkl = join(output_dir, 'vae_methyl_arr.pkl')

    input_dict = pickle.load(open(input_pkl,'rb'))
    methyl_array=MethylationArray(*extract_pheno_beta_df_from_pickle_dict(input_dict))

    train_methyl_array, test_methyl_array = methyl_array.split_train_test(train_p=train_percent, stratified=True, disease_only=True, key='disease', subtype_delimiter=',')

    train_methyl_dataset = get_methylation_dataset(train_methyl_array,'disease') # train, test split? Add val set?

    test_methyl_dataset = get_methylation_dataset(test_methyl_array,'disease')

    if not batch_size:
        batch_size=len(methyl_dataset)

    train_methyl_dataloader = DataLoader(
        dataset=train_methyl_dataset,
        num_workers=9,
        batch_size=batch_size,
        shuffle=True)

    test_methyl_dataloader = DataLoader(
        dataset=test_methyl_dataset,
        num_workers=9,
        batch_size=1,
        shuffle=False)

    if not convolutional:
        model=TybaltTitusVAE(n_input=methyl_array.return_shape()[1],n_latent=n_latent,hidden_layer_encoder_topology=hidden_layer_encoder_topology,cuda=cuda)
    else:
        model = CVAE(n_latent=n_latent,in_shape=methyl_dataset.new_shape)

    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)
    sum_bce=True
    reduction = 'sum' if sum_bce else 'elementwise_mean'
    loss_fn = BCELoss(reduction=reduction) if bce_loss else MSELoss() # 'sum'
    scheduler_opts=dict(scheduler=scheduler,lr_scheduler_decay=decay,T_max=t_max,eta_min=eta_min,T_mult=t_mult)
    auto_encoder=AutoEncoder(autoencoder_model=model,n_epochs=n_epochs,loss_fn=loss_fn,optimizer=optimizer,cuda=cuda,kl_warm_up=kl_warm_up,beta=beta, scheduler_opts=scheduler_opts)
    if 0:
        auto_encoder.add_validation_set(test_methyl_dataloader)
    auto_encoder_snapshot = auto_encoder.fit(train_methyl_dataloader).model
    del test_methyl_dataloader, train_methyl_dataloader, test_methyl_dataset, train_methyl_dataset
    methyl_dataset_loader = DataLoader(
        dataset=get_methylation_dataset(methyl_array,'disease'),
        num_workers=9,
        batch_size=1,
        shuffle=False)
    latent_projection, sample_names, outcomes = auto_encoder.transform(methyl_dataset_loader)
    print(latent_projection.shape)

    sample_names = np.array([sample_name[0] for sample_name in sample_names]) # FIXME
    outcomes = np.array([outcome[0] for outcome in outcomes]) # FIXME
    outcome_dict=dict(zip(sample_names,outcomes))
    print(methyl_array.beta.index)
    latent_projection=pd.DataFrame(latent_projection,index=methyl_array.beta.index)
    methyl_array.beta=latent_projection
    methyl_array.write_pickle(output_pkl)
    latent_projection.to_csv(output_file)
    torch.save(auto_encoder_snapshot,output_model)
    pickle.dump(outcome_dict, open(outcome_dict_file,'wb'))
    return latent_projection, outcome_dict, auto_encoder_snapshot

@embed.command()
@click.option('-i', '--input_pkl', default='./final_preprocessed/methyl_array.pkl', help='Input database for beta and phenotype data.', type=click.Path(exists=False), show_default=True)
@click.option('-o', '--output_dir', default='./embeddings/', help='Output directory for embeddings.', type=click.Path(exists=False), show_default=True)
@click.option('-c', '--cuda', is_flag=True, help='Use GPUs.')
@click.option('-n', '--n_latent', default=64, help='Number of latent dimensions.', show_default=True)
@click.option('-lr', '--learning_rate', default=1e-3, help='Number of latent dimensions.', show_default=True)
@click.option('-wd', '--weight_decay', default=1e-4, help='Weight decay of adam optimizer.', show_default=True)
@click.option('-e', '--n_epochs', default=50, help='Number of epochs to train over.', show_default=True)
@click.option('-hlt', '--hidden_layer_encoder_topology', default='', help='Topology of hidden layers, comma delimited, leave empty for one layer encoder, eg. 100,100 is example of 5-hidden layer topology.', type=click.Path(exists=False), show_default=True)
@click.option('-kl', '--kl_warm_up', default=0, help='Number of epochs before introducing kl_loss.', show_default=True)
@click.option('-b', '--beta', default=1., help='Weighting of kl divergence.', show_default=True)
@click.option('-s', '--scheduler', default='null', help='Type of learning rate scheduler.', type=click.Choice(['null','exp','warm_restarts']),show_default=True)
@click.option('-d', '--decay', default=0.5, help='Learning rate scheduler decay for exp selection.', show_default=True)
@click.option('-t', '--t_max', default=10, help='Number of epochs before cosine learning rate restart.', show_default=True)
@click.option('-eta', '--eta_min', default=1e-6, help='Minimum cosine LR.', show_default=True)
@click.option('-m', '--t_mult', default=2., help='Multiply current restart period times this number given number of restarts.', show_default=True)
@click.option('-bce', '--bce_loss', is_flag=True, help='Use bce loss instead of MSE.')
@click.option('-bs', '--batch_size', default=50, show_default=True, help='Batch size.')
@click.option('-p', '--train_percent', default=0.8, help='Percent data training on.', show_default=True)
def perform_embedding(input_pkl,output_dir,cuda,n_latent,learning_rate,weight_decay,n_epochs,hidden_layer_encoder_topology, kl_warm_up, beta, scheduler, decay, t_max, eta_min, t_mult, bce_loss, batch_size, train_percent):
    """Perform variational autoencoding on methylation dataset."""
    hlt_list=filter(None,hidden_layer_encoder_topology.split(','))
    if hlt_list:
        hidden_layer_encoder_topology=list(map(int,hlt_list))
    else:
        hidden_layer_encoder_topology=[]
    embed_vae(input_pkl,output_dir,cuda,n_latent,learning_rate,weight_decay,n_epochs,hidden_layer_encoder_topology,False,kl_warm_up,beta, scheduler, decay, t_max, eta_min, t_mult, bce_loss, batch_size, train_percent)

#################

if __name__ == '__main__':
    embed()
