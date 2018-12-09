from preprocess import MethylationArray, extract_pheno_beta_df_from_pickle_dict
from models import AutoEncoder, TybaltTitusVAE
from datasets import get_methylation_dataset
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
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

def embed_vae(input_pkl,output_dir,cuda,n_latent,lr,weight_decay,n_epochs,hidden_layer_encoder_topology, convolutional = False):
    os.makedirs(output_dir,exist_ok=True)

    output_file = join(output_dir,'output_latent.csv')
    output_model = join(output_dir,'output_model.p')
    outcome_dict_file = join(output_dir,'output_outcomes.p')
    output_pkl = join(output_dir, 'vae_methyl_arr.pkl')

    input_dict = pickle.load(open(input_pkl,'rb'),protocol=4)
    methyl_array=MethylationArray(*extract_pheno_beta_df_from_pickle_dict(input_dict))

    methyl_dataset = get_methylation_dataset(methyl_array,'disease') # train, test split? Add val set?

    methyl_dataloader = DataLoader(
        dataset=methyl_dataset,
        num_workers=4,
        batch_size=1,
        shuffle=False)

    if not convolutional:
        model=TybaltTitusVAE(n_input=methyl_array.return_shape()[1],n_latent=n_latent,hidden_layer_encoder_topology=hidden_layer_encoder_topology)
    else:
        model = CVAE(n_latent=n_latent,in_shape=methyl_dataset.new_shape)

    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)

    loss_fn = MSELoss()

    auto_encoder=AutoEncoder(autoencoder_model=model,n_epochs=n_epochs,loss_fn=loss_fn,optimizer=optimizer,cuda=cuda)
    auto_encoder_snapshot = auto_encoder.fit(methyl_dataloader)
    latent_projection, sample_names, outcomes = auto_encoder.transform(methyl_dataloader)
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
def perform_embedding(input_pkl,output_dir,cuda,n_latent,learning_rate,weight_decay,n_epochs,hidden_layer_encoder_topology):
    hlt_list=filter(None,hidden_layer_encoder_topology.split(','))
    if hlt_list:
        hidden_layer_encoder_topology=list(map(int,hlt_list))
    else:
        hidden_layer_encoder_topology=[]
    embed_vae(input_pkl,output_dir,cuda,n_latent,learning_rate,weight_decay,n_epochs,hidden_layer_encoder_topology)

#################

if __name__ == '__main__':
    embed()
