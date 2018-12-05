from preprocess import MethylationArray, extract_pheno_beta_df_from_sql
from models import AutoEncoder, TybaltTitusVAE
from datasets import get_methylation_dataset
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
import pickle
import pandas as pd, numpy as np
import click
from os.path import join


output_dir = 'output_dir'
cuda=True
hidden_layer_encoder_topology=[100,100]
n_latent = 100
input_db='methylation_array.db'
lr=0.001
weight_decay=0.0001
n_epochs = n_epochs

output_file = join(output_dir,'output_latent.csv')
output_model = join(output_dir,'output_model.p')
outcome_dict_file = join(output_dir,'output_outcomes.p')

conn = sqlite3.connect(input_db)
methyl_array=MethylationArray(*extract_pheno_beta_df_from_sql(conn))
conn.close()


model=TybaltTitusVAE(n_input=methyl_array.return_shape()[1],n_latent=n_latent,hidden_layer_encoder_topology)

optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)

loss_fn = MSELoss()

methyl_dataset = get_methylation_dataset(methylation_array) # train, test split? Add val set?

methyl_dataloader = DataLoader(
    dataset=methyl_dataset,
    num_workers=4,
    batch_size=1,
    shuffle=False)

auto_encoder=AutoEncoder(autoencoder_model=model,n_epochs=n_epochs,loss_fn=loss_fn,optimizer=optimizer,cuda=cuda)
auto_encoder_snapshot = auto_encoder.fit(methyl_dataloader)
latent_projection, sample_names, outcomes = auto_encoder.transform(methyl_dataloader)
outcome_dict=dict(zip(sample_names,outcomes))
pd.DataFrame(latent_projection,index=sample_names).to_csv(output_file)
torch.save(auto_encoder_snapshot,output_model)
pickle.dump(outcome_dict, open(outcome_dict_file,'wb'))
