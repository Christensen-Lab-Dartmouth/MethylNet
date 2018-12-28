from preprocess import MethylationArray, extract_pheno_beta_df_from_pickle_dict
from models import TybaltTitusVAE, CVAE, VAE_MLP, MLPFinetuneVAE
from datasets import get_methylation_dataset
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss, BCELoss, CrossEntropyLoss
import pickle
import pandas as pd, numpy as np
import click
import os
from os.path import join
from collections import Counter

CONTEXT_SETTINGS = dict(help_option_names=['-h','--help'], max_content_width=90)

@click.group(context_settings= CONTEXT_SETTINGS)
@click.version_option(version='0.1')
def prediction():
    pass

def predict(input_pkl,input_vae_pkl,output_dir,cuda,interest_cols,categorical,disease_only,hidden_layer_topology,learning_rate_vae,learning_rate_mlp,weight_decay,n_epochs, scheduler='null', decay=0.5, t_max=10, eta_min=1e-6, t_mult=2, batch_size=50, train_percent=0.8, n_workers=8, add_validation_set=False, loss_reduction='sum'):
    os.makedirs(output_dir,exist_ok=True)

    output_file = join(output_dir,'predictions.csv')
    output_gt_file = join(output_dir,'ground_truth.csv')
    output_file_latent = join(output_dir,'latent.csv')
    output_model = join(output_dir,'output_model.p')
    output_pkl = join(output_dir, 'vae_mlp_methyl_arr.pkl')
    output_onehot_encoder = join(output_dir, 'one_hot_encoder.p')

    input_dict = pickle.load(open(input_pkl,'rb'))
    vae_model = torch.load(input_vae_pkl)

    methyl_array=MethylationArray(*extract_pheno_beta_df_from_pickle_dict(input_dict))

    train_methyl_array, test_methyl_array = methyl_array.split_train_test(train_p=train_percent, stratified=(True if categorical else False), disease_only=disease_only, key=interest_cols[0], subtype_delimiter=',')

    if len(interest_cols) == 1 and disease_only:
        print(interest_cols)
        interest_cols[0] += '_only'
        print(train_methyl_array.pheno[interest_cols[0]].unique())
        print(test_methyl_array.pheno[interest_cols[0]].unique())

    train_methyl_dataset = get_methylation_dataset(train_methyl_array,interest_cols,categorical=categorical, predict=True) # train, test split? Add val set?
    print(list(train_methyl_dataset.encoder.get_feature_names()))
    test_methyl_dataset = get_methylation_dataset(test_methyl_array,interest_cols,categorical=categorical, predict=True, categorical_encoder=train_methyl_dataset.encoder)

    if not batch_size:
        batch_size=len(methyl_dataset)

    train_methyl_dataloader = DataLoader(
        dataset=train_methyl_dataset,
        num_workers=n_workers,
        batch_size=batch_size,
        shuffle=True)


    test_methyl_dataloader = DataLoader(
        dataset=test_methyl_dataset,
        num_workers=n_workers,
        batch_size=min(batch_size,len(test_methyl_dataset)),
        shuffle=False)

    model=VAE_MLP(vae_model=vae_model,categorical=categorical,hidden_layer_topology=hidden_layer_topology,n_output=train_methyl_dataset.outcome_col.shape[1])

    class_weights=[]
    if categorical:
        out_weight=Counter(np.argmax(train_methyl_dataset.outcome_col,axis=1))
        #total_samples=sum(out_weight.values())
        for k in sorted(list(out_weight.keys())):
            class_weights.append(1./float(out_weight[k])) # total_samples
        class_weights=np.array(class_weights)
        class_weights=(class_weights/class_weights.sum()).tolist()
        print(class_weights)

    if class_weights:
        class_weights = torch.FloatTensor(class_weights)
        if cuda:
            class_weights = class_weights.cuda()
    else:
        class_weights = None

    optimizer_vae = torch.optim.Adam(model.vae.parameters(), lr = learning_rate_vae, weight_decay=weight_decay)
    optimizer_mlp = torch.optim.Adam(model.mlp.parameters(), lr = learning_rate_mlp, weight_decay=weight_decay)
    loss_fn = CrossEntropyLoss(reduction=loss_reduction,weight= class_weights) if categorical else MSELoss(reduction=loss_reduction) # 'sum'
    scheduler_opts=dict(scheduler=scheduler,lr_scheduler_decay=decay,T_max=t_max,eta_min=eta_min,T_mult=t_mult)
    vae_mlp=MLPFinetuneVAE(mlp_model=model,n_epochs=n_epochs,categorical=categorical,loss_fn=loss_fn,optimizer_vae=optimizer_vae,optimizer_mlp=optimizer_mlp,cuda=cuda, scheduler_opts=scheduler_opts)
    if add_validation_set:
        vae_mlp.add_validation_set(test_methyl_dataloader)
    vae_mlp_snapshot = vae_mlp.fit(train_methyl_dataloader).model
    train_encoder = train_methyl_dataset.encoder
    del train_methyl_dataloader, train_methyl_dataset
    """methyl_dataset=get_methylation_dataset(methyl_array,interest_cols,predict=True)
    methyl_dataset_loader = DataLoader(
        dataset=methyl_dataset,
        num_workers=9,
        batch_size=1,
        shuffle=False)"""
    Y_pred, Y_true, latent_projection, sample_names = vae_mlp.predict(test_methyl_dataloader)
    test_methyl_array = test_methyl_dataset.to_methyl_array()
    """if categorical:
        Y_true=test_methyl_dataset.encoder.inverse_transform(Y_true)[:,np.newaxis]
        Y_pred=test_methyl_dataset.encoder.inverse_transform(Y_pred)[:,np.newaxis]"""
    #sample_names = np.array([sample_name[0] for sample_name in sample_names]) # FIXME
    #outcomes = np.array([outcome[0] for outcome in outcomes]) # FIXME
    Y_pred=pd.DataFrame(Y_pred,index=test_methyl_array.beta.index)#dict(zip(sample_names,outcomes))
    Y_true=pd.DataFrame(Y_true,index=test_methyl_array.beta.index)
    print(methyl_array.beta)
    latent_projection=pd.DataFrame(latent_projection,index=test_methyl_array.beta.index)
    test_methyl_array.beta=latent_projection
    test_methyl_array.write_pickle(output_pkl)
    latent_projection.to_csv(output_file_latent)
    torch.save(vae_mlp_snapshot,output_model)
    Y_pred.to_csv(output_file)#pickle.dump(outcome_dict, open(outcome_dict_file,'wb'))
    Y_true.to_csv(output_gt_file)
    pickle.dump(train_encoder,open(output_onehot_encoder,'wb'))
    return latent_projection, Y_pred, Y_true, vae_mlp_snapshot

@prediction.command() # FIXME finish this!!
@click.option('-i', '--input_pkl', default='./final_preprocessed/methyl_array.pkl', help='Input database for beta and phenotype data.', type=click.Path(exists=False), show_default=True)
@click.option('-vae', '--input_vae_pkl', default='./embeddings/output_model.p', help='Trained VAE.', type=click.Path(exists=False), show_default=True)
@click.option('-o', '--output_dir', default='./predictions/', help='Output directory for predictions.', type=click.Path(exists=False), show_default=True)
@click.option('-c', '--cuda', is_flag=True, help='Use GPUs.')
@click.option('-ic', '--interest_cols', default=['disease'], multiple=True, help='Specify columns looking to make predictions on.', show_default=True)
@click.option('-cat', '--categorical', is_flag=True, help='Multi-class prediction.', show_default=True)
@click.option('-do', '--disease_only', is_flag=True, help='Only look at disease, or text before subtype_delimiter.')
@click.option('-hlt', '--hidden_layer_topology', default='', help='Topology of hidden layers, comma delimited, leave empty for one layer encoder, eg. 100,100 is example of 5-hidden layer topology.', type=click.Path(exists=False), show_default=True)
@click.option('-lr_vae', '--learning_rate_vae', default=1e-5, help='Learning rate VAE.', show_default=True)
@click.option('-lr_mlp', '--learning_rate_mlp', default=1e-3, help='Learning rate MLP.', show_default=True)
@click.option('-wd', '--weight_decay', default=1e-4, help='Weight decay of adam optimizer.', show_default=True)
@click.option('-e', '--n_epochs', default=50, help='Number of epochs to train over.', show_default=True)
@click.option('-s', '--scheduler', default='null', help='Type of learning rate scheduler.', type=click.Choice(['null','exp','warm_restarts']),show_default=True)
@click.option('-d', '--decay', default=0.5, help='Learning rate scheduler decay for exp selection.', show_default=True)
@click.option('-t', '--t_max', default=10, help='Number of epochs before cosine learning rate restart.', show_default=True)
@click.option('-eta', '--eta_min', default=1e-6, help='Minimum cosine LR.', show_default=True)
@click.option('-m', '--t_mult', default=2., help='Multiply current restart period times this number given number of restarts.', show_default=True)
@click.option('-bs', '--batch_size', default=50, show_default=True, help='Batch size.')
@click.option('-p', '--train_percent', default=0.8, help='Percent data training on.', show_default=True)
@click.option('-w', '--n_workers', default=9, show_default=True, help='Number of workers.')
@click.option('-v', '--add_validation_set', is_flag=True, help='Evaluate validation set.')
@click.option('-l', '--loss_reduction', default='sum', show_default=True, help='Type of reduction on loss function.', type=click.Choice(['sum','elementwise_mean','none']))
def make_prediction(input_pkl,input_vae_pkl,output_dir,cuda,interest_cols,categorical,disease_only,hidden_layer_topology,learning_rate_vae,learning_rate_mlp,weight_decay,n_epochs, scheduler='null', decay=0.5, t_max=10, eta_min=1e-6, t_mult=2, batch_size=50, train_percent=0.8, n_workers=8, add_validation_set=False, loss_reduction='sum'):
    """Perform variational autoencoding on methylation dataset."""
    hlt_list=filter(None,hidden_layer_topology.split(','))
    if hlt_list:
        hidden_layer_topology=list(map(int,hlt_list))
    else:
        hidden_layer_topology=[]
    predict(input_pkl,input_vae_pkl,output_dir,cuda,list(interest_cols),categorical,disease_only,hidden_layer_topology,learning_rate_vae,learning_rate_mlp,weight_decay,n_epochs, scheduler, decay, t_max, eta_min, t_mult, batch_size, train_percent, n_workers, add_validation_set, loss_reduction)

#################

if __name__ == '__main__':
    prediction()
