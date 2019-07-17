from pymethylprocess.MethylationDataTypes import MethylationArray,MethylationArrays, extract_pheno_beta_df_from_pickle_dict
import pickle
import pandas as pd, numpy as np
import click
import os, subprocess
from os.path import join
import time

CONTEXT_SETTINGS = dict(help_option_names=['-h','--help'], max_content_width=90)

@click.group(context_settings= CONTEXT_SETTINGS)
@click.version_option(version='0.1')
def embed():
    pass

def embed_vae(train_pkl,output_dir,cuda,n_latent,lr,weight_decay,n_epochs,hidden_layer_encoder_topology, kl_warm_up=0, beta=1., scheduler='null', decay=0.5, t_max=10, eta_min=1e-6, t_mult=2, bce_loss=False, batch_size=50, val_pkl='val_methyl_array.pkl', n_workers=9, convolutional = False, height_kernel_sizes=[], width_kernel_sizes=[], add_validation_set=False, loss_reduction='sum', stratify_column='disease'):
    from methylnet.models import AutoEncoder, TybaltTitusVAE
    from methylnet.datasets import get_methylation_dataset
    import torch
    from torch.utils.data import DataLoader
    from torch.nn import MSELoss, BCELoss
    os.makedirs(output_dir,exist_ok=True)

    output_file = join(output_dir,'output_latent.csv')
    output_model = join(output_dir,'output_model.p')
    training_curve_file = join(output_dir, 'training_val_curve.p')
    outcome_dict_file = join(output_dir,'output_outcomes.p')
    output_pkl = join(output_dir, 'vae_methyl_arr.pkl')

    #input_dict = pickle.load(open(input_pkl,'rb'))
    #methyl_array=MethylationArray(*extract_pheno_beta_df_from_pickle_dict(input_dict))
    #print(methyl_array.beta)
    train_methyl_array, val_methyl_array = MethylationArray.from_pickle(train_pkl), MethylationArray.from_pickle(val_pkl)#methyl_array.split_train_test(train_p=train_percent, stratified=True, disease_only=True, key='disease', subtype_delimiter=',')

    train_methyl_dataset = get_methylation_dataset(train_methyl_array,stratify_column) # train, test split? Add val set?

    val_methyl_dataset = get_methylation_dataset(val_methyl_array,stratify_column)

    if not batch_size:
        batch_size=len(methyl_dataset)

    train_batch_size = min(batch_size,len(train_methyl_dataset))
    val_batch_size = min(batch_size,len(val_methyl_dataset))

    train_methyl_dataloader = DataLoader(
        dataset=train_methyl_dataset,
        num_workers=n_workers,#n_workers
        batch_size=train_batch_size,
        shuffle=True,
        pin_memory=False)

    val_methyl_dataloader = DataLoader(
        dataset=val_methyl_dataset,
        num_workers=n_workers,
        batch_size=val_batch_size,
        shuffle=True,
        pin_memory=False)

    scaling_factors = dict(train=float(len(train_methyl_dataset))/((len(train_methyl_dataset)//train_batch_size)*train_batch_size),
                           val=float(len(val_methyl_dataset))/((len(val_methyl_dataset)//val_batch_size)*val_batch_size),
                           train_batch_size=train_batch_size,val_batch_size=val_batch_size)
    print('SCALE',len(train_methyl_dataset),len(val_methyl_dataset),train_batch_size,val_batch_size, scaling_factors)
    n_input = train_methyl_array.return_shape()[1]
    if not convolutional:
        model=TybaltTitusVAE(n_input=n_input,n_latent=n_latent,hidden_layer_encoder_topology=hidden_layer_encoder_topology,cuda=cuda)
    else:
        model = CVAE(n_latent=n_latent,in_shape=methyl_dataset.new_shape, kernel_heights=height_kernel_sizes, kernel_widths=width_kernel_sizes, n_pre_latent=n_latent*2) # change soon

    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)
    loss_fn = BCELoss(reduction=loss_reduction) if bce_loss else MSELoss(reduction=loss_reduction) # 'sum'
    scheduler_opts=dict(scheduler=scheduler,lr_scheduler_decay=decay,T_max=t_max,eta_min=eta_min,T_mult=t_mult)
    auto_encoder=AutoEncoder(autoencoder_model=model,n_epochs=n_epochs,loss_fn=loss_fn,optimizer=optimizer,cuda=cuda,kl_warm_up=kl_warm_up,beta=beta, scheduler_opts=scheduler_opts)
    if add_validation_set:
        auto_encoder.add_validation_set(val_methyl_dataloader)
    auto_encoder = auto_encoder.fit(train_methyl_dataloader)
    train_methyl_array=train_methyl_dataset.to_methyl_array()
    val_methyl_array=val_methyl_dataset.to_methyl_array()
    del val_methyl_dataloader, train_methyl_dataloader, val_methyl_dataset, train_methyl_dataset

    methyl_dataset=get_methylation_dataset(MethylationArrays([train_methyl_array,val_methyl_array]).combine(),stratify_column)
    methyl_dataset_loader = DataLoader(
        dataset=methyl_dataset,
        num_workers=n_workers,
        batch_size=1,
        shuffle=False)
    latent_projection, _, _ = auto_encoder.transform(methyl_dataset_loader)
    #print(latent_projection.shape)
    methyl_array = methyl_dataset.to_methyl_array()
    #sample_names = np.array([sample_name[0] for sample_name in sample_names]) # FIXME
    #outcomes = np.array([outcome[0] for outcome in outcomes]) # FIXME
    #outcome_dict=dict(zip(sample_names,outcomes))
    #print(methyl_array.beta)
    latent_projection=pd.DataFrame(latent_projection,index=methyl_array.beta.index)
    methyl_array.beta=latent_projection
    methyl_array.write_pickle(output_pkl)
    latent_projection.to_csv(output_file)
    pickle.dump(auto_encoder.training_plot_data,open(training_curve_file,'wb'))
    torch.save(auto_encoder.model,output_model)
    #pickle.dump(outcome_dict, open(outcome_dict_file,'wb'))
    return latent_projection, None, scaling_factors, n_input, auto_encoder

@embed.command()
@click.option('-i', '--train_pkl', default='./train_val_test_sets/train_methyl_array.pkl', help='Input database for beta and phenotype data.', type=click.Path(exists=False), show_default=True)
@click.option('-o', '--output_dir', default='./embeddings/', help='Output directory for embeddings.', type=click.Path(exists=False), show_default=True)
@click.option('-c', '--cuda', is_flag=True, help='Use GPUs.')
@click.option('-n', '--n_latent', default=64, help='Number of latent dimensions.', show_default=True)
@click.option('-lr', '--learning_rate', default=1e-3, help='Learning rate.', show_default=True)
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
@click.option('-vp', '--val_pkl', default='./train_val_test_sets/val_methyl_array.pkl', help='Validation Set Methylation Array Location.', show_default=True, type=click.Path(exists=False),)
@click.option('-w', '--n_workers', default=9, show_default=True, help='Number of workers.')
@click.option('-conv', '--convolutional', is_flag=True, help='Use convolutional VAE.')
@click.option('-hs', '--height_kernel_sizes', default=[], multiple=True, help='Heights of convolutional kernels.')
@click.option('-ws', '--width_kernel_sizes', default=[], multiple=True, help='Widths of convolutional kernels.')
@click.option('-v', '--add_validation_set', is_flag=True, help='Evaluate validation set.')
@click.option('-l', '--loss_reduction', default='sum', show_default=True, help='Type of reduction on loss function.', type=click.Choice(['sum','elementwise_mean','none']))
@click.option('-hl', '--hyperparameter_log', default='embeddings/embed_hyperparameters_log.csv', show_default=True, help='CSV file containing prior runs.', type=click.Path(exists=False))
@click.option('-sc', '--stratify_column', default='disease', show_default=True, help='Column to stratify samples on.', type=click.Path(exists=False))
@click.option('-j', '--job_name', default='embed_job', show_default=True, help='Embedding job name.', type=click.Path(exists=False))
def perform_embedding(train_pkl,output_dir,cuda,n_latent,learning_rate,weight_decay,n_epochs,hidden_layer_encoder_topology, kl_warm_up, beta, scheduler, decay, t_max, eta_min, t_mult, bce_loss, batch_size, val_pkl, n_workers, convolutional, height_kernel_sizes, width_kernel_sizes, add_validation_set, loss_reduction, hyperparameter_log,stratify_column,job_name):
    """Perform variational autoencoding on methylation dataset."""
    hlt_list=filter(None,hidden_layer_encoder_topology.split(','))
    if hlt_list:
        hidden_layer_encoder_topology=list(map(int,hlt_list))
    else:
        hidden_layer_encoder_topology=[]
    _,_,scaling_factors,n_input,autoencoder = embed_vae(train_pkl,output_dir,cuda,n_latent,learning_rate,weight_decay,n_epochs,hidden_layer_encoder_topology,kl_warm_up,beta, scheduler, decay, t_max, eta_min, t_mult, bce_loss, batch_size, val_pkl, n_workers, convolutional, height_kernel_sizes, width_kernel_sizes, add_validation_set, loss_reduction, stratify_column)
    hyperparameter_row = [job_name,n_epochs, autoencoder.best_epoch, autoencoder.min_loss, autoencoder.min_val_loss, autoencoder.min_val_kl_loss, autoencoder.min_val_recon_loss, autoencoder.min_loss*scaling_factors['train'], autoencoder.min_val_loss*scaling_factors['val'], autoencoder.min_val_kl_loss*scaling_factors['val'], autoencoder.min_val_recon_loss*scaling_factors['val'], autoencoder.min_val_recon_loss+(autoencoder.min_val_kl_loss/beta if beta > 0. else 0.), n_input, n_latent, str(hidden_layer_encoder_topology), learning_rate, weight_decay, beta, kl_warm_up, scheduler, t_max, t_mult, scaling_factors['train_batch_size'], scaling_factors['val_batch_size']]
    hyperparameter_df = pd.DataFrame(columns=['job_name','n_epochs',"best_epoch", "min_loss", "min_val_loss", "min_val_kl_loss", "min_val_recon_loss", "min_loss-batchsize_adj", "min_val_loss-batchsize_adj", "min_val_kl_loss-batchsize_adj", "min_val_recon_loss-batchsize_adj", "min_val_beta-adj_loss", "n_input", "n_latent", "hidden_layer_encoder_topology", "learning_rate", "weight_decay", "beta", "kl_warm_up", "scheduler", "t_max", "t_mult", "train_batch_size",'val_batch_size'])
    hyperparameter_df.loc[0] = hyperparameter_row
    if os.path.exists(hyperparameter_log):
        print('APPEND')
        hyperparameter_df_former = pd.read_csv(hyperparameter_log)
        hyperparameter_df_former=hyperparameter_df_former[[col for col in list(hyperparameter_df) if not col.startswith('Unnamed')]]
        hyperparameter_df=pd.concat([hyperparameter_df_former,hyperparameter_df],axis=0)
    hyperparameter_df.to_csv(hyperparameter_log)

@embed.command()
@click.option('-hcsv', '--hyperparameter_input_csv', default='embeddings/embed_hyperparameters_scan_input.csv', show_default=True, help='CSV file containing hyperparameter inputs.', type=click.Path(exists=False))
@click.option('-hl', '--hyperparameter_output_log', default='embeddings/embed_hyperparameters_log.csv', show_default=True, help='CSV file containing prior runs.', type=click.Path(exists=False))
@click.option('-g', '--generate_input', is_flag=True, help='Generate hyperparameter input csv.')
@click.option('-c', '--job_chunk_size', default=4, help='If not series, chunk up and run these number of commands at once..')
@click.option('-sc', '--stratify_column', default='disease', show_default=True, help='Column to stratify samples on.', type=click.Path(exists=False))
@click.option('-r', '--reset_all', is_flag=True, help='Run all jobs again.')
@click.option('-t', '--torque', is_flag=True, help='Submit jobs on torque.')
@click.option('-gpu', '--gpu', default=-1, help='If torque submit, which gpu to use.', show_default=True)
@click.option('-gn', '--gpu_node', default=-1, help='If torque submit, which gpu node to use.', show_default=True)
@click.option('-nh', '--nohup', is_flag=True, help='Nohup launch jobs.')
@click.option('-mc', '--model_complexity_factor', default=1., help='Degree of neural network model complexity for hyperparameter search. Search for less wide and less deep networks with a lower complexity value, bounded between 0 and infinity.', show_default=True)
@click.option('-b', '--set_beta', default=1., help='Set beta value, bounded between 0 and infinity. Set to -1 ', show_default=True)
@click.option('-j', '--n_jobs', default=4, help='Number of jobs to generate.')
@click.option('-n', '--n_jobs_relaunch', default=0, help='Relaunch n top jobs from previous run.', show_default=True)
@click.option('-cp', '--crossover_p', default=0., help='Rate of crossover between hyperparameters.', show_default=True)
@click.option('-v', '--val_loss_column', default='min_val_loss-batchsize_adj', help='Validation loss column.', type=click.Path(exists=False))
@click.option('-a', '--additional_command', default='', help='Additional command to input for torque run.', type=click.Path(exists=False))
@click.option('-cu', '--cuda', is_flag=True, help='Use GPUs.')
@click.option('-grid', '--hyperparameter_yaml', default='', help='YAML file with custom subset of hyperparameter grid.', type=click.Path(exists=False))
def launch_hyperparameter_scan(hyperparameter_input_csv, hyperparameter_output_log, generate_input, job_chunk_size, stratify_column, reset_all, torque, gpu, gpu_node, nohup, model_complexity_factor, set_beta,n_jobs, n_jobs_relaunch, crossover_p, val_loss_column, additional_command, cuda, hyperparameter_yaml):
    """Launch randomized grid search of neural network hyperparameters."""
    from methylnet.hyperparameter_scans import coarse_scan, find_top_jobs
    custom_jobs=[]
    if n_jobs_relaunch:
        custom_jobs=find_top_jobs(hyperparameter_input_csv, hyperparameter_output_log,n_jobs_relaunch, crossover_p, val_loss_column)
    if os.path.exists(hyperparameter_yaml):
        from ruamel.yaml import safe_load as load
        with open(hyperparameter_yaml) as f:
            new_grid = load(f)
        #print(new_grid)
    else:
        new_grid = {}
    coarse_scan(hyperparameter_input_csv, hyperparameter_output_log, generate_input, job_chunk_size, stratify_column, reset_all, torque, gpu, gpu_node, nohup, mlp=False, model_complexity_factor=model_complexity_factor, set_beta=set_beta,n_jobs=n_jobs, custom_jobs=custom_jobs, additional_command=additional_command, cuda=cuda, new_grid=new_grid)

#################

if __name__ == '__main__':
    embed()
