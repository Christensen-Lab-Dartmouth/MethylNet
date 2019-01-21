from MethylationDataTypes import MethylationArray, extract_pheno_beta_df_from_pickle_dict
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

def embed_vae(input_pkl,output_dir,cuda,n_latent,lr,weight_decay,n_epochs,hidden_layer_encoder_topology, kl_warm_up=0, beta=1., scheduler='null', decay=0.5, t_max=10, eta_min=1e-6, t_mult=2, bce_loss=False, batch_size=50, train_percent=0.8, n_workers=9, convolutional = False, height_kernel_sizes=[], width_kernel_sizes=[], add_validation_set=False, loss_reduction='sum', stratify_column='disease'):
    from models import AutoEncoder, TybaltTitusVAE, CVAE
    from datasets import get_methylation_dataset
    import torch
    from torch.utils.data import DataLoader
    from torch.nn import MSELoss, BCELoss
    os.makedirs(output_dir,exist_ok=True)

    output_file = join(output_dir,'output_latent.csv')
    output_model = join(output_dir,'output_model.p')
    outcome_dict_file = join(output_dir,'output_outcomes.p')
    output_pkl = join(output_dir, 'vae_methyl_arr.pkl')
    train_test_idx_file = join(output_dir, 'train_test_idx.p')

    input_dict = pickle.load(open(input_pkl,'rb'))
    methyl_array=MethylationArray(*extract_pheno_beta_df_from_pickle_dict(input_dict))
    #print(methyl_array.beta)
    train_methyl_array, test_methyl_array = methyl_array.split_train_test(train_p=train_percent, stratified=True, disease_only=True, key='disease', subtype_delimiter=',')

    train_test_idx_dict={}
    train_test_idx_dict['train'], train_test_idx_dict['test'] = train_methyl_array.return_idx(), test_methyl_array.return_idx()

    train_methyl_dataset = get_methylation_dataset(train_methyl_array,stratify_column) # train, test split? Add val set?

    test_methyl_dataset = get_methylation_dataset(test_methyl_array,stratify_column)

    if not batch_size:
        batch_size=len(methyl_dataset)

    train_methyl_dataloader = DataLoader(
        dataset=train_methyl_dataset,
        num_workers=n_workers,#n_workers
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False)

    test_methyl_dataloader = DataLoader(
        dataset=test_methyl_dataset,
        num_workers=n_workers,
        batch_size=min(batch_size,len(test_methyl_dataset)),
        shuffle=True,
        pin_memory=False)

    n_input = methyl_array.return_shape()[1]
    if not convolutional:
        model=TybaltTitusVAE(n_input=n_input,n_latent=n_latent,hidden_layer_encoder_topology=hidden_layer_encoder_topology,cuda=cuda)
    else:
        model = CVAE(n_latent=n_latent,in_shape=methyl_dataset.new_shape, kernel_heights=height_kernel_sizes, kernel_widths=width_kernel_sizes, n_pre_latent=n_latent*2) # change soon

    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)
    loss_fn = BCELoss(reduction=loss_reduction) if bce_loss else MSELoss(reduction=loss_reduction) # 'sum'
    scheduler_opts=dict(scheduler=scheduler,lr_scheduler_decay=decay,T_max=t_max,eta_min=eta_min,T_mult=t_mult)
    auto_encoder=AutoEncoder(autoencoder_model=model,n_epochs=n_epochs,loss_fn=loss_fn,optimizer=optimizer,cuda=cuda,kl_warm_up=kl_warm_up,beta=beta, scheduler_opts=scheduler_opts)
    if add_validation_set:
        auto_encoder.add_validation_set(test_methyl_dataloader)
    auto_encoder = auto_encoder.fit(train_methyl_dataloader)
    del test_methyl_dataloader, train_methyl_dataloader, test_methyl_dataset, train_methyl_dataset
    methyl_dataset=get_methylation_dataset(methyl_array,stratify_column)
    methyl_dataset_loader = DataLoader(
        dataset=methyl_dataset,
        num_workers=n_workers,
        batch_size=1,
        shuffle=False)
    latent_projection, sample_names, outcomes = auto_encoder.transform(methyl_dataset_loader)
    #print(latent_projection.shape)
    methyl_array = methyl_dataset.to_methyl_array()
    #sample_names = np.array([sample_name[0] for sample_name in sample_names]) # FIXME
    #outcomes = np.array([outcome[0] for outcome in outcomes]) # FIXME
    outcome_dict=dict(zip(sample_names,outcomes))
    #print(methyl_array.beta)
    latent_projection=pd.DataFrame(latent_projection,index=methyl_array.beta.index)
    methyl_array.beta=latent_projection
    methyl_array.write_pickle(output_pkl)
    latent_projection.to_csv(output_file)
    torch.save(auto_encoder.model,output_model)
    pickle.dump(outcome_dict, open(outcome_dict_file,'wb'))
    pickle.dump(train_test_idx_dict,open(train_test_idx_file,'wb'))
    return latent_projection, outcome_dict, n_input, auto_encoder

@embed.command()
@click.option('-i', '--input_pkl', default='./final_preprocessed/methyl_array.pkl', help='Input database for beta and phenotype data.', type=click.Path(exists=False), show_default=True)
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
@click.option('-p', '--train_percent', default=0.8, help='Percent data training on.', show_default=True)
@click.option('-w', '--n_workers', default=9, show_default=True, help='Number of workers.')
@click.option('-conv', '--convolutional', is_flag=True, help='Use convolutional VAE.')
@click.option('-hs', '--height_kernel_sizes', default=[], multiple=True, help='Heights of convolutional kernels.')
@click.option('-ws', '--width_kernel_sizes', default=[], multiple=True, help='Widths of convolutional kernels.')
@click.option('-v', '--add_validation_set', is_flag=True, help='Evaluate validation set.')
@click.option('-l', '--loss_reduction', default='sum', show_default=True, help='Type of reduction on loss function.', type=click.Choice(['sum','elementwise_mean','none']))
@click.option('-hl', '--hyperparameter_log', default='embeddings/embed_hyperparameters_log.csv', show_default=True, help='CSV file containing prior runs.', type=click.Path(exists=False))
@click.option('-sc', '--stratify_column', default='disease', show_default=True, help='Column to stratify samples on.', type=click.Path(exists=False))
def perform_embedding(input_pkl,output_dir,cuda,n_latent,learning_rate,weight_decay,n_epochs,hidden_layer_encoder_topology, kl_warm_up, beta, scheduler, decay, t_max, eta_min, t_mult, bce_loss, batch_size, train_percent, n_workers, convolutional, height_kernel_sizes, width_kernel_sizes, add_validation_set, loss_reduction, hyperparameter_log,stratify_column):
    """Perform variational autoencoding on methylation dataset."""
    hlt_list=filter(None,hidden_layer_encoder_topology.split(','))
    if hlt_list:
        hidden_layer_encoder_topology=list(map(int,hlt_list))
    else:
        hidden_layer_encoder_topology=[]
    _,_,n_input,autoencoder = embed_vae(input_pkl,output_dir,cuda,n_latent,learning_rate,weight_decay,n_epochs,hidden_layer_encoder_topology,kl_warm_up,beta, scheduler, decay, t_max, eta_min, t_mult, bce_loss, batch_size, train_percent, n_workers, convolutional, height_kernel_sizes, width_kernel_sizes, add_validation_set, loss_reduction, stratify_column)
    hyperparameter_row = [n_epochs, autoencoder.best_epoch, autoencoder.min_loss, autoencoder.min_val_loss, n_input, n_latent, str(hidden_layer_encoder_topology), learning_rate, weight_decay, beta, kl_warm_up, scheduler, t_max, t_mult, batch_size, train_percent]
    if os.path.exists(hyperparameter_log):
        print('APPEND')
        hyperparameter_df = pd.read_csv(hyperparameter_log)
        hyperparameter_df=hyperparameter_df[[col for col in list(hyperparameter_df) if not col.startswith('Unnamed')]]
        hyperparameter_df.append(hyperparameter_row)
    else:
        print('CREATE')
        hyperparameter_df = pd.DataFrame(columns=['n_epochs',"best_epoch", "min_loss", "min_val_loss", "n_input", "n_latent", "hidden_layer_encoder_topology", "learning_rate", "weight_decay", "beta", "kl_warm_up", "scheduler", "t_max", "t_mult", "batch_size", "train_percent"])
        hyperparameter_df.loc[0] = hyperparameter_row
    hyperparameter_df.to_csv(hyperparameter_log)

@embed.command()
@click.option('-hcsv', '--hyperparameter_input_csv', default='embeddings/embed_hyperparameters_scan_input.csv', show_default=True, help='CSV file containing hyperparameter inputs.', type=click.Path(exists=False))
@click.option('-hl', '--hyperparameter_output_log', default='embeddings/embed_hyperparameters_log.csv', show_default=True, help='CSV file containing prior runs.', type=click.Path(exists=False))
@click.option('-g', '--generate_input', is_flag=True, help='Generate hyperparameter input csv.')
@click.option('-c', '--job_chunk_size', default=4, help='If not series, chunk up and run these number of commands at once..')
@click.option('-sc', '--stratify_column', default='disease', show_default=True, help='Column to stratify samples on.', type=click.Path(exists=False))
@click.option('-r', '--reset_all', is_flag=True, help='Run all jobs again.')
@click.option('-t', '--torque', is_flag=True, help='Submit jobs on torque.')
@click.option('-gpu', '--gpu', default=0, help='If torque submit, which gpu to use.', show_default=True)
@click.option('-gn', '--gpu_node', default=1, help='If torque submit, which gpu node to use.', show_default=True)
@click.option('-nh', '--nohup', is_flag=True, help='Nohup launch jobs.')
def launch_hyperparameter_scan(hyperparameter_input_csv, hyperparameter_output_log, generate_input, job_chunk_size, stratify_column, reset_all, torque, gpu, gpu_node, nohup):
    if generate_input:
        df=pd.DataFrame(np.array([False,300,1e-3,1e-4,50,'300,300,300',0,200,'warm_restarts',10,1e-7,1.,512,0.85,4,'sum'])[np.newaxis,:],columns=['completed','--n_latent','--learning_rate','--weight_decay','--n_epochs','--hidden_layer_encoder_topology', '--kl_warm_up', '--beta', '--scheduler', '--t_max', '--eta_min', '--t_mult', '--batch_size', '--train_percent', '--n_workers', '--loss_reduction'])
        df.to_csv(hyperparameter_input_csv)
    else:
        from itertools import cycle
        from pathos.multiprocessing import ProcessingPool as Pool
        def run(x):
            print(x)
            subprocess.call(x,shell=True)
        lower = lambda x: x.lower()
        if not torque:
            gpus=cycle(range(4))
        else:
            gpus=cycle([gpu])
        df=pd.read_csv(hyperparameter_input_csv)
        if reset_all:
            df.loc[:,'completed']=False
        df_final = df[df['completed'].astype(str).map(lower)=='false'].reset_index(drop=True)[['--n_latent','--learning_rate','--weight_decay','--n_epochs','--hidden_layer_encoder_topology', '--kl_warm_up', '--beta', '--scheduler', '--t_max', '--eta_min', '--t_mult', '--batch_size', '--train_percent', '--n_workers', '--loss_reduction']]
        commands=[]
        for i in range(df_final.shape[0]):
            commands.append('python embedding.py perform_embedding -bce -c -v -hl {} -sc {} {}'.format(hyperparameter_output_log,stratify_column,' '.join(['{} {}'.format(k,v) for k,v in [(k2,df_final.loc[i,k2]) for k2 in list(df_final)]])))
        for i in range(len(commands)):
            commands[i] = '{} {} {} {}'.format('CUDA_VISIBLE_DEVICES="{}"'.format(next(gpus)),'nohup' if nohup and not torque else '',commands[i],'&' if nohup and not torque else '')
        if torque:
            with open('pbs_embed_hyperparameters.sh','r') as f:
                deploy_txt=f.read().replace('COMMAND',commands[0]).replace('HOST','g0{}'.format(gpu_node))
            with open('pbs_gpu_deploy.sh','w') as f:
                f.write(deploy_txt)
            command = 'mksub pbs_gpu_deploy.sh'
            print(command)
            job=os.popen(command).read().strip('\n')
            df.loc[np.arange(df.shape[0])==np.where(df['completed'].astype(str).map(lower)=='false')[0][0],'completed']=job
        else:
            commands = np.array_split(commands,len(commands)//job_chunk_size)
            for command_list in commands:
                if nohup:
                    for command in command_list:
                        subprocess.call(command,shell=True)
                else:
                    pool = Pool(len(command_list))
                    pool.map(run, command_list)
                    pool.close()
                    pool.join()
            df.loc[:,'completed']=True
        df[[col for col in list(df) if not col.startswith('Unnamed')]].to_csv(hyperparameter_input_csv)

#################

if __name__ == '__main__':
    embed()
