"""
hyperparameter_scans.py
=======================
Run randomized grid search to find ideal model hyperparameters, with possible deployments to batch system for scalability.
"""

import os, pandas as pd, numpy as np, subprocess
import time
from methylnet.torque_jobs import assemble_run_torque

def find_top_jobs(hyperparameter_input_csv,hyperparameter_output_log, n_top_jobs, crossover_p=0, val_loss_column='min_val_loss'):
    """Finds top performing jobs from hyper parameter scan to rerun and cross-over parameters.

    Parameters
    ----------
    hyperparameter_input_csv : str
        CSV file containing hyperparameter inputs.
    hyperparameter_output_log : str
        CSV file containing prior runs.
    n_top_jobs : int
        Number of top jobs to select
    crossover_p : float
        Rate of cross over of reused hyperparameters
    val_loss_column : str
        Loss column used to select top jobs

    Returns
    -------
    list
        List of list of new parameters for jobs to run.

    """

    custom_jobs = pd.read_csv(hyperparameter_input_csv)
    hyperparam_output = pd.read_csv(hyperparameter_output_log)[['job_name',val_loss_column]]
    best_outputs = hyperparam_output.sort_values(val_loss_column,ascending=True).iloc[:n_top_jobs,:]
    custom_jobs = custom_jobs[np.isin(custom_jobs['--job_name'].values,best_outputs['job_name'].values)]
    if custom_jobs.shape[0]==0 and best_outputs.shape[0]>0:
        custom_jobs = best_outputs.rename(columns={'--{}'.format(k):k for k in list(best_outputs)})[list(custom_jobs)]
        custom_jobs.loc[:,'--hidden_layer_topology']=custom_jobs.loc[:,'--hidden_layer_topology'].map(lambda x: x.replace('[','').replace(']',''))
    custom_jobs.loc[:,'--job_name']='False'
    if crossover_p:
        for j in range(1,custom_jobs.shape[1]):
            vals=custom_jobs.iloc[:,j].unique()
            for i in range(custom_jobs.shape[0]):
                if np.random.rand() <= crossover_p:
                    custom_jobs.iloc[i,j]=np.random.choice(vals)
    return [custom_jobs]

def replace_grid(old_grid, new_grid, topology_grid):
    """Replaces old hyperparameter search grid with one supplied by YAML."""
    #print(old_grid,new_grid)
    for k in new_grid.keys():
        if k != 'topology_grid':
            old_grid["--{}".format(k)] = new_grid[k]
    if 'topology_grid' in new_grid:
        topology_grid = new_grid.pop('topology_grid')
    return old_grid, topology_grid

def generate_topology(topology_grid, probability_decay_factor=0.9):
    """Generates list denoting neural network topology, list of hidden layer sizes.

    Parameters
    ----------
    topology_grid : list
        List of different hidden layer sizes (number neurons) to choose from.
    probability_decay_factor : float
        Degree of neural network model complexity for hyperparameter search. Search for less wide networks with a lower complexity value, bounded between 0 and infinity.

    Returns
    -------
    list
        List of hidden layer sizes.

    """

    probability_decay_factor=float(max(probability_decay_factor,0.))
    p = probability_decay_factor**np.arange(len(topology_grid))
    p /= sum(p)
    n_layer_attempts = int(round(3*probability_decay_factor))
    if n_layer_attempts:
        topology=list(filter(lambda x: x,np.random.choice(topology_grid,n_layer_attempts,p=p)))
    else:
        topology=[]
    if topology:
        return ','.join(map(str,topology))
    else:
        return ''
    return ''

def coarse_scan(hyperparameter_input_csv, hyperparameter_output_log, generate_input, job_chunk_size, stratify_column, reset_all, torque, gpu, gpu_node, nohup, mlp=False, custom_jobs=[], model_complexity_factor=0.9, set_beta=-1., n_jobs=4, categorical=True, add_softmax=False, additional_command = "", cuda=True, new_grid = {}):
    """Perform randomized hyperparameter grid search

    Parameters
    ----------
    hyperparameter_input_csv : type
        CSV file containing hyperparameter inputs.
    hyperparameter_output_log : type
        CSV file containing prior runs.
    generate_input : type
        Generate hyperparameter input csv.
    job_chunk_size : type
        Number of jobs to be launched at same time.
    stratify_column : type
        Performing classification?
    reset_all : type
        Rerun all jobs previously scanned.
    torque : type
        Run jobs using torque.
    gpu : type
        What GPU to use, set to -1 to be agnostic to GPU selection.
    gpu_node : type
        What GPU to use, set to -1 to be agnostic to GPU selection, for torque submission.
    nohup : type
        Launch jobs using nohup.
    mlp : type
        If running prediction job (classification/regression) after VAE.
    custom_jobs : type
        Supply custom job parameters to be run.
    model_complexity_factor : type
        Degree of neural network model complexity for hyperparameter search. Search for less wide networks with a lower complexity value, bounded between 0 and infinity.
    set_beta : type
        Don't hyperparameter scan over beta (KL divergence weight), and set it to value.
    n_jobs : type
        Number of jobs to generate.
    categorical : type
        Classification task?
    add_softmax : type
        Add softmax layer at end of neural network.
    cuda : type
        Whether to use GPU?
    """
    from itertools import cycle
    from pathos.multiprocessing import ProcessingPool as Pool
    os.makedirs(hyperparameter_input_csv[:hyperparameter_input_csv.rfind('/')],exist_ok=True)
    generated_input=[]
    np.random.seed(int(time.time()))
    if mlp:
        grid={'--learning_rate_vae':[1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1],'--learning_rate_mlp':[1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1,5e-1],
              '--weight_decay':[1e-4],'--n_epochs':[25,50,75,100,200,500,700], '--scheduler':['warm_restarts','null'], '--t_max':[10],
              '--eta_min':[1e-7,1e-6], '--t_mult':[1.,1.2,1.5,2],
              '--batch_size':[50,100,256,512], '--dropout_p':[0.,0.1,0.2,0.3,0.5],
              '--n_workers':[4], '--loss_reduction':['sum']}
        topology_grid = [0,100,200,300,500,1000,2000,3000,4096]
    else:
        grid={'--n_latent':[100,150,200,300,500],
              '--learning_rate':[5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1],
              '--weight_decay':[1e-4],'--n_epochs':[25,50,75,100,200,500,700],
              '--kl_warm_up':[0,20], '--beta':[0.,0.5,1,10,50,100,200,500] if set_beta == -1. else [set_beta],
              '--scheduler':['warm_restarts','null'], '--t_max':[10],
              '--eta_min':[1e-7,1e-6], '--t_mult':[1.,1.2,1.5,2],
              '--batch_size':[50,100,256,512],
              '--n_workers':[4], '--loss_reduction':['sum']}
        topology_grid=[0,100,200,300,500,1000,2000]
    if new_grid:
        grid,topology_grid=replace_grid(grid,new_grid,topology_grid)
    grid['--hidden_layer_topology' if mlp else '--hidden_layer_encoder_topology']=[generate_topology(topology_grid,probability_decay_factor=model_complexity_factor) for i in range(40)]
    if generate_input:
        for i in range(n_jobs):
            generated_input.append(['False']+[np.random.choice(grid[k]) for k in grid])
        generated_input=[pd.DataFrame(generated_input,columns=['--job_name']+list(grid.keys()))]
    if custom_jobs:
        custom_jobs[0].loc[:,'--job_name']='False'
        generated_input=custom_jobs
    def run(x):
        print(x)
        subprocess.call(x,shell=True)
    lower = lambda x: x.lower()
    if gpu == -1:
        gpus=cycle(range(4))
    else:
        gpus=cycle([gpu])
    if os.path.exists(hyperparameter_input_csv):
        df=pd.read_csv(hyperparameter_input_csv)
        df=[df[[col for col in list(df) if not col.startswith('Unnamed')]]]
    else:
        df = []
    df=pd.concat(df+generated_input,axis=0)[['--job_name']+list(grid.keys())].fillna('')
    print(df)
    if reset_all:
        df.loc[:,'--job_name']='False'
    df_final = df[df['--job_name'].astype(str).map(lower)=='false'].reset_index(drop=True)[list(grid.keys())]
    commands=[]
    for i in range(df_final.shape[0]):
        job_id = str(np.random.randint(0,100000000))
        if not mlp:
            commands.append('sh -c "methylnet-embed perform_embedding -bce {} -v -j {} -hl {} -sc {} {} && pymethyl-visualize transform_plot -i embeddings/vae_methyl_arr.pkl -o visualizations/{}_vae_embed.html -c {} -nn 10 "'.format("-c" if cuda else "",job_id,hyperparameter_output_log,stratify_column,' '.join(['{} {}'.format(k2,df_final.loc[i,k2]) for k2 in list(df_final) if (df_final.loc[i,k2] != '' and df_final.loc[i,k2] != np.nan)]),job_id,stratify_column))
        else:
            commands.append('sh -c "methylnet-predict make_prediction {} {} {} {} -v {} -j {} -hl {} {} && {}"'.format("-c" if cuda else "",'-sft' if add_softmax else '','-cat' if categorical else '',''.join([' -ic {}'.format(col) for col in stratify_column]),'-do' if stratify_column[0]=='disease_only' else '',job_id,hyperparameter_output_log,' '.join(['{} {}'.format(k2,df_final.loc[i,k2]) for k2 in list(df_final) if (df_final.loc[i,k2] != '' and df_final.loc[i,k2] != np.nan)]),
                                '&&'.join([" pymethyl-visualize transform_plot -i predictions/vae_mlp_methyl_arr.pkl -o visualizations/{}_{}_mlp_embed.html -c {} -nn 8 ".format(job_id,col,col) for col in stratify_column]))) #-do
        df.loc[np.arange(df.shape[0])==np.where(df['--job_name'].astype(str).map(lower)=='false')[0][0],'--job_name']=job_id
    for i in range(len(commands)):
        commands[i] = '{} {} {} {}'.format('CUDA_VISIBLE_DEVICES="{}"'.format(next(gpus)) if not torque else "",'nohup' if nohup else '',commands[i],'&' if nohup else '') # $gpuNum
    if torque:
        for command in commands:
            job = assemble_run_torque(command, use_gpu=cuda, additions=additional_command, queue='gpuq' if cuda else "normal", time=4, ngpu=1, additional_options='' if gpu_node == -1 else ' -l hostlist=g0{}'.format(gpu_node))
    else:
        if len(commands) == 1:
            subprocess.call(commands[0],shell=True)
        else:
            commands = np.array_split(commands,len(commands)//job_chunk_size)
            for command_list in commands:
                if nohup:
                    for command in command_list:
                        print(command)
                        subprocess.call(command,shell=True)
                else:
                    for command in command_list:
                        subprocess.call(command,shell=True)
                    """pool = Pool(len(command_list))
                    pool.map(run, command_list)
                    pool.close()
                    pool.join()"""
    df.to_csv(hyperparameter_input_csv)
