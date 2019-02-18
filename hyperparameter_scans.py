import os, pandas as pd, numpy as np, subprocess
import time

def find_top_jobs(hyperparameter_input_csv,hyperparameter_output_log, n_top_jobs, crossover_p=0, val_loss_column='min_val_loss'):
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

def generate_topology(topology_grid, probability_decay_factor=0.9):
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

def coarse_scan(hyperparameter_input_csv, hyperparameter_output_log, generate_input, job_chunk_size, stratify_column, reset_all, torque, gpu, gpu_node, nohup, mlp=False, custom_jobs=[], model_complexity_factor=0.9, set_beta=-1., n_jobs=4, categorical=True):
    from itertools import cycle
    from pathos.multiprocessing import ProcessingPool as Pool
    os.makedirs(hyperparameter_input_csv[:hyperparameter_input_csv.rfind('/')],exist_ok=True)
    generated_input=[]
    np.random.seed(int(time.time()))
    if mlp:
        grid={#'--hidden_layer_topology':['', '100', '100,100', '100,200', '100,4096,300', '300,500,400', '200', '200,200', '300', '200,300,200', '500,300', '500,100', '1000,1000', '1000,2000,1000', '500,1000,500', '1000,3000,1000'],
              '--learning_rate_vae':[1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1],'--learning_rate_mlp':[1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1,5e-1],
              '--weight_decay':[1e-4],'--n_epochs':[25,50,75,100,200,500,700], '--scheduler':['warm_restarts','null'], '--t_max':[10],
              '--eta_min':[1e-7,1e-6], '--t_mult':[1.,1.2,1.5,2],
              '--batch_size':[50,100,256,512], '--dropout_p':[0.,0.1,0.2,0.3,0.5],
              '--n_workers':[4], '--loss_reduction':['sum']}
        topology_grid = [0,100,200,300,500,1000,2000,3000,4096]
    else:
        grid={'--n_latent':[100,150,200,300,500],
              '--learning_rate':[5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1],
              '--weight_decay':[1e-4],'--n_epochs':[25,50,75,100,200,500,700],
              #'--hidden_layer_encoder_topology':['','100','100,100','200','200,200','300','300,300','500','500,500','500,300','500,100','300,200'],
              '--kl_warm_up':[0,20], '--beta':[0.,0.5,1,10,50,100,200,500] if set_beta == -1. else [set_beta],
              '--scheduler':['warm_restarts','null'], '--t_max':[10],
              '--eta_min':[1e-7,1e-6], '--t_mult':[1.,1.2,1.5,2],
              '--batch_size':[50,100,256,512],
              '--n_workers':[4], '--loss_reduction':['sum']}
        topology_grid=[0,100,200,300,500,1000,2000]
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
            commands.append('sh -c "time python embedding.py perform_embedding -bce -c -v -j {} -hl {} -sc {} {} && python visualizations.py transform_plot -i embeddings/vae_methyl_arr.pkl -o visualizations/{}_vae_embed.html -c {} -nn 10 "'.format(job_id,hyperparameter_output_log,stratify_column,' '.join(['{} {}'.format(k2,df_final.loc[i,k2]) for k2 in list(df_final) if (df_final.loc[i,k2] != '' and df_final.loc[i,k2] != np.nan)]),job_id,stratify_column))
        else:
            commands.append('sh -c "time python predictions.py make_prediction {} {} -c -v {} -j {} -hl {} {} && {}"'.format('-cat' if categorical else '',''.join(['-ic {}'.format(col) for col in stratify_column]),'-do' if stratify_column[0]=='disease_only' else '',job_id,hyperparameter_output_log,' '.join(['{} {}'.format(k2,df_final.loc[i,k2]) for k2 in list(df_final) if (df_final.loc[i,k2] != '' and df_final.loc[i,k2] != np.nan)]),
                                '&&'.join([" python visualizations.py transform_plot -i predictions/vae_mlp_methyl_arr.pkl -o visualizations/{}_{}_mlp_embed.html -c {} -nn 8 ".format(job_id,col,col) for col in stratify_column]))) #-do
        df.loc[np.arange(df.shape[0])==np.where(df['--job_name'].astype(str).map(lower)=='false')[0][0],'--job_name']=job_id
    for i in range(len(commands)):
        commands[i] = '{} {} {} {}'.format('CUDA_VISIBLE_DEVICES="{}"'.format(next(gpus) if not torque else "$gpuNum"),'nohup' if nohup else '',commands[i],'&' if nohup else '')
    if torque:
        for command in commands:
            with open('pbs_embed_hyperparameters.sh','r') as f:
                deploy_txt=f.read().replace('COMMAND',command)#.replace('HOST','g0{}'.format(gpu_node))
            with open('pbs_gpu_deploy.sh','w') as f:
                f.write(deploy_txt)
            command = 'mksub pbs_gpu_deploy.sh'+('' if gpu_node == -1 else ' -l hostlist=g0{}'.format(gpu_node))
            print(command)
            job=os.popen(command).read().strip('\n')
        #df.loc[np.arange(df.shape[0])==np.where(df['--job_name'].astype(str).map(lower)=='false')[0][0],'--job_name']=job
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
                    pool = Pool(len(command_list))
                    pool.map(run, command_list)
                    pool.close()
                    pool.join()
    df.to_csv(hyperparameter_input_csv)
