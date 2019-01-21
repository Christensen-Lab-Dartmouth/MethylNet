import os, pandas as pd, numpy as np, subprocess

def coarse_scan(hyperparameter_input_csv, hyperparameter_output_log, generate_input, job_chunk_size, stratify_column, reset_all, torque, gpu, gpu_node, nohup, mlp=False):
    from itertools import cycle
    from pathos.multiprocessing import ProcessingPool as Pool
    generated_input=[]
    if mlp:
        grid={'--in_progress':True}
    else:
        grid={'--n_latent':[100,150,300,500],
              '--learning_rate':[1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,5e-2],
              '--weight_decay':[1e-4],'--n_epochs':[25,50,75,100],
              '--hidden_layer_encoder_topology':['','500,500','300,300,300','500,300','300','300,200'],
              '--kl_warm_up':[0], '--beta':[0.5,1,10,50,100,200,500],
              '--scheduler':['warm_restarts'], '--t_max':[10],
              '--eta_min':[1e-7,1e-6], '--t_mult':[1.,1.2,1.5,2],
              '--batch_size':[50,100,256,512], '--train_percent':[.85],
              '--n_workers':[6], '--loss_reduction':['sum']}
    if generate_input:
        if os.path.exists(hyperparameter_input_csv):
            for i in range(4):
                generated_input.append([False]+[np.random.choice(grid[k]) for k in grid])
            generated_input=[pd.DataFrame(generated_input,columns=['completed']+list(grid.keys()))]
        else:
            df=pd.DataFrame(np.array([False]+[grid[k][0] for k in grid])[np.newaxis,:],columns=['completed']+list(grid.keys()))
            df.to_csv(hyperparameter_input_csv)
            exit()
    def run(x):
        print(x)
        subprocess.call(x,shell=True)
    lower = lambda x: x.lower()
    if not torque and gpu == -1:
        gpus=cycle(range(4))
    else:
        gpus=cycle([gpu])
    df=pd.read_csv(hyperparameter_input_csv)
    df=df[[col for col in list(df) if not col.startswith('Unnamed')]]
    if generated_input:
        df=pd.concat([df]+generated_input,axis=0)
    if reset_all:
        df.loc[:,'completed']=False
    df_final = df[df['completed'].astype(str).map(lower)=='false'].reset_index(drop=True)[list(grid.keys())]
    commands=[]
    for i in range(df_final.shape[0]):
        job_id = str(np.random.randint(0,100000000))
        if not mlp:
            commands.append('sh -c "python embedding.py perform_embedding -bce -c -v -hl {} -sc {} {} && python visualizations.py transform_plot -i embeddings/vae_methyl_arr.pkl -o visualizations/{}_vae_embed.html -c disease_only -nn 10 "'.format(hyperparameter_output_log,stratify_column,' '.join(['{} {}'.format(k2,df_final.loc[i,k2]) for k2 in list(df_final) if df_final.loc[i,k2] != '']),job_id))
        else:
            commands.append('sh -c "python predictions.py make_prediction -cat -c -v -do -hl {} {} && python visualizations.py transform_plot -i predictions/vae_mlp_methyl_arr.pkl -o visualizations/{}_mlp_embed.html -c disease_only -nn 10 "'.format(hyperparameter_output_log,' '.join(['{} {}'.format(k2,df_final.loc[i,k2]) for k2 in list(df_final) if df_final.loc[i,k2] != '']),job_id))
        df.loc[np.arange(df.shape[0])==np.where(df['completed'].astype(str).map(lower)=='false')[0][0],'completed']=job_id
    for i in range(len(commands)):
        commands[i] = '{} {} {} {}'.format('CUDA_VISIBLE_DEVICES="{}"'.format(next(gpus)),'nohup' if nohup else '',commands[i],'&' if nohup else '')
    if torque:
        with open('pbs_embed_hyperparameters.sh','r') as f:
            deploy_txt=f.read().replace('COMMAND','\n'.join(commands+['wait'])).replace('HOST','g0{}'.format(gpu_node))
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
                    print(command)
                    subprocess.call(command,shell=True)
            else:
                pool = Pool(len(command_list))
                pool.map(run, command_list)
                pool.close()
                pool.join()
        #df.loc[:,'completed']=True
    df.to_csv(hyperparameter_input_csv)
