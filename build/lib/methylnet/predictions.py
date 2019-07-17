from pymethylprocess.MethylationDataTypes import MethylationArray, extract_pheno_beta_df_from_pickle_dict
from methylnet.models import *
from methylnet.datasets import get_methylation_dataset
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss, BCELoss, CrossEntropyLoss, NLLLoss
import pickle
import pandas as pd, numpy as np
import click
import os, copy
from os.path import join
from collections import Counter

CONTEXT_SETTINGS = dict(help_option_names=['-h','--help'], max_content_width=90)

@click.group(context_settings= CONTEXT_SETTINGS)
@click.version_option(version='0.1')
def prediction():
    pass

def train_predict(train_pkl,test_pkl,input_vae_pkl,output_dir,cuda,interest_cols,categorical,disease_only,hidden_layer_topology,learning_rate_vae,learning_rate_mlp,weight_decay,dropout_p,n_epochs, scheduler='null', decay=0.5, t_max=10, eta_min=1e-6, t_mult=2, batch_size=50, val_pkl='val_methyl_array.pkl', n_workers=8, add_validation_set=False, loss_reduction='sum', add_softmax=False):
    os.makedirs(output_dir,exist_ok=True)

    output_file = join(output_dir,'results.csv')
    training_curve_file = join(output_dir, 'training_val_curve.p')
    results_file = join(output_dir,'results.p')
    output_file_latent = join(output_dir,'latent.csv')
    output_model = join(output_dir,'output_model.p')
    output_pkl = join(output_dir, 'vae_mlp_methyl_arr.pkl')
    output_onehot_encoder = join(output_dir, 'one_hot_encoder.p')

    #input_dict = pickle.load(open(input_pkl,'rb'))
    if cuda:
        vae_model = torch.load(input_vae_pkl)
        vae_model.cuda_on=True
    else:
        vae_model = torch.load(input_vae_pkl,map_location='cpu')
        vae_model.cuda_on=False

    train_methyl_array, val_methyl_array, test_methyl_array = MethylationArray.from_pickle(train_pkl), MethylationArray.from_pickle(val_pkl), MethylationArray.from_pickle(test_pkl)#methyl_array.split_train_test(train_p=train_percent, stratified=(True if categorical else False), disease_only=disease_only, key=interest_cols[0], subtype_delimiter=',')

    if not categorical:
        train_methyl_array.remove_na_samples(interest_cols)
        val_methyl_array.remove_na_samples(interest_cols)
        test_methyl_array.remove_na_samples(interest_cols)

    print(train_methyl_array.beta.shape)
    print(val_methyl_array.beta.shape)
    print(test_methyl_array.beta.shape)

    if len(interest_cols) == 1 and disease_only and interest_cols[0].endswith('_only')==False:
        print(interest_cols)
        interest_cols[0] += '_only'
        print(train_methyl_array.pheno[interest_cols[0]].unique())
        print(test_methyl_array.pheno[interest_cols[0]].unique())

    train_methyl_dataset = get_methylation_dataset(train_methyl_array,interest_cols,categorical=categorical, predict=True) # train, test split? Add val set?
    #print(list(train_methyl_dataset.encoder.get_feature_names()))
    val_methyl_dataset = get_methylation_dataset(val_methyl_array,interest_cols,categorical=categorical, predict=True, categorical_encoder=train_methyl_dataset.encoder)
    test_methyl_dataset = get_methylation_dataset(test_methyl_array,interest_cols,categorical=categorical, predict=True, categorical_encoder=train_methyl_dataset.encoder)


    if not batch_size:
        batch_size=len(train_methyl_dataset)
    train_batch_size = min(batch_size,len(train_methyl_dataset))
    val_batch_size = min(batch_size,len(val_methyl_dataset))

    train_methyl_dataloader = DataLoader(
        dataset=train_methyl_dataset,
        num_workers=n_workers,
        batch_size=train_batch_size,
        shuffle=True)

    val_methyl_dataloader = DataLoader(
        dataset=val_methyl_dataset,
        num_workers=n_workers,
        batch_size=val_batch_size,
        shuffle=True) # False

    test_methyl_dataloader = DataLoader(
        dataset=test_methyl_dataset,
        num_workers=n_workers,
        batch_size=min(batch_size,len(test_methyl_dataset)),
        shuffle=False)

    scaling_factors = dict(val=float(len(val_methyl_dataset))/((len(val_methyl_dataset)//val_batch_size)*val_batch_size),
                           train_batch_size=train_batch_size,val_batch_size=val_batch_size)

    model=VAE_MLP(vae_model=vae_model,categorical=categorical,hidden_layer_topology=hidden_layer_topology,n_output=train_methyl_dataset.outcome_col.shape[1],dropout_p=dropout_p, add_softmax=add_softmax)

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
        vae_mlp.add_validation_set(val_methyl_dataloader)
    vae_mlp = vae_mlp.fit(train_methyl_dataloader)
    if 'encoder' in dir(train_methyl_dataset):
        pickle.dump(train_methyl_dataset.encoder,open(output_onehot_encoder,'wb'))
    results = dict(test={},train={},val={})
    results['train']['y_pred'], results['train']['y_true'], _, _ = vae_mlp.predict(train_methyl_dataloader)
    results['val']['y_pred'], results['val']['y_true'], _, _ = vae_mlp.predict(val_methyl_dataloader)
    del train_methyl_dataloader, train_methyl_dataset
    """methyl_dataset=get_methylation_dataset(methyl_array,interest_cols,predict=True)
    methyl_dataset_loader = DataLoader(
        dataset=methyl_dataset,
        num_workers=9,
        batch_size=1,
        shuffle=False)"""
    Y_pred, Y_true, latent_projection, _ = vae_mlp.predict(test_methyl_dataloader) # FIXME change to include predictions for all classes for AUC
    results['test']['y_pred'], results['test']['y_true'] = copy.deepcopy(Y_pred), copy.deepcopy(Y_true)
    if categorical:
        Y_true=Y_true.argmax(axis=1)[:,np.newaxis]
        Y_pred=Y_pred.argmax(axis=1)[:,np.newaxis]
    test_methyl_array = test_methyl_dataset.to_methyl_array()
    """if categorical:
        Y_true=test_methyl_dataset.encoder.inverse_transform(Y_true)[:,np.newaxis]
        Y_pred=test_methyl_dataset.encoder.inverse_transform(Y_pred)[:,np.newaxis]"""
    #sample_names = np.array(list(test_methyl_array.beta.index)) # FIXME
    #outcomes = np.array([outcome[0] for outcome in outcomes]) # FIXME
    Y_pred=pd.DataFrame(Y_pred.flatten() if (np.array(Y_pred.shape)==1).any() else Y_pred,index=test_methyl_array.beta.index,columns=(['y_pred'] if categorical else interest_cols))#dict(zip(sample_names,outcomes))
    Y_true=pd.DataFrame(Y_true.flatten() if (np.array(Y_true.shape)==1).any() else Y_true,index=test_methyl_array.beta.index,columns=(['y_true'] if categorical else interest_cols))
    results_df = pd.concat([Y_pred,Y_true],axis=1) if categorical else pd.concat([Y_pred.rename(columns={name:name+'_pred' for name in list(Y_pred)}),Y_true.rename(columns={name:name+'_true' for name in list(Y_pred)})],axis=1)  # FIXME
    latent_projection=pd.DataFrame(latent_projection,index=test_methyl_array.beta.index)
    test_methyl_array.beta=latent_projection
    test_methyl_array.write_pickle(output_pkl)
    pickle.dump(results,open(results_file,'wb'))
    pickle.dump(vae_mlp.training_plot_data,open(training_curve_file,'wb'))
    latent_projection.to_csv(output_file_latent)
    torch.save(vae_mlp.model,output_model)
    results_df.to_csv(output_file)#pickle.dump(outcome_dict, open(outcome_dict_file,'wb'))
    return latent_projection, Y_pred, Y_true, vae_mlp, scaling_factors

# ADD OUTPUT METRICS AND TRAINING PLOT CURVE

@prediction.command() # FIXME finish this!!
@click.option('-i', '--train_pkl', default='./train_val_test_sets/train_methyl_array.pkl', help='Input database for beta and phenotype data.', type=click.Path(exists=False), show_default=True)
@click.option('-tp', '--test_pkl', default='./train_val_test_sets/test_methyl_array.pkl', help='Test database for beta and phenotype data.', type=click.Path(exists=False), show_default=True)
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
@click.option('-dp', '--dropout_p', default=0.2, help='Dropout Percentage.', show_default=True)
@click.option('-e', '--n_epochs', default=50, help='Number of epochs to train over.', show_default=True)
@click.option('-s', '--scheduler', default='null', help='Type of learning rate scheduler.', type=click.Choice(['null','exp','warm_restarts']),show_default=True)
@click.option('-d', '--decay', default=0.5, help='Learning rate scheduler decay for exp selection.', show_default=True)
@click.option('-t', '--t_max', default=10, help='Number of epochs before cosine learning rate restart.', show_default=True)
@click.option('-eta', '--eta_min', default=1e-6, help='Minimum cosine LR.', show_default=True)
@click.option('-m', '--t_mult', default=2., help='Multiply current restart period times this number given number of restarts.', show_default=True)
@click.option('-bs', '--batch_size', default=50, show_default=True, help='Batch size.')
@click.option('-vp', '--val_pkl', default='./train_val_test_sets/val_methyl_array.pkl', help='Validation Set Methylation Array Location.', show_default=True, type=click.Path(exists=False),)
@click.option('-w', '--n_workers', default=9, show_default=True, help='Number of workers.')
@click.option('-v', '--add_validation_set', is_flag=True, help='Evaluate validation set.')
@click.option('-l', '--loss_reduction', default='sum', show_default=True, help='Type of reduction on loss function.', type=click.Choice(['sum','elementwise_mean','none']))
@click.option('-hl', '--hyperparameter_log', default='predictions/predict_hyperparameters_log.csv', show_default=True, help='CSV file containing prior runs.', type=click.Path(exists=False))
@click.option('-j', '--job_name', default='predict_job', show_default=True, help='Embedding job name.', type=click.Path(exists=False))
@click.option('-sft', '--add_softmax', is_flag=True, help='Add softmax for predicting probability distributions. Experimental.')
def make_prediction(train_pkl,test_pkl,input_vae_pkl,output_dir,cuda,interest_cols,categorical,disease_only,hidden_layer_topology,learning_rate_vae,learning_rate_mlp,weight_decay,dropout_p,n_epochs, scheduler='null', decay=0.5, t_max=10, eta_min=1e-6, t_mult=2, batch_size=50, val_pkl='val_methyl_array.pkl', n_workers=8, add_validation_set=False, loss_reduction='sum', hyperparameter_log='predictions/predict_hyperparameters_log.csv', job_name='predict_job', add_softmax=False):
    """Train prediction model by fine-tuning VAE and appending/training MLP to make classification/regression predictions on MethylationArrays."""
    hlt_list=filter(None,hidden_layer_topology.split(','))
    if hlt_list:
        hidden_layer_topology=list(map(int,hlt_list))
    else:
        hidden_layer_topology=[]
    latent_projection, Y_pred, Y_true, vae_mlp, scaling_factors = train_predict(train_pkl,test_pkl,input_vae_pkl,output_dir,cuda,list(interest_cols),categorical,disease_only,hidden_layer_topology,learning_rate_vae,learning_rate_mlp,weight_decay,dropout_p,n_epochs, scheduler, decay, t_max, eta_min, t_mult, batch_size, val_pkl, n_workers, add_validation_set, loss_reduction, add_softmax)
    accuracy, precision, recall, f1 = -1,-1,-1,-1
    if categorical:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        accuracy, precision, recall, f1 = accuracy_score(Y_true,Y_pred), precision_score(Y_true,Y_pred,average='weighted'), recall_score(Y_true,Y_pred,average='weighted'), f1_score(Y_true,Y_pred,average='weighted')
    else:
        from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error
        accuracy, precision, recall, f1 = mean_squared_error(Y_true,Y_pred,multioutput='raw_values'), r2_score(Y_true,Y_pred,multioutput='raw_values'), explained_variance_score(Y_true,Y_pred,multioutput='raw_values'), mean_absolute_error(Y_true,Y_pred,multioutput='raw_values')
        if len(interest_cols)>1:
            accuracy, precision, recall, f1 = ';'.join([str(v) for v in accuracy]),';'.join([str(v) for v in precision]),';'.join([str(v) for v in recall]),';'.join([str(v) for v in f1])
        else:
            accuracy, precision, recall, f1=accuracy[0], precision[0], recall[0], f1[0]
    hyperparameter_row = [job_name,n_epochs, vae_mlp.best_epoch, vae_mlp.min_loss, vae_mlp.min_val_loss, vae_mlp.min_val_loss*scaling_factors['val'], accuracy, precision, recall, f1, vae_mlp.model.vae.n_input, vae_mlp.model.vae.n_latent, str(hidden_layer_topology), learning_rate_vae, learning_rate_mlp, weight_decay, scheduler, t_max, t_mult, eta_min, scaling_factors['train_batch_size'], scaling_factors['val_batch_size'], dropout_p]
    hyperparameter_df = pd.DataFrame(columns=['job_name','n_epochs',"best_epoch", "min_loss", "min_val_loss", "min_val_loss-batchsize_adj","test_accuracy" if categorical else 'neg_mean_squared_error', "test_precision" if categorical else 'r2_score', "test_recall" if categorical else 'explained_variance', "test_f1" if categorical else 'mean_absolute_error', "n_input", "n_latent", "hidden_layer_encoder_topology", "learning_rate_vae", "learning_rate_mlp", "weight_decay", "scheduler", "t_max", "t_mult", "eta_min","train_batch_size", "val_batch_size", "dropout_p"])
    hyperparameter_df.loc[0] = hyperparameter_row
    if os.path.exists(hyperparameter_log):
        print('APPEND')
        hyperparameter_df_former = pd.read_csv(hyperparameter_log)
        hyperparameter_df_former=hyperparameter_df_former[[col for col in list(hyperparameter_df) if not col.startswith('Unnamed')]]
        hyperparameter_df=pd.concat([hyperparameter_df_former,hyperparameter_df],axis=0)
    hyperparameter_df.to_csv(hyperparameter_log)

@prediction.command()
@click.option('-tp', '--test_pkl', default='./train_val_test_sets/test_methyl_array.pkl', help='Test database for beta and phenotype data.', type=click.Path(exists=False), show_default=True)
@click.option('-m', '--model_pickle', default='./predictions/output_model.p', help='Pytorch model containing forward_predict method.', type=click.Path(exists=False), show_default=True)
@click.option('-bs', '--batch_size', default=50, show_default=True, help='Batch size.')
@click.option('-w', '--n_workers', default=9, show_default=True, help='Number of workers.')
@click.option('-ic', '--interest_cols', default=['disease'], multiple=True, help='Specify columns looking to make predictions on.', show_default=True)
@click.option('-cat', '--categorical', is_flag=True, help='Multi-class prediction.', show_default=True)
@click.option('-c', '--cuda', is_flag=True, help='Use GPUs.')
@click.option('-e', '--categorical_encoder', default='./predictions/one_hot_encoder.p', help='One hot encoder if categorical model. If path exists, then return top positive controbutions per samples of that class. Encoded values must be of sample class as interest_col.', type=click.Path(exists=False), show_default=True)
@click.option('-o', '--output_dir', default='./new_predictions/', help='Output directory for predictions.', type=click.Path(exists=False), show_default=True)
def make_new_predictions(test_pkl, model_pickle, batch_size, n_workers, interest_cols, categorical, cuda, categorical_encoder, output_dir):
    """Run prediction model again to further assess outcome. Only evaluate prediction model."""
    os.makedirs(output_dir,exist_ok=True)
    test_methyl_array = MethylationArray.from_pickle(test_pkl) # generate results pickle to run through classification/regression report
    if cuda:
        model = torch.load(model_pickle)
        model.vae.cuda_on=True
    else:
        model = torch.load(model_pickle,map_location='cpu')
        model.vae.cuda_on=False
    if not categorical:
        test_methyl_array.remove_na_samples(interest_cols if len(interest_cols)>1 else interest_cols[0])
    if os.path.exists(categorical_encoder):
        categorical_encoder=pickle.load(open(categorical_encoder,'rb'))
    else:
        categorical_encoder=None
    test_methyl_dataset = get_methylation_dataset(test_methyl_array,interest_cols,categorical=categorical, predict=True, categorical_encoder=categorical_encoder)
    test_methyl_dataloader = DataLoader(
        dataset=test_methyl_dataset,
        num_workers=n_workers,
        batch_size=min(batch_size,len(test_methyl_dataset)),
        shuffle=False)
    vae_mlp=MLPFinetuneVAE(mlp_model=model,categorical=categorical,cuda=cuda)

    Y_pred, Y_true, latent_projection, _ = vae_mlp.predict(test_methyl_dataloader)

    results = dict(test={})
    results['test']['y_pred'], results['test']['y_true'] = copy.deepcopy(Y_pred), copy.deepcopy(Y_true)

    if categorical:
        Y_true=Y_true.argmax(axis=1)[:,np.newaxis]
        Y_pred=Y_pred.argmax(axis=1)[:,np.newaxis]
    test_methyl_array = test_methyl_dataset.to_methyl_array()

    Y_pred=pd.DataFrame(Y_pred.flatten() if (np.array(Y_pred.shape)==1).any() else Y_pred,index=test_methyl_array.beta.index,columns=(['y_pred'] if categorical else interest_cols))
    Y_true=pd.DataFrame(Y_true.flatten() if (np.array(Y_true.shape)==1).any() else Y_true,index=test_methyl_array.beta.index,columns=(['y_true'] if categorical else interest_cols))
    results_df = pd.concat([Y_pred,Y_true],axis=1) if categorical else pd.concat([Y_pred.rename(columns={name:name+'_pred' for name in list(Y_pred)}),Y_true.rename(columns={name:name+'_true' for name in list(Y_pred)})],axis=1)  # FIXME
    latent_projection=pd.DataFrame(latent_projection,index=test_methyl_array.beta.index)
    test_methyl_array.beta=latent_projection

    output_file = join(output_dir,'results.csv')
    results_file = join(output_dir,'results.p')
    output_file_latent = join(output_dir,'latent.csv')
    output_pkl = join(output_dir, 'vae_mlp_methyl_arr.pkl')

    test_methyl_array.write_pickle(output_pkl)
    pickle.dump(results,open(results_file,'wb'))
    latent_projection.to_csv(output_file_latent)
    results_df.to_csv(output_file)

@prediction.command()
@click.option('-hcsv', '--hyperparameter_input_csv', default='predictions/predict_hyperparameters_scan_input.csv', show_default=True, help='CSV file containing hyperparameter inputs.', type=click.Path(exists=False))
@click.option('-hl', '--hyperparameter_output_log', default='predictions/predict_hyperparameters_log.csv', show_default=True, help='CSV file containing prior runs.', type=click.Path(exists=False))
@click.option('-g', '--generate_input', is_flag=True, help='Generate hyperparameter input csv.')
@click.option('-c', '--job_chunk_size', default=4, help='If not series, chunk up and run these number of commands at once..')
@click.option('-ic', '--interest_cols', default=['disease_only'], multiple=True, help='Column to stratify samples on.')
@click.option('-cat', '--categorical', is_flag=True, help='Whether to run categorical analysis or not.')
@click.option('-r', '--reset_all', is_flag=True, help='Run all jobs again.')
@click.option('-t', '--torque', is_flag=True, help='Submit jobs on torque.')
@click.option('-gpu', '--gpu', default=-1, help='If torque submit, which gpu to use.', show_default=True)
@click.option('-gn', '--gpu_node', default=-1, help='If torque submit, which gpu node to use.', show_default=True)
@click.option('-nh', '--nohup', is_flag=True, help='Nohup launch jobs.')
@click.option('-n', '--n_jobs_relaunch', default=0, help='Relaunch n top jobs from previous run.', show_default=True)
@click.option('-cp', '--crossover_p', default=0., help='Rate of crossover between hyperparameters.', show_default=True)
@click.option('-mc', '--model_complexity_factor', default=1., help='Degree of neural network model complexity for hyperparameter search. Search for less wide networks with a lower complexity value, bounded between 0 and infinity.', show_default=True)
@click.option('-j', '--n_jobs', default=4, help='Number of jobs to generate.')
@click.option('-v', '--val_loss_column', default="min_val_loss-batchsize_adj", help='Validation loss column.', type=click.Path(exists=False))
@click.option('-sft', '--add_softmax', is_flag=True, help='Add softmax for predicting probability distributions.')
@click.option('-a', '--additional_command', default='', help='Additional command to input for torque run.', type=click.Path(exists=False))
@click.option('-cu', '--cuda', is_flag=True, help='Use GPUs.')
@click.option('-grid', '--hyperparameter_yaml', default='', help='YAML file with custom subset of hyperparameter grid.', type=click.Path(exists=False))
def launch_hyperparameter_scan(hyperparameter_input_csv, hyperparameter_output_log, generate_input, job_chunk_size, interest_cols, categorical, reset_all, torque, gpu, gpu_node, nohup, n_jobs_relaunch, crossover_p, model_complexity_factor,n_jobs, val_loss_column, add_softmax, additional_command, cuda, hyperparameter_yaml):
    """Run randomized hyperparameter scan of neural network hyperparameters."""
    from methylnet.hyperparameter_scans import coarse_scan, find_top_jobs
    custom_jobs=[]
    if n_jobs_relaunch:
        custom_jobs=find_top_jobs(hyperparameter_input_csv, hyperparameter_output_log,n_jobs_relaunch, crossover_p, val_loss_column)
    if os.path.exists(hyperparameter_yaml):
        from ruamel.yaml import safe_load as load
        with open(hyperparameter_yaml) as f:
            new_grid = load(f.read())
    else:
        new_grid = {}
    coarse_scan(hyperparameter_input_csv, hyperparameter_output_log, generate_input, job_chunk_size, interest_cols, reset_all, torque, gpu, gpu_node, nohup, mlp=True, custom_jobs=custom_jobs, model_complexity_factor=model_complexity_factor,n_jobs=n_jobs, categorical=categorical,add_softmax=add_softmax, additional_command=additional_command, cuda=cuda, new_grid=new_grid)

@prediction.command()
@click.option('-r', '--results_pickle', default='predictions/results.p', show_default=True, help='Results from training, validation, and testing.', type=click.Path(exists=False))
@click.option('-o', '--output_dir', default='results/', show_default=True, help='Output directory.', type=click.Path(exists=False))
def regression_report(results_pickle,output_dir):
    """Generate regression report that gives concise results from regression tasks."""
    # FIXME expand functionality
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="whitegrid")
    os.makedirs(output_dir,exist_ok=True)
    results_dict=pickle.load(open(results_pickle,'rb'))
    for k in results_dict:
        df_pred=pd.DataFrame(results_dict[k]['y_pred']).melt()
        df_pred=df_pred.rename(columns=dict(value='y_pred',variable='outcome_vars'))
        df_true=pd.DataFrame(results_dict[k]['y_true']).melt()
        df_true=df_true.rename(columns=dict(value='y_true',variable='true_vars'))
        df=pd.concat([df_pred,df_true],axis=1)
        #print(df)
        df=df[['outcome_vars','y_pred','y_true']]
        plt.figure()
        sns.lmplot(x="y_pred", y="y_true", col="outcome_vars", hue="outcome_vars",data=df, col_wrap=2)
        plt.savefig(os.path.join(output_dir,'{}_regression_results.png'.format(k)),dpi=300)

@prediction.command()
@click.option('-r', '--results_pickle', default='predictions/results.p', show_default=True, help='Results from training, validation, and testing.', type=click.Path(exists=False))
@click.option('-o', '--output_dir', default='results/', show_default=True, help='Output directory.', type=click.Path(exists=False))
@click.option('-e', '--categorical_encoder', default='./predictions/one_hot_encoder.p', help='One hot encoder if categorical model. If path exists, then return top positive controbutions per samples of that class. Encoded values must be of sample class as interest_col.', type=click.Path(exists=False), show_default=True)
@click.option('-a', '--average_mechanism', default='micro', show_default=True, help='Output directory.', type=click.Choice(['weighted','macro','micro','binary','samples']))
def classification_report(results_pickle,output_dir, categorical_encoder, average_mechanism):
    """Generate classification report that gives results from classification tasks."""
    from mlxtend.evaluate import bootstrap
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, roc_auc_score, confusion_matrix
    os.makedirs(output_dir,exist_ok=True)

    def extract_ys(Y):
        return Y[:,0], Y[:,1:]

    def accuracy(Y):
        y_true, y_pred=extract_ys(Y)
        y_pred = np.argmax(y_pred,axis=1)
        return accuracy_score(y_true, y_pred)

    def recall(Y):
        y_true, y_pred=extract_ys(Y)
        y_pred = np.argmax(y_pred,axis=1)
        return recall_score(y_true, y_pred, average=average_mechanism)

    def precision(Y):
        y_true, y_pred=extract_ys(Y)
        y_pred = np.argmax(y_pred,axis=1)
        return precision_score(y_true, y_pred, average=average_mechanism)

    def f1(Y):
        y_true, y_pred=extract_ys(Y)
        y_pred = np.argmax(y_pred,axis=1)
        return f1_score(y_true, y_pred, average=average_mechanism)

    def auc(Y):
        y_true, y_pred=extract_ys(Y)
        y_pred_labels=np.argmax(y_pred,1)
        supports={i:sum((y_pred_labels[np.squeeze(y_true==i)]==i).astype(int)) for i in np.unique(y_true)}
        final_auc_score=0.
        for i in supports:
            final_auc_score+=supports[i]*roc_auc_score((y_true==i).astype(int), y_pred[:,int(i)], average='weighted')
        final_auc_score/=sum(list(supports.values()))
        return final_auc_score

    def to_y_probas(y_pred):
        return np.exp(y_pred)/np.exp(y_pred).sum(1)[:,np.newaxis]

    results_dict=pickle.load(open(results_pickle,'rb'))
    if os.path.exists(categorical_encoder):
        categorical_encoder=pickle.load(open(categorical_encoder,'rb'))
    else:
        categorical_encoder=None
    df_roc=[]
    final_results=[]
    for k in results_dict:
        y_true=results_dict[k]['y_true']
        y_true_labels=np.argmax(y_true,axis=1)[:,np.newaxis]
        classes=np.unique(y_true_labels)
        y_pred=to_y_probas(results_dict[k]['y_pred'])
        y_pred_labels=np.argmax(y_pred,1).reshape((-1,1))
        #print(y_true_labels, y_pred_labels)
        out_classes = classes.astype(int)# if categorical_encoder == None else categorical_encoder.inverse_transform(classes.astype(int).reshape((-1,1)))
        class_labels = out_classes if categorical_encoder == None else categorical_encoder.inverse_transform(out_classes)
        pd.DataFrame(confusion_matrix(y_true_labels.astype(int).flatten(),
                                      y_pred_labels.astype(int).flatten(),labels=out_classes),index=class_labels,columns=class_labels).to_csv(join(output_dir,'{}_confusion_mat.csv'.format(k)))
        Y=np.hstack((y_true_labels,y_pred))
        supports={i:sum((y_pred_labels[np.squeeze(y_true_labels==i)]==i).astype(int)) for i in classes}
        fpr = dict()
        tpr = dict()
        for i in supports:
            fpr[i], tpr[i], _ = roc_curve(y_true[:,i], y_pred[:,i])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in supports]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in supports:
            mean_tpr += supports[i]*np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr/=sum(list(supports.values()))
        tpr,fpr=mean_tpr,all_fpr
        df=pd.DataFrame({'fpr' : fpr, 'tpr' : tpr})
        df['Legend'] = "{} Weighted ROC, AUC={}".format(k,round(auc(Y),2))
        df_roc.append(df)
        fns = dict(accuracy=accuracy,recall=recall,precision=precision,f1=f1,auc=auc)
        for fn in fns:
            print(k,fn)
            original, std_err, ci_bounds = bootstrap(Y, num_rounds=1000,
                                             func=fns[fn],
                                             ci=0.95,
                                             seed=123)
            low,high=ci_bounds
            final_results.append([k,fn,original,std_err,low,high])
    final_results=pd.DataFrame(final_results,columns=['DataSet','Metric','Score','Error','95% CI Low','95% CI High'])
    df_roc=pd.concat(df_roc)
    final_results.to_csv(join(output_dir,'final_classification_results.csv'))
    df_roc.to_csv(join(output_dir,'Weighted_ROC.csv'))

#################

if __name__ == '__main__':
    prediction()
