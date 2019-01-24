#https://www.bioconductor.org/packages/devel/bioc/vignettes/missMethyl/inst/doc/missMethyl.html#gene-ontology-analysis
import shap, numpy as np, pandas as pd
import torch
from datasets import RawBetaArrayDataSet, Transformer
from torch.autograd import Variable
from torch.utils.data import DataLoader
from os.path import join
import click
import os
import pickle
from MethylationDataTypes import MethylationArray, MethylationArrays, extract_pheno_beta_df_from_pickle_dict



# after attaching classifier to VAE

# use SHAP to extract CpGs

# use gometh on extracted CpGs for each class in classifier!

# output to file and analyze

"""Normal breast 5hmC
chip seq enrichment
roadmap -> hmm chromatin state learning
Encode
Blueprint
Geo
lola r packages
https://bioconductor.org/packages/release/bioc/html/rGREAT.html
https://academic.oup.com/bioinformatics/article/26/13/1662/201195
http://great.stanford.edu/public/html/
https://www.google.com/search?ei=_7wXXNPfEKyK5wLbpI2oAw&q=great+enrichment+genomic+cpg+r&oq=great+enrichment+genomic+cpg+r&gs_l=psy-ab.3..33i299l3.529.724..899...0.0..0.84.160.2......0....1..gws-wiz.dJT2-0two-Q
https://www.rdocumentation.org/packages/missMethyl/versions/1.6.2/topics/gometh
https://bioconductor.org/packages/devel/bioc/vignettes/methylGSA/inst/doc/methylGSA-vignette.html

5hmc project

Read Lola and great papers
"""

def to_tensor(arr):
    return Transformer().generate()(arr)

class CpGExplainer: # consider shap.kmeans or grab representative sample of each outcome in training set for background ~ 39 * 2 samples, 39 cancers, should speed things up, small training set when building explainer https://github.com/slundberg/shap/issues/372
    def __init__(self,prediction_function, cuda):
        self.prediction_function=prediction_function
        self.cuda = cuda

    def build_explainer(self, train_methyl_array, method='kernel', batch_size=100): # can interpret latent dimensions
        self.method = method
        if self.method =='kernel':
            self.explainer=shap.KernelExplainer(self.prediction_function, train_methyl_array.return_raw_beta_array(), link="identity")
        elif self.method == 'deep':
            self.explainer=shap.DeepExplainer(self.prediction_function, to_tensor(train_methyl_array.return_raw_beta_array()) if not self.cuda else to_tensor(train_methyl_array.return_raw_beta_array()).cuda())
        elif self.method == 'gradient':
            self.explainer=shap.GradientExplainer(self.prediction_function, to_tensor(train_methyl_array.return_raw_beta_array()) if not self.cuda else to_tensor(train_methyl_array.return_raw_beta_array()).cuda(), batch_size=batch_size)
        else:
            print('Not Implemented, default to kernel explainer.')
            self.method = 'kernel'
            self.explainer=shap.KernelExplainer(self.prediction_function, train_methyl_array.return_raw_beta_array(), link="identity")

    def return_top_shapley_features(self, test_methyl_array, n_samples, n_top_features, n_outputs, shap_sample_batch_size=None, interest_col='disease', prediction_classes=None, summary_plot_file='', return_shapley_values=True, feature_selection=False, top_outputs=None):
        n_batch = 1
        if shap_sample_batch_size != None and self.method != 'deep' and not feature_selection:
            n_batch = int(n_samples/shap_sample_batch_size)
            n_samples = shap_sample_batch_size
        test_arr = test_methyl_array.return_raw_beta_array()
        shap_values = np.zeros((top_outputs if top_outputs != None else n_outputs,)+test_arr.shape)
        additional_opts = {}
        if self.method != 'kernel':
            additional_opts['ranked_outputs']=top_outputs
            test_arr=to_tensor(test_arr) if not self.cuda else to_tensor(test_arr).cuda()
        for i in range(n_batch):
            click.echo("Shapley Batch {}".format(i))
            if self.method == 'kernel' or self.method == 'gradient': # ranked_outputs=ranked_outputs, add if feature_selection
                svals=(self.explainer.shap_values(test_arr, nsamples=n_samples, **additional_opts)[0] if (self.method == 'gradient' and additional_opts['ranked_outputs'] != None) else self.explainer.shap_values(test_arr, nsamples=n_samples))
                shap_values += np.stack(svals,axis=0) if type(svals) == type([]) else svals
            else:
                shap_values += (self.explainer.shap_values(test_arr, **additional_opts)[0] if additional_opts['ranked_outputs'] !=None else self.explainer.shap_values(test_arr))
        shap_values/=float(n_batch)
        cpgs=np.array(list(test_methyl_array.beta))
        self.cpg_global_shapley_scores = np.abs(shap_values).mean(0).mean(0)
        print(self.cpg_global_shapley_scores)
        if not feature_selection:
            top_cpgs=[]
            for i in range(len(shap_values)): # list of top cpgs, one per class
                if prediction_classes != None:
                    shap_values_idx = np.flatnonzero((test_methyl_array.pheno[interest_col].values == prediction_classes[i]).astype(int))
                    top_feature_idx = np.argsort(shap_values[i, shap_values_idx, :].mean(0)*-1)[:n_top_features]
                else:
                    top_feature_idx=np.argsort(np.abs(shap_values[i, ...]).mean(0)*-1)[:n_top_features]
                top_cpgs.append(np.vstack([cpgs[top_feature_idx],np.abs(shap_values[i, ...]).mean(0)[top_feature_idx]]).T) # -np.abs(shap_values) # should I only consider the positive cases
            self.top_cpgs = top_cpgs # return shapley values
            self.shapley_values = [pd.DataFrame(shap_values[i, ...],index=test_methyl_array.beta.index,columns=cpgs) for i in range(shap_values.shape[0])]
        if summary_plot_file:
            import matplotlib.pyplot as plt
            if feature_selection:
                plt.figure()
                cpg_idx=np.argsort(self.cpg_global_shapley_scores*-1)[:40]
                shap_values_reduced = [shap_values[i, :, cpg_idx] for i in range(shap_values.shape[0])]
                shap.summary_plot(shap_values_reduced, test_methyl_array.beta.iloc[:,cpg_idx], plot_type='bar', max_display=min(test_methyl_array.beta.shape[1],40), class_names=prediction_classes if (prediction_classes != None and not feature_selection) else None, show=False)
                plt.savefig('{}_feature_selection.png'.format(summary_plot_file[:summary_plot_file.rfind('.')]))
            else:
                for i in range(len(self.top_cpgs)):
                    plt.figure()
                    cpg_idx = self.top_cpgs[i][:,0]
                    if 40 < len(cpg_idx):
                        cpg_idx = cpg_idx[:40]
                    cpg_idx = np.isin(cpgs,cpg_idx)
                    shap_values_reduced = [shap_values[i, :, cpg_idx] for i in range(shap_values.shape[0])]
                    shap.summary_plot(shap_values_reduced, test_methyl_array.beta.iloc[:,cpg_idx], plot_type='bar', class_names=prediction_classes if prediction_classes != None else None, show=False)
                    plt.savefig('{}_{}.png'.format(summary_plot_file[:summary_plot_file.rfind('.')],prediction_classes[i] if prediction_classes != None else i))

    def feature_select(self, methyl_array, n_top_features):
        cpgs = methyl_array.return_cpgs()
        print(cpgs)
        cpgs=cpgs[np.argsort(self.cpg_global_shapley_scores*-1)[:n_top_features]]
        print(cpgs)
        return methyl_array.subset_cpgs(cpgs)


class BioInterpreter:
    def __init__(self, top_cpgs, prediction_classes=None):
        self.top_cpgs = top_cpgs
        from rpy2.robjects.packages import importr
        self.hg19 = importr('IlluminaHumanMethylation450kanno.ilmn12.hg19')
        self.missMethyl=importr('missMethyl')
        self.limma=importr('limma')
        self.lola=importr('LOLA')
        importr('simpleCache')
        self.prediction_classes = prediction_classes
        if self.prediction_classes == None:
            self.prediction_classes = list(range(len(self.top_cpgs)))
        else:
            self.prediction_classes=list(map(lambda x: x.replace(' ',''),prediction_classes))

    def gometh(self, collection='GO', allcpgs=[], prediction_idx=[]):# consider turn into generator go or kegg # add rgreat, lola, roadmap-chromatin hmm, atac-seq, chip-seq, gometh, Hi-C, bedtools, Owen's analysis
        import rpy2.robjects as robjects
        from rpy2.robjects import pandas2ri
        #robjects.packages.importr('org.Hs.eg.db')
        pandas2ri.activate()
        output_dfs={}
        if allcpgs:
            allcpgs=robjects.vectors.StrVector(allcpgs)
        else:
            allcpgs=robjects.r('NULL')
        if not prediction_idx:
            prediction_idx=range(len(self.top_cpgs))
        else:
            prediction_idx=map(int,prediction_idx)
        for i in prediction_idx:
            list_cpgs=self.top_cpgs[i]
            print('Start Prediction {} BIO'.format(self.prediction_classes[i]))
            list_cpgs=robjects.vectors.StrVector(list_cpgs[:,0].tolist())
            if collection == 'GENE':
                mappedEz = self.missMethyl.getMappedEntrezIDs(sig_cpg=list_cpgs, all_cpg = allcpgs, array_type='450K')
                gometh_output = robjects.r('function (mappedEz) {data.frame(mappedEz$sig.eg[1:10])}')(mappedEz) # sig.eg[1:10]
                print(gometh_output)
            else:
                gometh_output = self.missMethyl.gometh(sig_cpg=list_cpgs,all_cpg=allcpgs,collection=collection)
                gometh_output = self.limma.topKEGG(gometh_output) if collection=='KEGG' else self.limma.topGO(gometh_output)
            # FIXME add get genes
            # genes = self.missMethyl.getMappedEntrezIDs(sig.cpg, all.cpg = NULL, array.type, anno = NULL)
            output_dfs['prediction_{}'.format(self.prediction_classes[i])]=pandas2ri.ri2py(robjects.r['as'](gometh_output,'data.frame'))
            print('GO/KEGG Computed for Prediction {} Cpgs: {}'.format(self.prediction_classes[i], ' '.join(list_cpgs)))
        return output_dfs

    def run_lola(self, all_cpgs=[], prediction_idx=[], lola_db='', cores=8):
        import rpy2.robjects as robjects
        from rpy2.robjects import pandas2ri
        order_by_max_rnk=robjects.r("function (dt) {dt[order(meanRnk, decreasing=FALSE),]}")
        pandas2ri.activate()
        cpg_locations = pandas2ri.ri2py(robjects.r['as'](robjects.r('getAnnotation(IlluminaHumanMethylation450kanno.ilmn12.hg19)'),'data.frame'))
        if not prediction_idx:
            prediction_idx=range(len(self.top_cpgs))
        else:
            prediction_idx=map(int,prediction_idx)
        all_cpg_regions = robjects.r('makeGRangesFromDataFrame')(pandas2ri.py2ri(cpg_locations if not all_cpgs else cpg_locations.loc[all_cpgs,:]),start_field='pos',end_field='pos',starts_in_df_are_0based=False)
        #robjects.r('load("{}")'.format(lola_rdata))
        lolaDB = self.lola.loadRegionDB(lola_db)#
        output_dfs={}
        for i in prediction_idx:
            list_cpgs=self.top_cpgs[i][:,0]
            cpg_location_subset = cpg_locations.loc[list_cpgs,:]
            location_subset = robjects.r('makeGRangesFromDataFrame')(pandas2ri.py2ri(cpg_location_subset),start_field='pos',end_field='pos',starts_in_df_are_0based=False)
            lola_output=self.lola.runLOLA(location_subset,all_cpg_regions,lolaDB,cores=cores)
            output_dfs['prediction_{}'.format(self.prediction_classes[i])]=pandas2ri.ri2py(robjects.r['as'](order_by_max_rnk(lola_output),'data.frame')).iloc[:20,:]
        return output_dfs
        #https://academic.oup.com/bioinformatics/article/32/4/587/1743969

CONTEXT_SETTINGS = dict(help_option_names=['-h','--help'], max_content_width=90)

@click.group(context_settings= CONTEXT_SETTINGS)
@click.version_option(version='0.1')
def interpret():
    pass

def return_predict_function(model, cuda):
    def predict(loader): # model can be VAE_MLP, TybaltTitusVAE, or CVAE
        model.eval()
        outputs=[]
        for input in loader:
            input=Variable(input)
            if cuda:
                input = input.cuda()
            outputs.append(np.squeeze(model.forward_predict(input).detach().cpu().numpy()))
        outputs=np.vstack(outputs)
        return outputs
    return predict

# https://github.com/slundberg/shap/blob/master/shap/common.py
# Provided model function fails when applied to the provided data set.

def return_dataloader_construct(n_workers,batch_size):
    def construct_data_loader(raw_beta_array):
        raw_beta_dataset=RawBetaArrayDataSet(raw_beta_array,Transformer())
        raw_beta_dataloader=DataLoader(dataset=raw_beta_dataset,
            num_workers=n_workers,
            batch_size=batch_size,
            shuffle=False)
        return raw_beta_dataloader
    return construct_data_loader

def main_prediction_function(n_workers,batch_size, model, cuda):
    dataloader_constructor=return_dataloader_construct(n_workers,batch_size)
    predict_function=return_predict_function(model, cuda)
    def main_predict(raw_beta_array):
        return predict_function(dataloader_constructor(raw_beta_array))
    return main_predict

@interpret.command()
@click.option('-i', '--train_pkl', default='./train_val_test_sets/train_methyl_array.pkl', help='Input database for beta and phenotype data. Use ./predictions/vae_mlp_methyl_arr.pkl or ./embeddings/vae_mlp_methyl_arr.pkl for vae interpretations.', type=click.Path(exists=False), show_default=True)
@click.option('-v', '--val_pkl', default='./train_val_test_sets/val_methyl_array.pkl', help='Val database for beta and phenotype data. Use ./predictions/vae_mlp_methyl_arr.pkl or ./embeddings/vae_mlp_methyl_arr.pkl for vae interpretations.', type=click.Path(exists=False), show_default=True)
@click.option('-t', '--test_pkl', default='./train_val_test_sets/test_methyl_array.pkl', help='Pickle containing testing set.', type=click.Path(exists=False), show_default=True)
@click.option('-m', '--model_pickle', default='./predictions/output_model.p', help='Pytorch model containing forward_predict method.', type=click.Path(exists=False), show_default=True)
@click.option('-w', '--n_workers', default=9, show_default=True, help='Number of workers.')
@click.option('-bs', '--batch_size', default=512, show_default=True, help='Batch size.')
@click.option('-c', '--cuda', is_flag=True, help='Use GPUs.')
@click.option('-ns', '--n_samples', default=200, show_default=True, help='Number of samples for SHAP output.')
@click.option('-nf', '--n_top_features', default=20, show_default=True, help='Top features to select for shap outputs. If feature selection, instead use this number to choose number features that make up top global score.')
@click.option('-o', '--output_dir', default='./interpretations/shapley_explanations/', help='Output directory for interpretations.', type=click.Path(exists=False), show_default=True)
@click.option('-mth', '--method', default='kernel', help='Explainer type.', type=click.Choice(['kernel','deep','gradient']), show_default=True)
@click.option('-ssbs', '--shap_sample_batch_size', default=0, help='Break up shapley computations into batches. Set to 0 for only 1 batch.', show_default=True)
@click.option('-r', '--n_random_representative', default=0, help='Number of representative samples to choose for background selection. Is this number for regression models, or this number per class for classification problems.', show_default=True)
@click.option('-col', '--interest_col', default='disease', help='Column of interest for sample selection for explainer training.', type=click.Path(exists=False), show_default=True)
@click.option('-rt', '--n_random_representative_test', default=0, help='Number of representative samples to choose for test set. Is this number for regression models, or this number per class for classification problems.', show_default=True)
@click.option('-e', '--categorical_encoder', default='./predictions/one_hot_encoder.p', help='One hot encoder if categorical model. If path exists, then return top positive controbutions per samples of that class. Encoded values must be of sample class as interest_col.', type=click.Path(exists=False), show_default=True)
@click.option('-plt', '--plot_summary', default='', help='Plot shap summaries for top cpgs of each class.', type=click.Path(exists=False), show_default=True)
@click.option('-fs', '--feature_selection', is_flag=True, help='Perform feature selection using top global SHAP scores.', show_default=True)
@click.option('-top', '--top_outputs', default=0, help='Get shapley values for fewer outputs if feature selection.', show_default=True)
@click.option('-vae', '--vae_interpret', is_flag=True, help='Use model to get shapley values for VAE latent dimensions, only works if proper datasets are set.', show_default=True)
def return_important_cpgs(train_pkl, val, test_pkl, model_pickle, n_workers, batch_size, cuda, n_samples, n_top_features, output_dir, method, shap_sample_batch_size, n_random_representative, interest_col, n_random_representative_test, categorical_encoder, plot_summary, feature_selection, top_outputs, vae_interpret):
    os.makedirs(output_dir,exist_ok=True)
    if not top_outputs or not feature_selection:
        top_outputs = None
    if os.path.exists(categorical_encoder):
        categorical_encoder=pickle.load(open(categorical_encoder,'rb'))
        prediction_classes=list(categorical_encoder.categories_[0])
    else:
        prediction_classes = None
    train_methyl_array, val_methyl_array, test_methyl_array=MethylationArray.from_pickle(train_pkl), MethylationArray.from_pickle(val_pkl), MethylationArray.from_pickle(val_pkl), MethylationArray.from_pickle(test_pkl)#preprocessed_methyl_array.subset_index(train_test_idx_dict['train']), preprocessed_methyl_array.subset_index(train_test_idx_dict['test'])
    #preprocessed_methyl_array=MethylationArray(*extract_pheno_beta_df_from_pickle_dict(input_dict))
    train_methyl_array = MethylationArrays([train_methyl_array,val_methyl_array]).combine()
    train_methyl_array
    cpgs, train_samples= train_methyl_array.return_cpgs(), train_methyl_array.return_idx()
    cpgs, test_samples= test_methyl_array.return_cpgs(), test_methyl_array.return_idx()
    model = torch.load(model_pickle)
    if n_random_representative and method != 'gradient':
        train_methyl_array = train_methyl_array.subsample(interest_col, n_samples=n_random_representative, categorical=model.categorical if 'categorical' in dir(model) else False)
    if n_random_representative_test:
        test_methyl_array = test_methyl_array.subsample(interest_col, n_samples=n_random_representative_test, categorical=model.categorical if 'categorical' in dir(model) else False)
    if 'categorical' in dir(model) and model.categorical:
        print("TRAIN:\n",train_methyl_array.pheno[interest_col].groupby(interest_col).count())
        print("TEST:\n",test_methyl_array.pheno[interest_col].groupby(interest_col).count())
    model.eval()
    if cuda:
        model = model.cuda()
    if vae_interpret:
        model.forward = model.mlp
    elif method == 'deep' or method == 'gradient':
            model.forward = model.forward_predict
    t_arr = train_methyl_array.return_raw_beta_array()
    prediction_function=main_prediction_function(n_workers,min(batch_size,t_arr.shape[0]), model, cuda) if method == 'kernel' else model
    # in order for shap to work, prediction_function must work with methylation array beta values
    n_test_results_outputs=main_prediction_function(n_workers, 1, model, cuda)(t_arr[:2, ...]).shape[1]
    # if above does not work, shap will not work
    cpg_explainer = CpGExplainer(prediction_function, cuda)
    cpg_explainer.build_explainer(train_methyl_array, method, batch_size=batch_size)
    if not shap_sample_batch_size:
        shap_sample_batch_size = None
    cpg_explainer.return_top_shapley_features(test_methyl_array, n_samples, n_top_features, n_outputs=n_test_results_outputs, shap_sample_batch_size=shap_sample_batch_size, interest_col=interest_col, prediction_classes=prediction_classes, top_outputs=top_outputs, summary_plot_file=(join(output_dir,'summary.png') if plot_summary else ''), feature_selection=feature_selection)
    if feature_selection:
        print('FEATURE SELECT')
        feature_selected_methyl_array=cpg_explainer.feature_select(MethylationArrays([train_methyl_array,test_methyl_array]).combine(),n_top_features)
        feature_selected_methyl_array.write_pickle(join(output_dir, 'feature_selected_methyl_array.pkl'))
    else:
        top_cpgs = cpg_explainer.top_cpgs
        shapley_values = cpg_explainer.shapley_values
        output_top_cpgs=join(output_dir,'top_cpgs.p')
        output_all_cpgs=join(output_dir,'all_cpgs.p')
        output_shapley_values=join(output_dir,'shapley_values.p')
        output_explainer=join(output_dir,'explainer.p')
        pickle.dump(train_methyl_array.return_cpgs(),open(output_all_cpgs,'wb'))
        pickle.dump(top_cpgs,open(output_top_cpgs,'wb'))
        pickle.dump(shapley_values,open(output_shapley_values,'wb'))
        pickle.dump(cpg_explainer.explainer,open(output_explainer,'wb'))

@interpret.command()
@click.option('-o', '--output_dir', default='./lola_db/', help='Output directory for lola dbs.', type=click.Path(exists=False), show_default=True)
def grab_lola_db_cache(output_dir):
    os.makedirs(output_dir,exist_ok=True)
    core_dir = join(output_dir,'core')
    extended_dir = join(output_dir,'extended')
    subprocess.call("wget http://big.databio.org/regiondb/LOLACoreCaches_180412.tgz && wget http://big.databio.org/regiondb/LOLAExtCaches_170206.tgz && mv *.tgz {0} && tar -xvzf {0}/LOLAExtCaches_170206.tgz && mv {0}/scratch {1} && tar -xvzf {0}/LOLACoreCaches_180412.tgz && mv {0}/nm {2}".format(output_dir,extended_dir,core_dir),shell=True)

@interpret.command()
@click.option('-a', '--all_cpgs_pickle', default='./interpretations/shapley_explanations/all_cpgs.p', help='List of all cpgs used in shapley analysis.', type=click.Path(exists=False), show_default=True)
@click.option('-t', '--top_cpgs_pickle', default='./interpretations/shapley_explanations/top_cpgs.p', help='Pickle containing top CpGs.', type=click.Path(exists=False), show_default=True)
@click.option('-e', '--categorical_encoder', default='./predictions/one_hot_encoder.p', help='One hot encoder if categorical model.', type=click.Path(exists=False), show_default=True)
@click.option('-o', '--output_dir', default='./interpretations/biological_explanations/', help='Output directory for interpretations.', type=click.Path(exists=False), show_default=True)
@click.option('-i', '--prediction_idx', multiple=True, default=[], help='Prediction indices, leave empty to output all indices. Inputs must be int.', show_default=True)
@click.option('-w', '--analysis', default='GO', help='Choose biological analysis.', type=click.Choice(['GO','KEGG','GENE', 'LOLA']), show_default=True)
@click.option('-n', '--n_workers', default=8, help='Number workers.', show_default=True)
@click.option('-l', '--lola_db', default='./lola_db/core/nm/t1/resources/regions/LOLACore/hg19/', help='LOLA region db.', type=click.Path(exists=False), show_default=True)
def interpret_biology(all_cpgs_pickle,top_cpgs_pickle,categorical_encoder,output_dir, prediction_idx, analysis, n_workers, lola_db):
    """Add categorical encoder as custom input, then can use to change names of output csvs to match disease if encoder exists."""
    os.makedirs(output_dir,exist_ok=True)
    if os.path.exists(categorical_encoder):
        categorical_encoder=pickle.load(open(categorical_encoder,'rb'))
        prediction_classes=list(categorical_encoder.categories_[0])
    else:
        prediction_classes = None
    if os.path.exists(all_cpgs_pickle):
        all_cpgs=list(pickle.load(open(all_cpgs_pickle,'rb')))
    else:
        all_cpgs = []
    top_cpgs=pickle.load(open(top_cpgs_pickle,'rb'))
    bio_interpreter = BioInterpreter(top_cpgs, prediction_classes)
    prediction_idx=list(prediction_idx)
    if analysis in ['GO','KEGG','GENE']:
        analysis_outputs=bio_interpreter.gometh(analysis, allcpgs=[], prediction_idx=prediction_idx)
    elif analysis == 'LOLA':
        analysis_outputs=bio_interpreter.run_lola(all_cpgs=[], prediction_idx=prediction_idx, lola_db=lola_db, cores=n_workers)
    for k in analysis_outputs:
        output_csv=join(output_dir,'{}_{}.csv'.format(k,analysis))
        analysis_outputs[k].to_csv(output_csv)

# create force plots for each sample??? use output shapley values and output explainer

#################

if __name__ == '__main__':
    interpret()
