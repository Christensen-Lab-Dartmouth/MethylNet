#https://www.bioconductor.org/packages/devel/bioc/vignettes/missMethyl/inst/doc/missMethyl.html#gene-ontology-analysis
import shap, numpy as np
import torch
from datasets import RawBetaArrayDataSet, Transformer
from torch.autograd import Variable
from torch.utils.data import DataLoader
from os.path import join



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


class CpGExplainer:
    def __init__(self,prediction_function):
        self.prediction_function=prediction_function

    def build_explainer(self, train_methyl_array): # can interpret latent dimensions
        self.explainer=shap.KernelExplainer(self.prediction_function, train_methyl_array.return_raw_beta_array(), link="identity")

    def return_top_shapley_features(self, test_methyl_array, n_samples, n_top_features):
        shap_values = self.explainer.shap_values(test_methyl_array.return_raw_beta_array(), nsamples=n_samples)
        cpgs=np.array(list(test_methyl_array.beta))
        top_cpgs=[]
        for i in range(len(shap_values)): # list of top cpgs, one per class
            top_feature_idx=np.argsort(shap_values[i].mean(0)*-1)[:n_top_features]
            top_cpgs.append(np.vstack([cpgs[top_feature_idx],shap_values[i].mean(0)[top_feature_idx]]).T) # -np.abs(shap_values) # should I only consider the positive cases
        self.top_cpgs = top_cpgs # return shapley values
        self.shapley_features = [pd.DataFrame(shap_values[i],index=test_methyl_array.beta.index,columns=cpgs) for i in range(len(shap_values))]

class BioInterpreter:
    def __init__(self, top_cpgs):
        self.top_cpgs = top_cpgs

    def gometh(self, collection='GO', allcpgs=[]):# go or kegg # add rgreat, lola, roadmap-chromatin hmm, atac-seq, chip-seq, gometh, Hi-C, bedtools, Owen's analysis
        import rpy2.robjects as robjects
        from rpy2.robjects.packages import importr
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()
        output_dfs={}
        missMethyl=importr('missMethyl')
        limma=importr('limma')
        if allcpgs:
            allcpgs=robjects.vectors.StrVector(allcpgs)
        else:
            allcpgs=robjects.r('NULL')
        for i, list_cpgs in enumerate(self.top_cpgs):
            list_cpgs=robjects.vectors.StrVector(list_cpgs[:,0].tolist())
            gometh_output = missMethyl.gometh(sig_cpg=list_cpgs,all_cpg=allcpgs,collection=collection)
            gometh_output = limma.topKEGG(gometh_output) if collection=='KEGG' else limma.topGO(gometh_output)
            output_dfs['prediction_{}'.format(i)]=pandas2ri.ri2py(robjects.r['as'](gometh_output,'data.frame'))
        return output_dfs


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
@click.option('-i', '--input_pkl', default='./final_preprocessed/methyl_array.pkl', help='Input database for beta and phenotype data.', type=click.Path(exists=False), show_default=True)
@click.option('-t', '--train_test_idx_pkl', default='./predictions/train_test_idx.p', help='Pickle containing training and testing indices.', type=click.Path(exists=False), show_default=True)
@click.option('-m', '--model_pickle', default='./predictions/output_model.p', help='Pytorch model containing forward_predict method.', type=click.Path(exists=False), show_default=True)
@click.option('-w', '--n_workers', default=9, show_default=True, help='Number of workers.')
@click.option('-bs', '--batch_size', default=50, show_default=True, help='Batch size.')
@click.option('-c', '--cuda', is_flag=True, help='Use GPUs.')
@click.option('-ns', '--n_samples', default=500, show_default=True, help='Number of samples for SHAP output.')
@click.option('-nf', '--n_top_features', default=20, show_default=True, help='Top features to select for shap outputs.')
@click.option('-o', '--output_dir', default='./interpretations/shapley_explanations/', help='Output directory for interpretations.', type=click.Path(exists=False), show_default=True)
def return_important_cpgs(input_pkl, train_test_idx_pkl, model_pickle, n_workers, batch_size, cuda, n_samples, n_top_features, output_dir):
    os.makedirs(output_dir,exist_ok=True)
    input_dict = pickle.load(open(input_pkl,'rb'))
    preprocessed_methyl_array=MethylationArray(*extract_pheno_beta_df_from_pickle_dict(input_dict))
    cpgs, samples= preprocessed_methyl_array.return_cpgs(), preprocessed_methyl_array.return_idx()
    train_test_idx_dict=pickle.load(open(train_test_idx_pkl,'rb'))
    train_methyl_array, test_methyl_array=preprocessed_methyl_array.subset_index(train_test_idx_dict['train']), preprocessed_methyl_array.subset_index(train_test_idx_dict['test'])
    model = torch.load(model_pickle)
    prediction_function=main_prediction_function(n_workers,batch_size, model, cuda)
    cpg_explainer = CpGExplainer(prediction_function)
    cpg_explainer.build_explainer(train_methyl_array)
    cpg_explainer.return_top_shapley_features(test_methyl_array, n_samples, n_top_features)
    top_cpgs = cpg_explainer.top_cpgs
    shapley_values = cpg_explainer.shapley_values
    output_top_cpgs=join(output_dir,'top_cpgs.p')
    output_shapley_values=join(output_dir,'shapley_values.p')
    output_explainer=join(output_dir,'explainer.p')
    pickle.dump(top_cpgs,open(output_top_cpgs,'wb'))
    pickle.dump(shapley_values,open(output_shapley_values,'wb'))
    pickle.dump(cpg_explainer.explainer,open(output_explainer,'wb'))

@interpret.command()
@click.option('-t', '--top_cpgs_pickle', default='./interpretations/shapley_explanations/top_cpgs.p', help='Pickle containing top CpGs.', type=click.Path(exists=False), show_default=True)
@click.option('-o', '--output_dir', default='./interpretations/biological_explanations/', help='Output directory for interpretations.', type=click.Path(exists=False), show_default=True)
def gometh_cpgs(top_cpgs_pickle,output_dir):
    os.makedirs(output_dir,exist_ok=True)
    top_cpgs=pickle.load(open(top_cpgs_pickle,'rb'))
    bio_interpreter = BioInterpreter(top_cpgs)
    go_outputs=bio_interpreter.gometh('GO')
    kegg_outputs=bio_interpreter.gometh('KEGG')
    for k in go_outputs:
        output_csvs={'GO':join(output_dir,'{}_GO.csv'.format(k)),'KEGG':join(output_dir,'{}_KEGG.csv'.format(k))}
        go_outputs[k].to_csv(output_csvs['GO'])
        kegg_outputs[k].to_csv(output_csvs['KEGG'])

# create force plots for each sample??? use output shapley values and output explainer

#################

if __name__ == '__main__':
    interpret()
