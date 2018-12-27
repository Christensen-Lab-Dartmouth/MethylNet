#https://www.bioconductor.org/packages/devel/bioc/vignettes/missMethyl/inst/doc/missMethyl.html#gene-ontology-analysis
import shap, numpy as np
import torch

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


class CpGFinder:
    def __init__(self,model_master):
        self.model_master=model_master # has predict method
        self.model_master.output_latent = False

    def build_explainer(self, train_data): # can interpret latent dimensions
        self.explainer=shap.KernelExplainer(model.predict if 'predict' in dir(model) else model.transform, train_data, link="identity")

    def return_top_shapley_features(self, test_data, n_samples, n_top_features):
        shap_values = explainer.shap_values(test_data, nsamples=n_samples)
        top_cpgs=[]
        for i in range(len(shap_values)): # list of top cpgs, one per class
            top_cpgs.append(np.array(list(test_data))[np.argsort(shap_values[i].mean(0)*-1)[:n_top_features]]) # -np.abs(shap_values) # should I only consider the positive cases
        self.top_cpgs = top_cpgs # return shapley values

class BioInterpreter:
    def __init__(self, cpg_finder_obj):
        self.top_cpgs = cpg_finder_obj.top_cpgs

    def p(): pass # add rgreat, lola, roadmap-chromatin hmm, atac-seq, chip-seq, gometh, Hi-C, bedtools, Owen's analysis

CONTEXT_SETTINGS = dict(help_option_names=['-h','--help'], max_content_width=90)

@click.group(context_settings= CONTEXT_SETTINGS)
@click.version_option(version='0.1')
def interpret():
    pass

def data_loader_wrapped_predict(): # major modifications needed, think up new idea

@interpret.command()
@click.option('-v', '--add_validation_set', is_flag=True, help='Evaluate validation set.')
@click.option('-l', '--loss_reduction', default='sum', show_default=True, help='Type of reduction on loss function.', type=click.Choice(['sum','elementwise_mean','none']))
def return_important_cpgs(input_pkl,input_vae_pkl,output_dir,cuda,interest_cols,categorical,disease_only,hidden_layer_topology,learning_rate,weight_decay,n_epochs, scheduler='null', decay=0.5, t_max=10, eta_min=1e-6, t_mult=2, batch_size=50, train_percent=0.8, n_workers=8, add_validation_set=False, loss_reduction='sum'):

    cpg_finder.return_top_shapley_features()
    cpgs = cpg_finder.top_cpgs



#################

if __name__ == '__main__':
    interpret()
