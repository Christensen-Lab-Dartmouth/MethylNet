#https://www.bioconductor.org/packages/devel/bioc/vignettes/missMethyl/inst/doc/missMethyl.html#gene-ontology-analysis
import shap, numpy as np, pandas as pd
import torch
from datasets import RawBetaArrayDataSet, Transformer
from torch.autograd import Variable
from torch.utils.data import DataLoader
from os.path import join
import click
import os, copy
import pickle
from pymethylprocess.MethylationDataTypes import MethylationArray, MethylationArrays, extract_pheno_beta_df_from_pickle_dict



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

class ShapleyData:
    def __init__(self):
        """
        top_cpgs:
            classes:
                invididuals:
                    individual:
                        top_cpgs
                overall:
                    top_cpgs
            overall:
                top_cpgs
        shapley_values:
            class:
                individual:
                    shapley_values
            overall:
                shapley_values
        methods to access data
        force_plot
        summary_plot
        explainer.expected_value
        save figures
        add methods for misclassified
        return global shapley scores
        """
        self.top_cpgs={'by_class':{},'overall':{}}
        self.shapley_values={'by_class':{},'overall':{}}

    def add_class(self, class_name, shap_df, cpgs, n_top_cpgs, add_top_negative=False):
        signed = -1 if not add_top_negative else 1
        self.shapley_values['by_class'][class_name]=shap_df
        shap_vals=shap_df.values
        class_importance_shaps = shap_vals.mean(0)
        top_idx = np.argsort(class_importance_shaps*signed)[:n_top_cpgs]
        self.top_cpgs['by_class'][class_name]={'by_individual':{},'overall':{}}
        self.top_cpgs['by_class'][class_name]['overall']=pd.DataFrame(np.hstack([cpgs[top_idx][:,np.newaxis],class_importance_shaps[top_idx][:,np.newaxis]]),columns=['cpg','shapley_value'])
        top_idxs = np.argsort(shap_vals*signed)[:,:n_top_cpgs]
        for i,individual in enumerate(list(shap_df.index)):
            self.top_cpgs['by_class'][class_name]['by_individual'][individual]=pd.DataFrame(shap_df.iloc[i,top_idxs[i,:]].T.reset_index(drop=False).values,columns=['cpg','shapley_value'])

    def add_global_importance(self, global_importance_shaps, cpgs, n_top_cpgs):
        self.shapley_values['overall']=pd.DataFrame(global_importance_shaps[:,np.newaxis],columns=['shapley_values'],index=cpgs)
        top_ft_idx=np.argsort(global_importance_shaps*-1)[:n_top_cpgs]
        self.top_cpgs['overall']=pd.DataFrame(np.hstack([cpgs[top_ft_idx][:,np.newaxis],global_importance_shaps[top_ft_idx][:,np.newaxis]]),columns=['cpg','shapley_value'])

    def to_pickle(self,output_pkl):
        os.makedirs(output_pkl[:output_pkl.rfind('/')],exist_ok=True)
        pickle.dump(self, open(output_pkl,'wb'))

    @classmethod
    def from_pickle(self,input_pkl):
        return pickle.load(open(input_pkl,'rb'))

class ShapleyDataExplorer:
    def __init__(self, shapley_data):
        self.shapley_data=shapley_data
        self.indiv2class={}
        for class_name in self.shapley_data.top_cpgs['by_class']:
            for individual in self.shapley_data.top_cpgs['by_class'][class_name]['by_individual'].keys():
                self.indiv2class[individual]=class_name

    def return_shapley_data_by_methylation_status(self, methyl_array):
        methylation_shapley_data_dict = {'hyper':copy.deepcopy(self.shapley_data),'hypo':copy.deepcopy(self.shapley_data)}
        for individual in self.list_individuals(return_list=True):
            class_name,individual,top_shap_df=self.view_methylation(individual,methyl_array)
            hyper_df = top_shap_df[top_shap_df['methylation']>0.5]
            hypo_df = top_shap_df[top_shap_df['methylation']<0.5]
            methylation_shapley_data_dict['hyper'].top_cpgs['by_class'][class_name]['by_individual'][individual]=hyper_df[['cpg','shapley_value']]
            methylation_shapley_data_dict['hypo'].top_cpgs['by_class'][class_name]['by_individual'][individual]=hypo_df[['cpg','shapley_value']]
        for class_name,individuals in self.list_individuals().items():
            top_shap_df=self.extract_class(class_name)
            top_shap_df['methylation']=methyl_array.beta.loc[individuals,self.shapley_data.top_cpgs['by_class'][class_name]['overall']['cpg'].values].mean(axis=0).values
            hyper_df = top_shap_df[top_shap_df['methylation']>0.5]
            hypo_df = top_shap_df[top_shap_df['methylation']<0.5]
            methylation_shapley_data_dict['hyper'].top_cpgs['by_class'][class_name]['overall']=hyper_df[['cpg','shapley_value']]
            methylation_shapley_data_dict['hypo'].top_cpgs['by_class'][class_name]['overall']=hypo_df[['cpg','shapley_value']]
        return methylation_shapley_data_dict

    def add_bin_continuous_classes(self):
        """Bin continuous outcome variable and extract top positive and negative CpGs for each new class."""
        pass

    def return_cpg_sets(self):
        from functools import reduce
        def return_cpg_set(shap_df):
            return set(shap_df['cpg'].values.tolist())
        cpg_sets={}
        for class_name,individuals in self.list_individuals().items():
            cpg_set = [set(self.extract_class(class_name)['cpg'].values.tolist())]
            for individual in individuals:
                cpg_set.append(set(self.extract_individual(individual)[1]['cpg'].values.tolist()))
            cpg_set=set(reduce(lambda x,y: x.union(y),cpg_set))
            cpg_sets[class_name] = cpg_set
        cpg_exclusion_sets={}
        for class_name in cpg_sets:
            #print("Calculating exclusion set for {}".format(class_name))
            cpg_exclusion_sets[class_name]=set(reduce(lambda x,y:x.union(y),[cpg_set for class_name_query,cpg_set in cpg_sets.items() if class_name_query != class_name]))
        return cpg_sets, cpg_exclusion_sets

    def extract_class(self, class_name, class_intersect=False):
        if class_intersect:
            shap_dfs=[]
            for individual in self.shapley_data.top_cpgs['by_class'][class_name]['by_individual'].keys():
                shap_dfs.append(self.shapley_data.top_cpgs['by_class'][class_name]['by_individual'][individual].set_index('cpg'))
                df=pd.concat([shap_dfs],axis=1,join='inner')
                df['shapley_value']=df.values.sum(axis=1)
                df['cpg'] = np.array(list(df.index))
            return df[['cpg','shapley_value']]
        else:
            return self.shapley_data.top_cpgs['by_class'][class_name]['overall']

    def extract_individual(self, individual):
        class_name=self.indiv2class[individual]
        return class_name,self.shapley_data.top_cpgs['by_class'][class_name]['by_individual'][individual]

    def regenerate_individual_shap_values(self, n_top_cpgs):
        shapley_data = copy.deepcopy(self.shapley_data)
        for class_name in shapley_data.top_cpgs['by_class']:
            cpgs=np.array(list(shapley_data.shapley_values['by_class'][class_name]))
            shap_df=shapley_data.shapley_values['by_class'][class_name]
            shap_vals=shap_df.values
            class_importance_shaps = shap_vals.mean(0)
            top_idx = np.argsort(class_importance_shaps*-1)[:n_top_cpgs]
            shapley_data.top_cpgs['by_class'][class_name]['overall']=pd.DataFrame(np.hstack([cpgs[top_idx][:,np.newaxis],class_importance_shaps[top_idx][:,np.newaxis]]),columns=['cpg','shapley_value'])
            top_idxs = np.argsort(shap_vals*-1)[:,:n_top_cpgs]
            for i,individual in enumerate(shapley_data.top_cpgs['by_class'][class_name]['by_individual'].keys()):
                shapley_data.top_cpgs['by_class'][class_name]['by_individual'][individual]=pd.DataFrame(shap_df.iloc[i,top_idxs[i,:]].T.reset_index(drop=False).values,columns=['cpg','shapley_value'])
        return shapley_data

    def view_methylation(self, individual, methyl_arr):
        class_name,top_shap_df=self.extract_individual(individual)
        top_shap_df['methylation']=methyl_arr.beta.loc[individual,top_shap_df['cpg'].values].values
        return class_name,individual,top_shap_df

    def extract_methylation_array(self, methyl_arr, classes_only=True):
        from functools import reduce
        total_cpgs=[]
        for class_name in shapley_data.top_cpgs['by_class']:
            cpgs=np.array(list(shapley_data.shapley_values['by_class'][class_name]['overall'][:,0]))
            if not classes_only:
                cpgs = reduce(np.union1d,[cpgs]+[shapley_data.top_cpgs['by_class'][class_name]['by_individual'][individual][:,0]
                                          for individual in shapley_data.top_cpgs['by_class'][class_name]['by_individual'].keys()])
            total_cpgs.append(cpgs)
        all_cpgs = reduce(np.union1d,total_cpgs)
        methyl_arr.beta=methyl_arr.beta.loc[:,all_cpgs]
        return methyl_arr

    def limit_number_top_cpgs(self, n_top_cpgs):
        shapley_data = copy.deepcopy(self.shapley_data)
        if shapley_data.top_cpgs['overall']:
            shapley_data.top_cpgs['overall']=shapley_data.top_cpgs['overall'].iloc[:n_top_cpgs]
        for class_name in shapley_data.top_cpgs['by_class']:
            shapley_data.top_cpgs['by_class'][class_name]['overall']=shapley_data.top_cpgs['by_class'][class_name]['overall'].iloc[:n_top_cpgs]
            for individual in shapley_data.top_cpgs['by_class'][class_name]['by_individual']:
                shapley_data.top_cpgs['by_class'][class_name]['by_individual'][individual]=shapley_data.top_cpgs['by_class'][class_name]['by_individual'][individual].iloc[:n_top_cpgs]
        return shapley_data

    def list_individuals(self, return_list=False):
        if not return_list:
            individuals={class_name:list(self.shapley_data.top_cpgs['by_class'][class_name]['by_individual'].keys()) for class_name in self.shapley_data.top_cpgs['by_class']}
            return individuals
        else:
            from functools import reduce
            return list(reduce(lambda x,y:x+y, [list(self.shapley_data.top_cpgs['by_class'][class_name]['by_individual'].keys()) for class_name in self.shapley_data.top_cpgs['by_class']]))

    def list_classes(self):
        classes = list(self.shapley_data.top_cpgs['by_class'].keys())
        return classes

    def return_top_cpgs(self, classes=[], individuals=[], class_intersect=False, cpg_exclusion_sets=None, cpg_sets=None):
        top_cpgs={}
        if classes:
            for class_name in classes:
                top_cpg_df = self.extract_class(class_name, class_intersect)
                if cpg_exclusion_sets != None:
                    unique_cpgs = np.array(list(set(top_cpg_df['cpg'].values.tolist())-cpg_exclusion_sets[class_name]))
                    top_cpgs[class_name]= top_cpg_df[np.isin(top_cpg_df['cpg'].values,unique_cpgs)]
                else:
                    top_cpgs[class_name]= top_cpg_df
        if individuals:
            for indiv in individuals:
                class_name,top_cpg_df=self.extract_individual(indiv)
                if cpg_exclusion_sets != None:
                    unique_cpgs = np.array(list(set(top_cpg_df['cpg'].values.tolist())-cpg_exclusion_sets[class_name]))
                    print(len(top_cpg_df['cpg'].values),len(unique_cpgs))
                    top_cpgs['{}_{}'.format(class_name,indiv)]=top_cpg_df[np.isin(top_cpg_df['cpg'].values,unique_cpgs)]
                else:
                    top_cpgs['{}_{}'.format(class_name,indiv)]=top_cpg_df
        if cpg_sets != None:
            print(cpg_sets)
            intersected_cpgs=np.array(list(set.intersection(*list(cpg_sets.values()))))
            top_cpgs['intersection']=pd.DataFrame(intersected_cpgs[:,np.newaxis],columns=['cpg'])
            top_cpgs['intersection']['shapley_value']=-1
        return top_cpgs

    def jaccard_similarity_top_cpgs(self,class_names,individuals=False,overall=False, cooccurrence=False):
        from itertools import combinations
        from functools import reduce
        def jaccard_similarity(list1, list2):
            s1 = set(list1)
            s2 = set(list2)
            return len(s1.intersection(s2)) / len(s1.union(s2))
        def cooccurrence_fn(list1,list2):
            return len(set(list1).intersection(set(list2)))
        x={}
        if class_names[0]=='all':
            class_names=list(self.shapley_data.top_cpgs['by_class'].keys())
        for class_name in class_names:
            if overall:
                x[class_name]=self.shapley_data.top_cpgs['by_class'][class_name]['overall']['cpg'].values.tolist()
            if individuals:
                for indiv,df in list(self.shapley_data.top_cpgs['by_class'][class_name]['by_individual'].items()):
                    x['{}_{}'.format(class_name,indiv)]=df['cpg'].values.tolist()
        indivs=list(x.keys())
        similarity_matrix=pd.DataFrame(np.eye(len(x)),index=indivs,columns=indivs)
        for i,j in combinations(indivs,r=2):
            similarity_matrix.loc[i,j] = (round(jaccard_similarity(x[i],x[j]),3) if not cooccurrence else cooccurrence_fn(x[i],x[j]))
            similarity_matrix.loc[j,i] = similarity_matrix.loc[i,j]
        if cooccurrence:
            for i in indivs:
                similarity_matrix.loc[i,i]=len(x[i])
        return similarity_matrix

class PlotCircos:
    def __init__(self):
        from rpy2.robjects.packages import importr
        import rpy2.robjects as robjects
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()
        self.ggbio=importr('ggbio')
        self.data = robjects.r("""data("CRC", package = "biovizBase")""")
        self.hg19 = importr('IlluminaHumanMethylation450kanno.ilmn12.hg19')
        self.GRanges = importr('GenomicRanges')
        self.generate_base_plot = robjects.r("""function () {
                                p <- ggbio()
                                return(p)}""")
        self.add_plot_data = robjects.r("""function(p_data_new, p_data=c()) {
                                        p_data=c(p_data,p_data_new)
                                        return(p_data)
                                        }""")
        self.generate_final_plot = robjects.r("""function(p,grs) {
                                              p<- ggbio()
                                              for (gr in grs) {
                                              seqlevelsStyle(gr)<-"NCBI"
                                              p<-p+circle(gr, geom = "rect", color = "steelblue")
                                              }
                                               p<-p + circle(hg19sub,geom = "ideo", fill = "gray70") + circle(hg19sub,geom = "scale", size = 2)+
                                                circle(hg19sub,geom = "text", aes(label = seqnames), vjust = 0, size = 3)
                                               return(p)}""")

    def plot_cpgs(self, top_cpgs, output_dir='./'):
        import rpy2.robjects as robjects
        from rpy2.robjects import pandas2ri
        from collections import defaultdict
        pandas2ri.activate()
        self.top_cpgs = top_cpgs
        cpg_locations = pandas2ri.ri2py(robjects.r['as'](robjects.r('getAnnotation(IlluminaHumanMethylation450kanno.ilmn12.hg19)'),'data.frame')).set_index(keys='Name',drop=False)
        print(cpg_locations)
        cpgs_grange = robjects.r('makeGRangesFromDataFrame')(pandas2ri.py2ri(cpg_locations),start_field='pos',end_field='pos',starts_in_df_are_0based=False)
        output_dfs = {}
        plots={}
        p=self.generate_base_plot()
        #ggsave = robjects.r("""function (file.name, p) {ggsave(file.name,p)}""")
        for i,k in enumerate(list(self.top_cpgs.keys())):
            list_cpgs=np.squeeze(self.top_cpgs[k].values[:,0])
            cpg_location_subset = cpg_locations.loc[list_cpgs,:]
            location_subset = robjects.r('makeGRangesFromDataFrame')(pandas2ri.py2ri(cpg_location_subset),start_field='pos',end_field='pos',starts_in_df_are_0based=False)
            locations=self.add_plot_data(location_subset,locations) if i>0 else self.add_plot_data(location_subset)
            #self.add_plot(p,location_subset)
        self.generate_final_plot(p,locations)
        self.ggbio.ggsave(join(output_dir,'{}.png'.format('_'.join([k for k in self.top_cpgs]))))

class CpGExplainer: # consider shap.kmeans or grab representative sample of each outcome in training set for background ~ 39 * 2 samples, 39 cancers, should speed things up, small training set when building explainer https://github.com/slundberg/shap/issues/372
    def __init__(self,prediction_function=None, cuda=False):
        self.prediction_function=prediction_function
        self.cuda = cuda
        self.explainer=None

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

    def return_shapley_scores(self, test_methyl_array, n_samples, n_outputs, shap_sample_batch_size=None, top_outputs=None): # split up into multiple methods
        n_batch = 1
        if shap_sample_batch_size != None and self.method != 'deep': #  and not feature_selection
            n_batch = int(n_samples/shap_sample_batch_size)
            n_samples = shap_sample_batch_size
        test_arr = test_methyl_array.return_raw_beta_array()
        shap_values = np.zeros((top_outputs if top_outputs != None else n_outputs,)+test_arr.shape)
        additional_opts = {}
        if self.method != 'kernel':
            additional_opts['ranked_outputs']=top_outputs
            test_arr=to_tensor(test_arr) if not self.cuda else to_tensor(test_arr).cuda()
        for i in range(n_batch):
            click.echo("Batch {}".format(i))
            print("Batch {}".format(i))
            shap_values += return_shap_values(test_arr, self.explainer, self.method, n_samples, additional_opts)
        shap_values/=float(n_batch)

        self.shap_values = shap_values

    def classifier_assign_scores_to_shap_data(self, test_methyl_array, n_top_features, interest_col='disease', prediction_classes=None):

        cpgs=test_methyl_array.return_cpgs()
        shapley_data = ShapleyData()

        self.cpg_global_shapley_scores = np.abs(self.shap_values).mean(0).mean(0)
        shapley_data.add_global_importance(self.cpg_global_shapley_scores,cpgs, n_top_features)
        """if cross_class:
            self.shap_values = np.abs(self.shap_values).mean(axis=0)"""
        for i in range(self.shap_values.shape[0]):
            class_name = prediction_classes[i] if prediction_classes != None else str(i)
            shap_df = pd.DataFrame(self.shap_values[i,...],index=test_methyl_array.beta.index,columns=cpgs)
            if shap_df.shape[0]:
                if prediction_classes != None:
                    shap_df = shap_df.loc[test_methyl_array.pheno[interest_col].values == class_name,:]
                shapley_data.add_class(class_name, shap_df, cpgs, n_top_features)

        self.shapley_data = shapley_data

    def regressor_assign_scores_to_shap_data(self, test_methyl_array, n_top_features, cell_names=[]):
        n_classes = (self.shap_values.shape[0] if len(self.shap_values.shape) == 3 else 0)

        cpgs=test_methyl_array.return_cpgs()

        shapley_data = ShapleyData()

        self.cpg_global_shapley_scores = (np.abs(self.shap_values).mean(0).mean(0) if n_classes else np.abs(self.shap_values).mean(0))
        shapley_data.add_global_importance(self.cpg_global_shapley_scores,cpgs, n_top_features)
        if n_classes:
            for i in range(self.shap_values.shape[0]):
                class_name = str(i) if not cell_names else cell_names[i]
                shap_df = pd.DataFrame(self.shap_values[i,...],index=test_methyl_array.beta.index,columns=cpgs)
                if shap_df.shape[0]:
                    shapley_data.add_class('{}_pos'.format(class_name), shap_df, cpgs, n_top_features)
                    shapley_data.add_class('{}_neg'.format(class_name), shap_df, cpgs, n_top_features, add_top_negative=True)
        else: # regression tasks
            shap_df = pd.DataFrame(self.shap_values,index=test_methyl_array.beta.index,columns=cpgs)
            shapley_data.add_class('regression_pos' if not cell_names else '{}_pos'.format(cell_names[0]), shap_df, cpgs, n_top_features)
            shapley_data.add_class('regression_neg' if not cell_names else '{}_neg'.format(cell_names[0]), shap_df, cpgs, n_top_features, add_top_negative=True)
        self.shapley_data = shapley_data

    def feature_select(self, methyl_array, n_top_features):
        cpgs = methyl_array.return_cpgs()
        print(cpgs)
        cpgs=cpgs[np.argsort(self.cpg_global_shapley_scores*-1)[:n_top_features]]
        print(cpgs)
        return methyl_array.subset_cpgs(cpgs)

    def return_shapley_predictions(self, test_methyl_array, sample_name, interest_col, encoder=None):
        prediction_class = test_methyl_array['pheno'].loc[sample_name,interest_col]
        prediction_class_labelled = None
        if encoder != None:
            prediction_class_labelled = encoder.transform(prediction_class)
        return "In development"

    @classmethod
    def from_explainer(explainer, method, cuda):
        cpg_explainer = CpGExplainer(cuda=cuda)
        cpg_explainer.explainer = explainer
        return cpg_explainer


class BioInterpreter:
    def __init__(self, dict_top_cpgs):
        self.top_cpgs = dict_top_cpgs
        from rpy2.robjects.packages import importr
        self.hg19 = importr('IlluminaHumanMethylation450kanno.ilmn12.hg19')
        self.GRanges = importr('GenomicRanges')
        self.missMethyl=importr('missMethyl')
        self.limma=importr('limma')
        self.lola=importr('LOLA')
        importr('simpleCache')
        """if self.prediction_classes == None:
            self.prediction_classes = list(range(len(self.top_cpgs)))
        else:
            self.prediction_classes=list(map(lambda x: x.replace(' ',''),prediction_classes))"""

    def gometh(self, collection='GO', allcpgs=[], length_output=20, gsea_analyses=[], gsea_pickle=''):# consider turn into generator go or kegg # add rgreat, lola, roadmap-chromatin hmm, atac-seq, chip-seq, gometh, Hi-C, bedtools, Owen's analysis
        import rpy2.robjects as robjects
        from rpy2.robjects import pandas2ri
        #robjects.packages.importr('org.Hs.eg.db')
        pandas2ri.activate()
        output_dfs={}
        if allcpgs:
            allcpgs=robjects.vectors.StrVector(allcpgs)
        else:
            allcpgs=robjects.r('NULL')
        if gsea_analyses:
            gsea_collections=pickle.load(open(gsea_pickle,'rb'))
            gene_sets=[]
            for analysis in gsea_analyses:
                gene_sets.extend(list(gsea_collections[analysis].items()))
            gene_sets=robjects.ListVector(dict(gene_sets))
        for k in self.top_cpgs:
            list_cpgs=self.top_cpgs[k].values
            print('Start Prediction {} BIO'.format(k))
            list_cpgs=robjects.vectors.StrVector(list_cpgs[:,0].tolist())
            if collection == 'GENE':
                mappedEz = self.missMethyl.getMappedEntrezIDs(sig_cpg=list_cpgs, all_cpg = allcpgs, array_type='450K')
                gometh_output = robjects.r('function (mappedEz, length.output) {data.frame(mappedEz$sig.eg[1:length.output])}')(mappedEz,length_output) # sig.eg[1:10]
                print(gometh_output)
            elif collection in ['GO','KEGG']:
                gometh_output = self.missMethyl.gometh(sig_cpg=list_cpgs,all_cpg=allcpgs,collection=collection,prior_prob=True)
                gometh_output = self.limma.topKEGG(gometh_output, number=length_output) if collection=='KEGG' else self.limma.topGO(gometh_output, number=length_output)
            else:
                gometh_output = self.missMethyl.gsameth(sig_cpg=list_cpgs,all_cpg=allcpgs,collection=gene_sets,prior_prob=True)
                gometh_output = robjects.r('as.table')(self.missMethyl.topGSA(gometh_output, number=length_output))
                #robjects.r('print')(gometh_output)
            # FIXME add get genes
            # genes = self.missMethyl.getMappedEntrezIDs(sig.cpg, all.cpg = NULL, array.type, anno = NULL)
            output_dfs['prediction_{}'.format(k)]=pandas2ri.ri2py(robjects.r['as'](gometh_output,'data.frame')) if not gsea_analyses else pandas2ri.ri2py(robjects.r('as.data.frame')(gometh_output)).pivot(index='Var1',columns='Var2',values='Freq').sort_values('P.DE')
            print(output_dfs['prediction_{}'.format(k)].head())
            #print('GO/KEGG/GSEA Computed for Prediction {} Cpgs: {}'.format(k, ' '.join(list_cpgs)))
        return output_dfs

    def get_nearby_cpg_shapleys(self, all_cpgs, max_gap):
        import rpy2.robjects as robjects
        from rpy2.robjects import pandas2ri
        from collections import defaultdict
        pandas2ri.activate()
        cpg_locations = pandas2ri.ri2py(robjects.r['as'](robjects.r('getAnnotation(IlluminaHumanMethylation450kanno.ilmn12.hg19)'),'data.frame')).set_index(keys='Name',drop=False)
        cpgs_grange = robjects.r('makeGRangesFromDataFrame')(pandas2ri.py2ri(cpg_locations),start_field='pos',end_field='pos',starts_in_df_are_0based=False)
        output_dfs = {}
        for k in self.top_cpgs:
            cpg_dict = defaultdict()
            list_cpgs=self.top_cpgs[k].values[:,0]
            cpg_dict.update(self.top_cpgs[k].values.tolist())
            cpg_location_subset = cpg_locations.loc[list_cpgs,:]
            location_subset = robjects.r('makeGRangesFromDataFrame')(pandas2ri.py2ri(cpg_location_subset),start_field='pos',end_field='pos',starts_in_df_are_0based=False)
            cpgs_overlap = list(pandas2ri.ri2py(robjects.r['as'](self.GRanges.findOverlaps(cpgs_grange,location_subset,maxgap=max_gap, type='any'),'data.frame')).index)
            for cpg in cpgs_overlap:
                cpg_dict[cpg]
            top_cpgs=np.array(list(cpg_dict.items()))
            output_dfs['prediction_{}'.format(k)]=pd.DataFrame(top_cpgs,index_col=None,columns=['CpG','Shapley Value'])
        return output_dfs


    def run_lola(self, all_cpgs=[], lola_db='', cores=8, collections=[],depletion=False):
        import rpy2.robjects as robjects
        from rpy2.robjects import pandas2ri
        order_by_max_rnk=robjects.r("function (dt) {dt[order(meanRnk, decreasing=FALSE),]}")
        pandas2ri.activate()
        cpg_locations = pandas2ri.ri2py(robjects.r['as'](robjects.r('getAnnotation(IlluminaHumanMethylation450kanno.ilmn12.hg19)'),'data.frame')).set_index(keys='Name',drop=False)
        all_cpg_regions = robjects.r('makeGRangesFromDataFrame')(pandas2ri.py2ri(cpg_locations if not all_cpgs else cpg_locations.loc[all_cpgs,:]),start_field='pos',end_field='pos',starts_in_df_are_0based=False)
        #robjects.r('load("{}")'.format(lola_rdata))
        lolaDB = self.lola.loadRegionDB(lola_db,collections=(robjects.r('NULL') if not collections else rpy2.robjects.vectors.StrVector(collections)))#
        output_dfs={}
        for k in self.top_cpgs:
            list_cpgs=self.top_cpgs[k].values[:,0]
            cpg_location_subset = cpg_locations.loc[list_cpgs,:]
            location_subset = robjects.r('makeGRangesFromDataFrame')(pandas2ri.py2ri(cpg_location_subset),start_field='pos',end_field='pos',starts_in_df_are_0based=False)
            lola_output=self.lola.runLOLA(location_subset,all_cpg_regions,lolaDB,cores=cores,direction=('depletion' if depletion else 'enrichment'))
            output_dfs['prediction_{}'.format(k)]=pandas2ri.ri2py(robjects.r['as'](order_by_max_rnk(lola_output),'data.frame'))#.iloc[:20,:]
            print(output_dfs['prediction_{}'.format(k)].head())
        return output_dfs
        #https://academic.oup.com/bioinformatics/article/32/4/587/1743969

CONTEXT_SETTINGS = dict(help_option_names=['-h','--help'], max_content_width=90)

@click.group(context_settings= CONTEXT_SETTINGS)
@click.version_option(version='0.1')
def interpret():
    pass

def return_shap_values(test_arr, explainer, method, n_samples, additional_opts):
    if method == 'kernel' or method == 'gradient': # ranked_outputs=ranked_outputs, add if feature_selection
        svals=(explainer.shap_values(test_arr, nsamples=n_samples, **additional_opts)[0] if (method == 'gradient' and additional_opts['ranked_outputs'] != None) else explainer.shap_values(test_arr, nsamples=n_samples))
        return np.stack(svals,axis=0) if type(svals) == type([]) else svals
    else:
        return (explainer.shap_values(test_arr, **additional_opts)[0] if additional_opts['ranked_outputs'] !=None else explainer.shap_values(test_arr))

def to_tensor(arr):
    return Transformer().generate()(arr)

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

def plot_lola_output_(lola_csv, plot_output_dir, description_col, cell_types):
    os.makedirs(plot_output_dir,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import seaborn as sns
    import matplotlib.pyplot as plt
    # from rpy2.robjects.packages import importr
    # import rpy2.robjects as robjects
    # from rpy2.robjects import pandas2ri
    # pandas2ri.activate()
    # ggplot=importr("ggplot2")
    # importr("ggpubr")
    # forcats=importr('forcats')
    #
    # create_bar_chart=robjects.r("""function(df,fill.col=NULL){
    #                     ggboxplot(df, x = "description", y = "oddsRatio", fill = fill.col,#,
    #                       color = "white",           # Set bar border colors to white
    #                       palette = "jco",            # jco journal color palett. see ?ggpar
    #
    #                       orientation = "horiz")}           # Sort the value in dscending order
    #                       #sort.by.groups = F, #TRUE,      # Sort inside each group
    #                       # sort.val = "desc",
    #                       #)
    #                 #}""")

    def create_bar_chart(df, description='description', cell_type=None):
        sns.set(style="whitegrid")
        f, ax = plt.subplots(figsize=(10,7))
        sns.despine(bottom=True, left=True)
        sns.stripplot(x="oddsRatio", y=description, hue=cell_type,
              data=df, dodge=True, jitter=True,
              alpha=.25, zorder=1)
        sns.pointplot(x="oddsRatio", y=description, hue=cell_type,
              data=df, dodge=.532, join=False, palette="dark",
              markers="d", scale=.75, ci=None)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[3:], labels[3:], title="Cell Types",
                  handletextpad=0, columnspacing=1,
                  loc="lower right", ncol=3, frameon=True)
        ax.set_xlim(left=0)
        return f,ax

    lola_results=pd.read_csv(lola_csv)#[['collection','description','oddsRatio','cellType']]

    # filter cell types here
    collections_df=lola_results.groupby('collection')
    for name,df in collections_df:
        top_results_df=df.iloc[:min(25,df.shape[0]),:].reset_index(drop=True)

        #top_results_df=pandas2ri.py2ri(top_results_df)#pandas2ri.py2ri(top_results_df)
        #robjects.r("print")(top_results_df)
        #top_results_df=robjects.r("function(df){data.frame(df,stringsAsFactors=FALSE)}")(robjects.r("function(df){lapply(df,as.character)}")(top_results_df)) # FIX
        #print(top_results_df)
        """RESET FACTORS ^^^"""
        #diagnose_df=pandas2ri.ri2py(top_results_df)
        top_results_df=top_results_df.fillna('NULL')
        plt.figure()
        f,ax=create_bar_chart(top_results_df,description_col,'cellType')
        plt.tight_layout()
        plt.savefig(join(plot_output_dir,lola_csv.split('/')[-1].replace('.csv','_{}.png'.format(name))))
        plt.close()
        #ggplot.ggsave(join(plot_output_dir,lola_csv.split('/')[-1].replace('.csv','_{}.png'.format(name))))
    # add shore island breakdown
@interpret.command()
@click.option('-c', '--command', default='', help='Submit torque job.', show_default=True)
def produce_shapley_data_torque(command):
    import subprocess
    with open('pbs_interpret.sh','r') as f1, open('pbs_submit.sh','w') as f2:
        f2.write(f1.read().replace('COMMAND',command))
    subprocess.call("mksub {}".format('pbs_submit.sh'),shell=True)

@interpret.command() # FIXME add abs or just -1, positive contributions or any contributions
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
@click.option('-cc', '--cross_class', is_flag=True, help='Find importance of features for prediction across classes.')
@click.option('-fs', '--feature_selection', is_flag=True, help='Perform feature selection using top global SHAP scores.', show_default=True)
@click.option('-top', '--top_outputs', default=0, help='Get shapley values for fewer outputs if feature selection.', show_default=True)
@click.option('-vae', '--vae_interpret', is_flag=True, help='Use model to get shapley values for VAE latent dimensions, only works if proper datasets are set.', show_default=True)
@click.option('-cl', '--pred_class', default='', help='Prediction class top cpgs.', type=click.Path(exists=False), show_default=True)
@click.option('-r', '--results_csv', default='./predictions/results.csv', help='Remove all misclassifications.', type=click.Path(exists=False), show_default=True)
@click.option('-ind', '--individual', default='', help='One individual top cpgs.', type=click.Path(exists=False), show_default=True)
@click.option('-rc', '--residual_cutoff', default=0., multiple=True, help='Activate for regression interpretations. Standard deviation in residuals before removing.', show_default=True)
@click.option('-cn', '--cell_names', default=[''], multiple=True, help='Multioutput names for multi-output regression.', show_default=True)
def produce_shapley_data(train_pkl, val_pkl, test_pkl, model_pickle, n_workers, batch_size, cuda, n_samples, n_top_features, output_dir, method, shap_sample_batch_size, n_random_representative, interest_col, n_random_representative_test, categorical_encoder, cross_class, feature_selection, top_outputs, vae_interpret, pred_class, results_csv, individual, residual_cutoff, cell_names):
    os.makedirs(output_dir,exist_ok=True)
    if not pred_class:
        pred_class = None
    if not top_outputs or not feature_selection:
        top_outputs = None
    if os.path.exists(categorical_encoder):
        categorical_encoder=pickle.load(open(categorical_encoder,'rb'))
        prediction_classes=list(categorical_encoder.categories_[0])
    else:
        prediction_classes = None
    cell_names=list(filter(None,cell_names))
    train_methyl_array, val_methyl_array, test_methyl_array=MethylationArray.from_pickle(train_pkl), MethylationArray.from_pickle(val_pkl), MethylationArray.from_pickle(test_pkl)#preprocessed_methyl_array.subset_index(train_test_idx_dict['train']), preprocessed_methyl_array.subset_index(train_test_idx_dict['test'])
    #preprocessed_methyl_array=MethylationArray(*extract_pheno_beta_df_from_pickle_dict(input_dict))
    train_methyl_array = MethylationArrays([train_methyl_array,val_methyl_array]).combine()
    cpgs, train_samples= train_methyl_array.return_cpgs(), train_methyl_array.return_idx()
    cpgs, test_samples= test_methyl_array.return_cpgs(), test_methyl_array.return_idx()
    model = torch.load(model_pickle)
    if os.path.exists(results_csv):
        results_df=pd.read_csv(results_csv)
        if residual_cutoff:
            replace_dict={k:k.replace('_pred','') for k in list(results_df)}
            pred_df = results_df.iloc[:,np.vectorize(lambda x: x.endswith('_pred'))(list(results_df))].rename(replace_dict)
            true_df = results_df.iloc[:,np.vectorize(lambda x: x.endswith('_true'))(list(results_df))].rename(replace_dict)
            residual_df = pred_df-true_df
            cell_names=list(residual_df)
            for col in residual_df:
                residual_df.loc[:,col] = (residual_df[col] = residual_df[col].mean())/residual_df[col].std(ddof=0)
            residual_df = (results_df <= residual_cutoff)
            residual_df = residual_df.all(axis=1)
            test_methyl_array=test_methyl_array.subset_index(np.array(list(residual_df.index)))
        else:
            test_methyl_array=test_methyl_array.subset_index(np.array(list((results_df['y_pred']==results_df['y_true']).index)))
    if individual:
        test_methyl_array=test_methyl_array.subset_index(np.array([individual]))
    if n_random_representative and method != 'gradient':
        train_methyl_array = train_methyl_array.subsample(interest_col, n_samples=n_random_representative, categorical=model.categorical if 'categorical' in dir(model) else False)
    if n_random_representative_test:
        test_methyl_array = test_methyl_array.subsample(interest_col, n_samples=n_random_representative_test, categorical=model.categorical if 'categorical' in dir(model) else False)
    if 'categorical' in dir(model) and model.categorical:
        print("TRAIN:")
        train_methyl_array.categorical_breakdown(interest_col)
        print("TEST:")
        test_methyl_array.categorical_breakdown(interest_col)
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
    cpg_explainer.return_shapley_scores(test_methyl_array, n_samples, n_outputs=n_test_results_outputs, shap_sample_batch_size=shap_sample_batch_size, top_outputs=top_outputs)
    if not residual_cutoff:
        cpg_explainer.classifier_assign_scores_to_shap_data(test_methyl_array, n_top_features, interest_col=interest_col, prediction_classes=prediction_classes)
    else:
        cpg_explainer.regressor_assign_scores_to_shap_data(test_methyl_array, n_top_features, cell_names=cell_names)
    if feature_selection:
        print('FEATURE SELECT')
        feature_selected_methyl_array=cpg_explainer.feature_select(MethylationArrays([train_methyl_array,test_methyl_array]).combine(),n_top_features)
        feature_selected_methyl_array.write_pickle(join(output_dir, 'feature_selected_methyl_array.pkl'))
    else:
        output_all_cpgs=join(output_dir,'all_cpgs.p')
        shapley_output=join(output_dir,'shapley_data.p')
        pickle.dump(train_methyl_array.return_cpgs(),open(output_all_cpgs,'wb'))
        cpg_explainer.shapley_data.to_pickle(shapley_output)

@interpret.command()
@click.option('-o', '--output_dir', default='./lola_db/', help='Output directory for lola dbs.', type=click.Path(exists=False), show_default=True)
def grab_lola_db_cache(output_dir):
    os.makedirs(output_dir,exist_ok=True)
    core_dir = join(output_dir,'core')
    extended_dir = join(output_dir,'extended')
    subprocess.call("wget http://big.databio.org/regiondb/LOLACoreCaches_180412.tgz && wget http://big.databio.org/regiondb/LOLAExtCaches_170206.tgz && mv *.tgz {0} && tar -xvzf {0}/LOLAExtCaches_170206.tgz && mv {0}/scratch {1} && tar -xvzf {0}/LOLACoreCaches_180412.tgz && mv {0}/nm {2}".format(output_dir,extended_dir,core_dir),shell=True)

@interpret.command()
@click.option('-a', '--all_cpgs_pickle', default='./interpretations/shapley_explanations/all_cpgs.p', help='List of all cpgs used in shapley analysis.', type=click.Path(exists=False), show_default=True)
@click.option('-s', '--shapley_data_list', default=['./interpretations/shapley_explanations/shapley_data.p'], multiple=True, help='Pickle containing top CpGs.', type=click.Path(exists=False), show_default=True)
@click.option('-o', '--output_dir', default='./interpretations/biological_explanations/', help='Output directory for interpretations.', type=click.Path(exists=False), show_default=True)
@click.option('-w', '--analysis', default='GO', help='Choose biological analysis.', type=click.Choice(['GSEA','GO','KEGG','GENE', 'LOLA', 'NEAR_CPGS']), show_default=True)
@click.option('-n', '--n_workers', default=8, help='Number workers.', show_default=True)
@click.option('-l', '--lola_db', default='./lola_db/core/nm/t1/resources/regions/LOLACore/hg19/', help='LOLA region db.', type=click.Path(exists=False), show_default=True)
@click.option('-i', '--individuals', default=[''], multiple=True, help='Individuals to evaluate.', show_default=True)
@click.option('-c', '--classes', default=[''], multiple=True, help='Classes to evaluate.', show_default=True)
@click.option('-m', '--max_gap', default=1000, help='Genomic distance to search for nearby CpGs than found top cpgs shapleys.', show_default=True)
@click.option('-ci', '--class_intersect', is_flag=True, help='Compute class shapleys by intersection of individuals.', show_default=True)
@click.option('-lo', '--length_output', default=20, help='Number enriched terms to print.', show_default=True)
@click.option('-ss', '--set_subtraction', is_flag=True, help='Only consider CpGs relevant to particular class.', show_default=True)
@click.option('-int', '--intersection', is_flag=True, help='CpGs common to all classes.', show_default=True)
@click.option('-col', '--collections', default=[''], multiple=True, help='Lola collections.', type=click.Choice(['','cistrome_cistrome','codex','encode_tfbs,ucsc_features','cistrome_epigenome','encode_segmentation','sheffield_dnase','jaspar_motifs','roadmap_epigenomics']), show_default=True)
@click.option('-gsea', '--gsea_analyses', default=[''], multiple=True, help='Gene set enrichment analysis to choose from, if chosen will override other analysis options. http://software.broadinstitute.org/gsea/msigdb/collections.jsp', type=click.Choice(['','C1','C2','C2.KEGG,C2.REACTOME','C3','C3.TFT','C4.CGN','C4.CM','C4','C5','C6','C7','C8','H']), show_default=True)
@click.option('-gsp', '--gsea_pickle', default='./data/GSEA_GeneSets/gsea_collections.p', help='Gene set enrichment analysis to choose from.', type=click.Path(exists=False), show_default=True)
@click.option('-d', '--depletion', is_flag=True, help='Run depletion LOLA analysis instead of enrichment.', show_default=True)
def interpret_biology(all_cpgs_pickle,shapley_data_list,output_dir, analysis, n_workers, lola_db, individuals, classes, max_gap, class_intersect, length_output, set_subtraction, intersection, collections, gsea_analyses, gsea_pickle, depletion):
    """Add categorical encoder as custom input, then can use to change names of output csvs to match disease if encoder exists."""
    os.makedirs(output_dir,exist_ok=True)
    if os.path.exists(all_cpgs_pickle):
        all_cpgs=list(pickle.load(open(all_cpgs_pickle,'rb')))
    else:
        all_cpgs = []
    #top_cpgs=pickle.load(open(top_cpgs_pickle,'rb'))
    head_individuals=list(filter(None,individuals))
    head_classes=list(filter(None,classes))
    collections=list(filter(None,collections))
    gsea_analyses=list(filter(None,gsea_analyses))
    if gsea_analyses:
        analysis='GSEA'
    for i,shapley_data in enumerate(shapley_data_list):
        shapley_data=ShapleyData.from_pickle(shapley_data)
        shapley_data_explorer=ShapleyDataExplorer(shapley_data)
        if head_classes and head_classes[0]=='all':
            classes = shapley_data_explorer.list_classes()
        else:
            classes = copy.deepcopy(head_classes)
        if head_individuals and head_individuals[0]=='all':
            individuals = shapley_data_explorer.list_individuals(return_list=True)
        else:
            individuals = copy.deepcopy(head_individuals)
        cpg_exclusion_sets,cpg_sets=None,None
        if set_subtraction:
            _,cpg_exclusion_sets=shapley_data_explorer.return_cpg_sets()
        elif intersection:
            cpg_sets,_=shapley_data_explorer.return_cpg_sets()
        top_cpgs=shapley_data_explorer.return_top_cpgs(classes=classes,individuals=individuals,cpg_exclusion_sets=cpg_exclusion_sets,cpg_sets=cpg_sets)
        bio_interpreter = BioInterpreter(top_cpgs)
        if analysis in ['GO','KEGG','GENE']:
            analysis_outputs=bio_interpreter.gometh(analysis, allcpgs=[], length_output=length_output)
        elif analysis == 'LOLA':
            analysis_outputs=bio_interpreter.run_lola(all_cpgs=[], lola_db=lola_db, cores=n_workers, collections=collections, depletion=depletion)
        elif analysis == 'NEAR_CPGS':
            analysis_outputs=bio_interpreter.get_nearby_cpg_shapleys(all_cpgs=[],max_gap=max_gap, class_intersect=class_intersect)
        elif gsea_analyses:
            analysis_outputs=bio_interpreter.gometh('', allcpgs=[], length_output=length_output, gsea_analyses=gsea_analyses, gsea_pickle=gsea_pickle)
        for k in analysis_outputs:
            output_csv=join(output_dir,'{}_{}_{}.csv'.format(shapley_data_list[i].split('/')[-1],k,analysis if not gsea_analyses else '_'.join(gsea_analyses)).replace('/','-'))
            analysis_outputs[k].to_csv(output_csv)

@interpret.command()
@click.option('-s', '--shapley_data', default='./interpretations/shapley_explanations/shapley_data.p', help='Pickle containing top CpGs.', type=click.Path(exists=False), show_default=True)
@click.option('-c', '--classes_only', is_flag=True, help='Only take top CpGs from each class.', show_default=True)
@click.option('-o', '--output_dir', default='./interpretations/shapley_explanations/top_cpgs_extracted_methylarr/', help='Output directory for methylation array.', type=click.Path(exists=False), show_default=True)
@click.option('-t', '--test_pkl', default='./train_val_test_sets/test_methyl_array.pkl', help='Pickle containing testing set.', type=click.Path(exists=False), show_default=True)
def extract_methylation_array(shapley_data, classes_only, output_dir, test_pkl):
    os.makedirs(output_dir,exist_ok=True)
    shapley_data=ShapleyData.from_pickle(shapley_data)
    shapley_data_explorer=ShapleyDataExplorer(shapley_data)
    output_methyl_arr = shapley_data_explorer.extract_methylation_array(MethylationArray.from_pickle(test_pkl),classes_only)
    output_methyl_arr.write_pickle(os.join(output_dir,'extracted_methyl_arr.pkl'))
    output_methyl_arr.write_csvs(output_dir)

@interpret.command()
@click.option('-l', '--head_lola_dir', default='interpretations/biological_explanations/', help='Location of lola output csvs.', type=click.Path(exists=False), show_default=True)
@click.option('-o', '--plot_output_dir', default='./interpretations/biological_explanations/lola_plots/', help='Output directory for interpretations.', type=click.Path(exists=False), show_default=True)
@click.option('-d', '--description_col', default='description', help='Description column for categorical variables', type=click.Path(exists=False),show_default=True)
@click.option('-c', '--cell_types', default=[''], multiple=True, help='Cell types.', show_default=True)
@click.option('-ov', '--overwrite', is_flag=True, help='Overwrite existing plots.', show_default=True)
def plot_all_lola_outputs(head_lola_dir,plot_output_dir,description_col,cell_types,overwrite):
    import glob
    for lola_csv in glob.iglob(join(head_lola_dir,'**','*_LOLA.csv'),recursive=True):
        output_dir=join(plot_output_dir,lola_csv.split('/')[-2])
        if overwrite and len(glob.glob(join(output_dir,'{}*.png'.format(lola_csv.split('/')[-1].replace('.csv',''))))):
            continue
        else:
            plot_lola_output_(lola_csv, output_dir, description_col, cell_types)

@interpret.command()
@click.option('-l', '--lola_csv', default='', help='Location of lola output csv.', type=click.Path(exists=False), show_default=True)
@click.option('-o', '--plot_output_dir', default='./interpretations/biological_explanations/lola_plots/', help='Output directory for interpretations.', type=click.Path(exists=False), show_default=True)
@click.option('-d', '--description_col', default='description', help='Description column for categorical variables', type=click.Path(exists=False),show_default=True)
@click.option('-c', '--cell_types', default=[''], multiple=True, help='Cell types.', show_default=True)
def plot_lola_output(lola_csv, plot_output_dir, description_col, cell_types):
    plot_lola_output_(lola_csv, plot_output_dir, description_col, cell_types)

@interpret.command()
@click.option('-s', '--shapley_data', default='./interpretations/shapley_explanations/shapley_data.p', help='Pickle containing top CpGs.', type=click.Path(exists=False), show_default=True)
@click.option('-o', '--output_dir', default='./interpretations/shapley_explanations/shapley_data_by_methylation/', help='Output directory for cpg jaccard_stats.', type=click.Path(exists=False), show_default=True)
@click.option('-t', '--test_pkl', default='./train_val_test_sets/test_methyl_array.pkl', help='Pickle containing testing set.', type=click.Path(exists=False), show_default=True)
def split_hyper_hypo_methylation(shapley_data, output_dir, test_pkl):
    os.makedirs(output_dir,exist_ok=True)
    shapley_data=ShapleyData.from_pickle(shapley_data)
    shapley_data_explorer=ShapleyDataExplorer(shapley_data)
    test_methyl_array=MethylationArray.from_pickle(test_pkl)
    shap_data_methylation=shapley_data_explorer.return_shapley_data_by_methylation_status(test_methyl_array)
    shap_data_methylation['hypo'].to_pickle(join(output_dir,'hypo_shapley_data.p'))
    shap_data_methylation['hyper'].to_pickle(join(output_dir,'hyper_shapley_data.p'))

# create force plots for each sample??? use output shapley values and output explainer
@interpret.command()
@click.option('-s', '--shapley_data', default='./interpretations/shapley_explanations/shapley_data.p', help='Pickle containing top CpGs.', type=click.Path(exists=False), show_default=True)
@click.option('-c', '--class_names', default=[''], multiple=True, help='Class names.', show_default=True)
@click.option('-o', '--output_dir', default='./interpretations/shapley_explanations/top_cpgs_jaccard/', help='Output directory for cpg jaccard_stats.', type=click.Path(exists=False), show_default=True)
@click.option('-ov', '--overall', is_flag=True, help='Output overall similarity.', show_default=True)
@click.option('-i', '--include_individuals', is_flag=True, help='Output individuals.', show_default=True)
@click.option('-co', '--cooccurrence', is_flag=True, help='Output cooccurrence instead jaccard.', show_default=True)
def shapley_jaccard(shapley_data,class_names, output_dir, overall, include_individuals, cooccurrence):
    os.makedirs(output_dir,exist_ok=True)
    shapley_data=ShapleyData.from_pickle(shapley_data)
    shapley_data_explorer=ShapleyDataExplorer(shapley_data)
    outfilename=join(output_dir,'{}_jaccard.csv'.format('_'.join(class_names)))
    shapley_data_explorer.jaccard_similarity_top_cpgs(class_names,include_individuals,overall, cooccurrence).to_csv(outfilename)

@interpret.command()
@click.option('-s', '--shapley_data', default='./interpretations/shapley_explanations/shapley_data.p', help='Pickle containing top CpGs.', type=click.Path(exists=False), show_default=True)
@click.option('-i', '--individuals', default=[''], multiple=True, help='Individuals.', show_default=True)
@click.option('-o', '--output_dir', default='./interpretations/shapley_explanations/top_cpgs_methylation/', help='Output directory for cpg jaccard_stats.', type=click.Path(exists=False), show_default=True)
@click.option('-t', '--test_pkl', default='./train_val_test_sets/test_methyl_array.pkl', help='Pickle containing testing set.', type=click.Path(exists=False), show_default=True)
def view_methylation_top_cpgs(shapley_data,individuals, output_dir, test_pkl):
    os.makedirs(output_dir,exist_ok=True)
    shapley_data=ShapleyData.from_pickle(shapley_data)
    shapley_data_explorer=ShapleyDataExplorer(shapley_data)
    test_methyl_array=MethylationArray.from_pickle(test_pkl)
    for individual in individuals:
        class_name,individual,shap_df=shapley_data_explorer.view_methylation(individual=individual,methyl_arr=test_methyl_array)
        shap_df.to_csv(join(output_dir,"{}_{}.csv".format(class_name,individual)))

@interpret.command()
@click.option('-s', '--shapley_data', default='./interpretations/shapley_explanations/shapley_data.p', help='Pickle containing top CpGs.', type=click.Path(exists=False), show_default=True)
def list_individuals(shapley_data):
    shapley_data=ShapleyData.from_pickle(shapley_data)
    shapley_data_explorer=ShapleyDataExplorer(shapley_data)
    print(shapley_data_explorer.list_individuals())

@interpret.command()
@click.option('-s', '--shapley_data', default='./interpretations/shapley_explanations/shapley_data.p', help='Pickle containing top CpGs.', type=click.Path(exists=False), show_default=True)
def list_classes(shapley_data):
    shapley_data=ShapleyData.from_pickle(shapley_data)
    print(list(shapley_data.top_cpgs['by_class'].keys()))

@interpret.command()
@click.option('-s', '--shapley_data', default='./interpretations/shapley_explanations/shapley_data.p', help='Pickle containing top CpGs.', type=click.Path(exists=False), show_default=True)
@click.option('-nf', '--n_top_features', default=500, show_default=True, help='Top features to select for shap outputs.')
@click.option('-o', '--output_pkl', default='./interpretations/shapley_explanations/shapley_reduced_data.p', help='Pickle containing top CpGs, reduced number.', type=click.Path(exists=False), show_default=True)
def reduce_top_cpgs(shapley_data,n_top_features,output_pkl):
    os.makedirs(output_pkl[:output_pkl.rfind('/')],exist_ok=True)
    shapley_data=ShapleyData.from_pickle(shapley_data)
    shapley_data_explorer=ShapleyDataExplorer(shapley_data)
    shapley_data_explorer.limit_number_top_cpgs(n_top_cpgs=n_top_features).to_pickle(output_pkl)

@interpret.command()
@click.option('-s', '--shapley_data', default='./interpretations/shapley_explanations/shapley_data.p', help='Pickle containing top CpGs.', type=click.Path(exists=False), show_default=True)
@click.option('-nf', '--n_top_features', default=500, show_default=True, help='Top features to select for shap outputs.')
@click.option('-o', '--output_pkl', default='./interpretations/shapley_explanations/shapley_reduced_data.p', help='Pickle containing top CpGs, reduced number.', type=click.Path(exists=False), show_default=True)
def regenerate_top_cpgs(shapley_data,n_top_features,output_pkl):
    os.makedirs(output_pkl[:output_pkl.rfind('/')],exist_ok=True)
    shapley_data=ShapleyData.from_pickle(shapley_data)
    shapley_data_explorer=ShapleyDataExplorer(shapley_data)
    shapley_data_explorer.regenerate_individual_shap_values(n_top_cpgs=n_top_features).to_pickle(output_pkl)

@interpret.command()
@click.option('-s', '--shapley_data', default='./interpretations/shapley_explanations/shapley_data.p', help='Pickle containing top CpGs.', type=click.Path(exists=False), show_default=True)
@click.option('-o', '--output_dir', default='./interpretations/output_plots/', help='Output directory for output plots.', type=click.Path(exists=False), show_default=True)
@click.option('-i', '--individuals', default=[''], multiple=True, help='Individuals to evaluate.', show_default=True)
@click.option('-c', '--classes', default=[''], multiple=True, help='Classes to evaluate.', show_default=True)
def plot_top_cpgs(shapley_data,output_dir,individuals,classes):
    os.makedirs(output_dir,exist_ok=True)
    shapley_data=ShapleyData.from_pickle(shapley_data)
    shapley_data_explorer=ShapleyDataExplorer(shapley_data)
    individuals=list(filter(None,individuals))
    classes=list(filter(None,classes))
    if classes and classes[0]=='all':
        classes = shapley_data_explorer.list_classes()
    if individuals and individuals[0]=='all':
        individuals = shapley_data_explorer.list_individuals(return_list=True)
    cpg_exclusion_sets,cpg_sets=None,None
    set_subtraction,intersection=False,False
    if set_subtraction:
        _,cpg_exclusion_sets=shapley_data_explorer.return_cpg_sets()
    elif intersection:
        cpg_sets,_=shapley_data_explorer.return_cpg_sets()
    top_cpgs=shapley_data_explorer.return_top_cpgs(classes=classes,individuals=individuals,cpg_exclusion_sets=cpg_exclusion_sets,cpg_sets=cpg_sets)
    circos_plotter=PlotCircos()
    circos_plotter.plot_cpgs(top_cpgs,output_dir)

#################

if __name__ == '__main__':
    interpret()
