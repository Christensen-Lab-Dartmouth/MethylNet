import shap, numpy as np, pandas as pd
import torch
from methylnet.datasets import RawBetaArrayDataSet, Transformer
from torch.autograd import Variable
from torch.utils.data import DataLoader
from os.path import join
import click
import os, copy
import pickle
from pymethylprocess.MethylationDataTypes import MethylationArray, MethylationArrays, extract_pheno_beta_df_from_pickle_dict


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
            hyper_df = top_shap_df[top_shap_df['methylation']>=0.5]
            hypo_df = top_shap_df[top_shap_df['methylation']<0.5]
            methylation_shapley_data_dict['hyper'].top_cpgs['by_class'][class_name]['by_individual'][individual]=hyper_df[['cpg','shapley_value']]
            methylation_shapley_data_dict['hypo'].top_cpgs['by_class'][class_name]['by_individual'][individual]=hypo_df[['cpg','shapley_value']]
        for class_name,individuals in self.list_individuals().items():
            top_shap_df=self.extract_class(class_name)
            top_shap_df['methylation']=methyl_array.beta.loc[individuals,self.shapley_data.top_cpgs['by_class'][class_name]['overall']['cpg'].values].mean(axis=0).values
            hyper_df = top_shap_df[top_shap_df['methylation']>=0.5]
            hypo_df = top_shap_df[top_shap_df['methylation']<0.5]
            methylation_shapley_data_dict['hyper'].top_cpgs['by_class'][class_name]['overall']=hyper_df[['cpg','shapley_value']]
            methylation_shapley_data_dict['hypo'].top_cpgs['by_class'][class_name]['overall']=hypo_df[['cpg','shapley_value']]
        return methylation_shapley_data_dict

    def return_binned_shapley_data(self, original_class_name, outcome_col, add_top_negative=False):
        """Converts existing shap data into categorical variable"""
        class_names = outcome_col.unique()
        new_shapley_data = ShapleyData()
        new_shapley_data.shapley_values['overall']=self.shapley_data.shapley_values['overall']
        cpgs = new_shapley_data.shapley_values['overall'].index.values
        new_shapley_data.top_cpgs['overall']=self.shapley_data.top_cpgs['overall']
        n_top_cpgs = new_shapley_data.top_cpgs['overall'].shape[0]
        regression_classes = self.shapley_data.shapley_values['by_class'].keys()
        for regression_class in regression_classes:
            print(regression_class,original_class_name)
            if regression_class.startswith(original_class_name):
                shap_df = self.shapley_data.shapley_values['by_class'][regression_class]#['by_class'][regression_class]
                print(shap_df)
                for class_name in class_names:
                    print(class_name)
                    print(outcome_col[outcome_col==class_name])
                    idx = outcome_col[outcome_col==class_name].index.values
                    new_shapley_data.add_class(class_name,shap_df.loc[idx,:],cpgs,n_top_cpgs, add_top_negative)
        return new_shapley_data

    def add_bin_continuous_classes(self):
        """Bin continuous outcome variable and extract top positive and negative CpGs for each new class."""
        pass

    def add_abs_value_classes(self):
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

    def regenerate_individual_shap_values(self, n_top_cpgs, abs_val=False, neg_val=False):
        shapley_data = copy.deepcopy(self.shapley_data)
        mod_name = lambda name: (name.replace('_pos','').replace('_neg','') if abs_val else name)
        abs_val = lambda x: (x if not abs_val else np.abs(x))
        neg_val = (1 if neg_val else -1)
        for class_name in list(shapley_data.top_cpgs['by_class'].keys()):
            new_class_name = mod_name(class_name)
            cpgs=np.array(list(shapley_data.shapley_values['by_class'][class_name]))
            shap_df=shapley_data.shapley_values['by_class'][class_name]
            shap_vals=abs_val(shap_df.values)
            class_importance_shaps = abs_val(shap_vals).mean(0)
            top_idx = np.argsort(abs_val(class_importance_shaps)*neg_val)[:n_top_cpgs]
            shapley_data.top_cpgs['by_class'][class_name]['overall']=pd.DataFrame(np.hstack([cpgs[top_idx][:,np.newaxis],class_importance_shaps[top_idx][:,np.newaxis]]),columns=['cpg','shapley_value'])
            top_idxs = np.argsort(abs_val(shap_vals)*neg_val)[:,:n_top_cpgs]
            for i,individual in enumerate(shapley_data.top_cpgs['by_class'][class_name]['by_individual'].keys()):
                new_indiv_name = mod_name(individual)
                shapley_data.top_cpgs['by_class'][class_name]['by_individual'][new_indiv_name]=pd.DataFrame(shap_df.iloc[i,top_idxs[i,:]].T.reset_index(drop=False).values,columns=['cpg','shapley_value'])
                if new_indiv_name != individual:
                    del shapley_data.top_cpgs['by_class'][class_name]['by_individual'][individual]
            shapley_data.top_cpgs['by_class'][new_class_name]=shapley_data.top_cpgs['by_class'][class_name]
            shapley_data.shapley_values['by_class'][new_class_name]=shapley_data.shapley_values['by_class'][class_name]
            if new_class_name != class_name:
                del shapley_data.top_cpgs['by_class'][class_name], shapley_data.shapley_values['by_class'][class_name]
        return shapley_data

    def view_methylation(self, individual, methyl_arr):
        class_name,top_shap_df=self.extract_individual(individual)
        top_shap_df['methylation']=methyl_arr.beta.loc[individual,top_shap_df['cpg'].values].values
        return class_name,individual,top_shap_df

    def extract_methylation_array(self, methyl_arr, classes_only=True, global_vals=False, n_extract=1000, class_name=''):
        from functools import reduce
        total_cpgs=[]
        if global_vals:
            all_cpgs=self.shapley_data.top_cpgs['overall']['cpg'].values
            if n_extract < len(all_cpgs):
                all_cpgs=all_cpgs[:n_extract]
        else:
            if class_name:
                class_names = [class_name]
            else:
                class_names = list(self.shapley_data.top_cpgs['by_class'].keys())
            for class_name in class_names:
                cpgs=np.array(list(self.shapley_data.top_cpgs['by_class'][class_name]['overall'].values[:,0]))
                if n_extract < len(cpgs):
                    cpgs=cpgs[:n_extract]
                if not classes_only:
                    cpgs = reduce(np.union1d,[cpgs]+[self.shapley_data.top_cpgs['by_class'][class_name]['by_individual'][individual].values[:,0]
                                              for individual in self.shapley_data.top_cpgs['by_class'][class_name]['by_individual'].keys()])
                total_cpgs.append(cpgs)
            all_cpgs = reduce(np.union1d,total_cpgs)
        methyl_arr.beta=methyl_arr.beta.loc[:,all_cpgs]
        return methyl_arr

    def limit_number_top_cpgs(self, n_top_cpgs):
        shapley_data = copy.deepcopy(self.shapley_data)
        if isinstance(shapley_data.top_cpgs['overall'],pd.DataFrame):
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

    def return_global_importance_cpgs(self):
        return self.shapley_data.top_cpgs['overall']

    def jaccard_similarity_top_cpgs(self,class_names,individuals=False,overall=False, cooccurrence=False):
        from itertools import combinations
        from functools import reduce
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

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))

def cooccurrence_fn(list1,list2):
    return len(set(list1).intersection(set(list2)))

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
        self.flow=importr('FlowSorted.Blood.EPIC')
        self.cgager=importr('cgageR')

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

    def return_overlap_score(self, set_cpgs='IDOL', platform='450k', all_cpgs=[], output_csv='output_bio_intersect.csv', extract_library=False):
        import rpy2.robjects as robjects
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()
        if set_cpgs == 'IDOL':
            cpg_reference_set=set(robjects.r('IDOLOptimizedCpGs') if platform=='epic' else robjects.r('IDOLOptimizedCpGs450klegacy'))
        elif set_cpgs == 'clock':
            cpg_reference_set=set(open('data/age_cpgs.txt').read().splitlines()) # clock
        elif set_cpgs == 'epitoc':
            cpg_reference_set=set(robjects.r('EpiTOCcpgs'))
        elif set_cpgs == 'horvath':
            cpg_reference_set=set(robjects.r('as.vector(HorvathLongCGlist$MR_var)'))
        elif set_cpgs == 'hannum':
            cpg_reference_set=set(robjects.r('as.vector(hannumModel$marker)'))
        else:
            cpg_reference_set=set(robjects.r('as.vector(HorvathLongCGlist$MR_var)'))
        if all_cpgs:
            cpg_reference_set = cpg_reference_set.intersection(all_cpgs)
        intersected_cpg_sets = {}
        for k in self.top_cpgs:
            query_cpgs=set(self.top_cpgs[k].values[:,0])
            intersected_cpgs=query_cpgs.intersection(cpg_reference_set)
            intersected_cpg_sets[k]=intersected_cpgs
            overlap_score = round(float(len(intersected_cpgs)) / len(cpg_reference_set) * 100,2)
            print("{} top cpgs overlap with {}% of {} cpgs".format(k,overlap_score,set_cpgs))

        keys=list(intersected_cpg_sets.keys())
        if len(keys)>1:
            from itertools import combinations
            from functools import reduce
            similarity_matrix=pd.DataFrame(np.eye(len(keys)),index=keys,columns=keys)
            for i,j in combinations(keys,r=2):
                similarity_matrix.loc[i,j]=len(intersected_cpg_sets[i].intersection(intersected_cpg_sets[j]))
                similarity_matrix.loc[j,i]=similarity_matrix.loc[i,j]
            for k in keys:
                similarity_matrix.loc[k,k]=len(intersected_cpg_sets[k])
                print("{} shared cpgs: {}/{}".format(k,len(set(reduce(lambda x,y:x.union(y),[intersected_cpg_sets[k].intersection(intersected_cpgs) for k2,intersected_cpgs in intersected_cpg_sets.items() if k2 != k]))),similarity_matrix.loc[k,k]))
            similarity_matrix.loc[np.sort(keys),np.sort(keys)].to_csv(output_csv)
            if extract_library:
                return reduce(np.union1d,[list(intersected_cpg_sets[k]) for k in keys])
        return None

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
