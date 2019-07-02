"""
interpretation_classes.py
=======================
Contains core classes and functions to extracting explanations for the predictions at single samples, and then interrogates important CpGs for biological plausibility.
"""

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
    """Store SHAP results that stores feature importances of CpGs on varying degrees granularity.

    Attributes
    ----------
    top_cpgs : type
        Quick accessible CpGs that have the n highest SHAP scores.
    shapley_values : type
        Storing all SHAPley values for CpGs for varying degrees granularity. For classification problems, this only includes SHAP scores particular to the actual class.

    """
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

    def add_class(self, class_name, shap_df, cpgs, n_top_cpgs, add_top_negative=False, add_top_abs=False):
        """Store SHAP scores for particular class to Shapley_Data class. Save feature importances on granular level, explanations for individual and aggregate predictions.

        Parameters
        ----------
        class_name : type
            Particular class name to be stored in dictionary structure.
        shap_df : type
            SHAP data to pull from.
        cpgs : type
            All CpGs.
        n_top_cpgs : type
            Number of top CpGs for saving n top CpGs.
        add_top_negative : type
            Only consider top negative scores.

        """
        signed = -1 if not add_top_negative else 1
        self.shapley_values['by_class'][class_name]=shap_df
        shap_vals=shap_df.values
        class_importance_shaps = shap_vals.mean(0)
        top_idx = np.argsort(class_importance_shaps*signed)[:n_top_cpgs]
        self.top_cpgs['by_class'][class_name]={'by_individual':{},'overall':{}}
        self.top_cpgs['by_class'][class_name]['overall']=pd.DataFrame(np.hstack([cpgs[top_idx][:,np.newaxis],class_importance_shaps[top_idx][:,np.newaxis]]),columns=['cpg','shapley_value'])
        top_idxs = np.argsort(shap_vals*signed if not add_top_abs else -np.abs(shap_vals))[:,:n_top_cpgs]
        for i,individual in enumerate(list(shap_df.index)):
            self.top_cpgs['by_class'][class_name]['by_individual'][individual]=pd.DataFrame(shap_df.iloc[i,top_idxs[i,:]].T.reset_index(drop=False).values,columns=['cpg','shapley_value'])

    def add_global_importance(self, global_importance_shaps, cpgs, n_top_cpgs):
        """Add overall feature importances, globally across all samples.

        Parameters
        ----------
        global_importance_shaps : type
            Overall SHAP values to be saved, aggregated across all samples.
        cpgs : type
            All CpGs.
        n_top_cpgs : type
            Number of top CpGs to rank and save as well.

        """

        self.shapley_values['overall']=pd.DataFrame(global_importance_shaps[:,np.newaxis],columns=['shapley_values'],index=cpgs)
        top_ft_idx=np.argsort(global_importance_shaps*-1)[:n_top_cpgs]
        self.top_cpgs['overall']=pd.DataFrame(np.hstack([cpgs[top_ft_idx][:,np.newaxis],global_importance_shaps[top_ft_idx][:,np.newaxis]]),columns=['cpg','shapley_value'])

    def to_pickle(self,output_pkl, output_dict=False):
        """Export Shapley data to pickle.

        Parameters
        ----------
        output_pkl : type
            Output file to save SHAPley data.

        """
        os.makedirs(output_pkl[:output_pkl.rfind('/')],exist_ok=True)
        if output_dict:
            pickle.dump(dict(shapley_values=self.shapley_values,top_cpgs=self.top_cpgs), open(output_pkl,'wb'))
        else:
            pickle.dump(self, open(output_pkl,'wb'))

    @classmethod
    def from_pickle(self,input_pkl, from_dict=False):
        """Load SHAPley data from pickle.

        Parameters
        ----------
        input_pkl : type
            Input pickle.

        """
        if from_dict:
            d=pickle.load(open(input_pkl,'rb'))
            shapley_data = ShapleyData()
            shapley_data.shapley_values=d['shapley_values']
            shapley_data.top_cpgs=d['top_cpgs']
            return shapley_data
        else:
            return pickle.load(open(input_pkl,'rb'))



class ShapleyDataExplorer:
    """Datatype used to explore saved ShapleyData.

    Parameters
    ----------
    shapley_data : type
        ShapleyData instance to be explored.

    Attributes
    ----------
    indiv2class : type
        Maps individuals to their classes used for quick look-up.
    shapley_data

    """
    def __init__(self, shapley_data):
        self.shapley_data=shapley_data
        self.indiv2class={}
        for class_name in self.shapley_data.top_cpgs['by_class']:
            for individual in self.shapley_data.top_cpgs['by_class'][class_name]['by_individual'].keys():
                self.indiv2class[individual]=class_name

    def return_shapley_data_by_methylation_status(self, methyl_array, threshold):
        """Return dictionary containing two SHAPley datasets, each split by low/high levels of methylation. Todo: Define this using median methylation value vs 0.5.

        Parameters
        ----------
        methyl_array : type
            MethylationArray instance.

        Returns
        -------
        dictionary
            Contains shapley data by methylation status.

        """
        threshold = {'mean':methyl_array.beta.mean().mean(),'original':0.5}[threshold]
        methylation_shapley_data_dict = {'hyper':copy.deepcopy(self.shapley_data),'hypo':copy.deepcopy(self.shapley_data)}
        for individual in self.list_individuals(return_list=True):
            class_name,individual,top_shap_df=self.view_methylation(individual,methyl_array)
            hyper_df = top_shap_df[top_shap_df['methylation']>=threshold]
            hypo_df = top_shap_df[top_shap_df['methylation']<threshold]
            methylation_shapley_data_dict['hyper'].top_cpgs['by_class'][class_name]['by_individual'][individual]=hyper_df[['cpg','shapley_value']]
            methylation_shapley_data_dict['hypo'].top_cpgs['by_class'][class_name]['by_individual'][individual]=hypo_df[['cpg','shapley_value']]
        for class_name,individuals in self.list_individuals().items():
            top_shap_df=self.extract_class(class_name)
            top_shap_df['methylation']=methyl_array.beta.loc[individuals,self.shapley_data.top_cpgs['by_class'][class_name]['overall']['cpg'].values].mean(axis=0).values
            hyper_df = top_shap_df[top_shap_df['methylation']>=threshold]
            hypo_df = top_shap_df[top_shap_df['methylation']<threshold]
            methylation_shapley_data_dict['hyper'].top_cpgs['by_class'][class_name]['overall']=hyper_df[['cpg','shapley_value']]
            methylation_shapley_data_dict['hypo'].top_cpgs['by_class'][class_name]['overall']=hypo_df[['cpg','shapley_value']]
        return methylation_shapley_data_dict

    def make_shap_scores_abs(self):
        for class_name in self.list_classes():
            self.shapley_data.shapley_values['by_class'][class_name] = self.shapley_data.shapley_values['by_class'][class_name].abs()

    def return_binned_shapley_data(self, original_class_name, outcome_col, add_top_negative=False):
        """Converts existing shap data based on continuous variable predictions into categorical variable.

        Parameters
        ----------
        original_class_name : type
            Regression results were split into ClassName_pos and ClassName_neg, what is the ClassName?
        outcome_col : type
            Feed in from pheno sheet one the column to bin samples on.
        add_top_negative : type
            Looking to include negative SHAPs?

        Returns
        -------
        ShapleyData
            With new class labels, built from regression results.

        """
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
        """WIP."""
        pass

    def return_cpg_sets(self):
        """Return top sets of CpGs from classes and individuals, and the complementary set. Returns dictionary of sets of CpGs and the union of all the other sets minus the current set.


        Returns
        -------
        cpg_sets
            Dictionary of individuals and classes; CpGs contained in set for particular individual/class
        cpg_exclusion_sets
            Dictionary of individuals and classes; CpGs not contained in set for particular individual/class
        """

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

    def extract_class(self, class_name, class_intersect=False, get_shap_values=False):
        """Extract the top cpgs from a class

        Parameters
        ----------
        class_name : type
            Class to extract from?
        class_intersect : type
            Bool to extract from aggregation of SHAP values from individuals of current class, should have been done already.

        Returns
        -------
        DataFrame
            Cpgs and SHAP Values
        """
        if get_shap_values:
            return self.shapley_data.shapley_values['by_class'][class_name].mean(axis=0)
        else:
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

    def extract_individual(self, individual, get_shap_values=False):
        """Extract the top cpgs from an individual

        Parameters
        ----------
        individual : type
            Individual to extract from?

        Returns
        -------
        Tuple
            Class name of individual, DataFrame of Cpgs and SHAP Values
        """
        class_name=self.indiv2class[individual]
        return class_name,(self.shapley_data.top_cpgs['by_class'][class_name]['by_individual'][individual] if not get_shap_values else self.shapley_data.shapley_values['by_class'][class_name].loc[individual])

    def regenerate_individual_shap_values(self, n_top_cpgs, abs_val=False, neg_val=False):
        """Use original SHAP scores to make nested dictionary of top CpGs based on shapley score, can do this for ABS SHAP or Negative SHAP scores as well.

        Parameters
        ----------
        n_top_cpgs : type
            Description of parameter `n_top_cpgs`.
        abs_val : type
            Description of parameter `abs_val`.
        neg_val : type
            Description of parameter `neg_val`.

        Returns
        -------
        type
            Description of returned object.

        """
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
        """Use MethylationArray to output top SHAP values for individual with methylation data appended to output data frame.

        Parameters
        ----------
        individual : type
            Individual to query from ShapleyData
        methyl_arr : type
            Input MethylationArray

        Returns
        -------
        class_name
        individual
        top_shap_df
            Top Shapley DataFrame with top cpgs for individual, methylation data appended.

        """

        class_name,top_shap_df=self.extract_individual(individual)
        #print(top_shap_df['cpg'].values,len(top_shap_df['cpg'].values))
        top_shap_df['methylation']=methyl_arr.beta.loc[individual,top_shap_df['cpg'].values].values
        return class_name,individual,top_shap_df

    def extract_methylation_array(self, methyl_arr, classes_only=True, global_vals=False, n_extract=1000, class_name=''):
        """Subset MethylationArray Beta values by some top SHAP CpGs from classes or overall.

        Parameters
        ----------
        methyl_arr : type
            MethylationArray.
        classes_only : type
            Setting this to False will include the overall top CpGs per class and the top CpGs for individuals in that class.
        global_vals : type
            Use top CpGs overall, across the entire dataset.
        n_extract : type
            Number of top CpGs to subset.
        class_name : type
            Which class to subset top CpGs from? Blank to use union of all top CpGs.

        Returns
        -------
        MethylationArray
            Stores methylation beta and pheno data, reduced here by queried CpGs.

        """
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
        """Reduce the number of top CpGs.

        Parameters
        ----------
        n_top_cpgs : type
            Number top CpGs to retain.

        Returns
        -------
        ShapleyData
            Returns shapley data with fewer top CpGs.

        """
        shapley_data = copy.deepcopy(self.shapley_data)
        if isinstance(shapley_data.top_cpgs['overall'],pd.DataFrame):
            shapley_data.top_cpgs['overall']=shapley_data.top_cpgs['overall'].iloc[:n_top_cpgs]
        for class_name in shapley_data.top_cpgs['by_class']:
            shapley_data.top_cpgs['by_class'][class_name]['overall']=shapley_data.top_cpgs['by_class'][class_name]['overall'].iloc[:n_top_cpgs]
            for individual in shapley_data.top_cpgs['by_class'][class_name]['by_individual']:
                shapley_data.top_cpgs['by_class'][class_name]['by_individual'][individual]=shapley_data.top_cpgs['by_class'][class_name]['by_individual'][individual].iloc[:n_top_cpgs]
        return shapley_data

    def list_individuals(self, return_list=False):
        """List the individuals in the ShapleyData object.

        Parameters
        ----------
        return_list : type
            Return list of individual names rather than a dictionary of class:individual key-value pairs.

        Returns
        -------
        List or Dictionary
            Class: Individual dictionary or individuals are elements of list

        """
        if not return_list:
            individuals={class_name:list(self.shapley_data.top_cpgs['by_class'][class_name]['by_individual'].keys()) for class_name in self.shapley_data.top_cpgs['by_class']}
            return individuals
        else:
            from functools import reduce
            return list(reduce(lambda x,y:x+y, [list(self.shapley_data.top_cpgs['by_class'][class_name]['by_individual'].keys()) for class_name in self.shapley_data.top_cpgs['by_class']]))

    def list_classes(self):
        """List classes in ShapleyData object.

        Returns
        -------
        List
            List of classes

        """
        classes = list(self.shapley_data.top_cpgs['by_class'].keys())
        return classes

    def return_top_cpgs(self, classes=[], individuals=[], class_intersect=False, cpg_exclusion_sets=None, cpg_sets=None):
        """Given list of classes and individuals, export a dictionary containing data frames of top CpGs and their SHAP scores.

        Parameters
        ----------
        classes : type
            Higher level classes to extract CpGs from, list of classes to extract top CpGs from.
        individuals : type
            Individual samples to extract top CpGs from.
        class_intersect : type
            Whether the top CpGs should be chosen by aggregating the remaining individual scores.
        cpg_exclusion_sets : type
            Dict of sets of CpGs, where these ones contain CpGs not particular to particular class.
        cpg_sets : type
            Contains top CpGs found for each class.

        Returns
        -------
        Dict
            Top CpGs accessed and returned for further processing.

        """
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
        """Return overall globally important CpGs.

        Returns
        -------
        list

        """
        return self.shapley_data.top_cpgs['overall']

    def jaccard_similarity_top_cpgs(self,class_names,individuals=False,overall=False, cooccurrence=False):
        """Calculate Jaccard Similarity matrix between classes and individuals within those classes based on how they share sets of CpGs.

        Parameters
        ----------
        class_names : type
            Classes to include.
        individuals : type
            Individuals to include.
        overall : type
            Whether to use overall class top CpGs versus aggregate.
        cooccurrence : type
            Output cooccurence instead of jaccard.

        Returns
        -------
        pd.DataFrame
            Similarity matrix.

        """
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
    """Jaccard score between two lists.

    Parameters
    ----------
    list1
    list2

    Returns
    -------
    float
        Jaccard similarity between elements in list.

    """
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))

def cooccurrence_fn(list1,list2):
    """Cooccurence of elements between two lists.

    Parameters
    ----------
    list1
    list2

    Returns
    -------
    float
        Cooccurence between elements in list.

    """
    return len(set(list1).intersection(set(list2)))

class PlotCircos:
    """Plot Circos Diagram using ggbio (regular circos software may be better)

    Attributes
    ----------
    generate_base_plot : type
        Function to generate base plot
    add_plot_data : type
        Function to add more plot data
    generate_final_plot : type
        Function to plot data using Circos

    """
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
        """Plot top CpGs location in genome.

        Parameters
        ----------
        top_cpgs : type
            Input dataframe of top CpG name and SHAP scores. Future: Plot SHAP scores in locations?
        output_dir : type
            Where to output plots.

        """
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

class CpGExplainer:
    """Produces SHAPley explanation scores for individual predictions, approximating a complex model with a basic linear one, one coefficient per CpG per individual.

    Parameters
    ----------
    prediction_function : type
        Model or function that makes predictions on the data, can be sklearn .predict method or pytorch forward pass, etc...
    cuda : type
        Run on GPUs?

    Attributes
    ----------
    explainer : type
        Type of SHAPley explanation method; kernel (uses LIME model agnostic explainer), deep (uses DeepLift) or Gradient (integrated gradients and backprop for explanations)?
    prediction_function
    cuda

    """

    # consider shap.kmeans or grab representative sample of each outcome in training set for background ~ 39 * 2 samples, 39 cancers, should speed things up, small training set when building explainer https://github.com/slundberg/shap/issues/372
    def __init__(self,prediction_function=None, cuda=False):
        self.prediction_function=prediction_function
        self.cuda = cuda
        self.explainer=None

    def build_explainer(self, train_methyl_array, method='kernel', batch_size=100): # can interpret latent dimensions
        """Builds SHAP explainer using background samples.

        Parameters
        ----------
        train_methyl_array : type
            Train Methylation Array from which to populate background samples to make estimated explanations.
        method : type
            SHAP explanation method?
        batch_size : type
            Break up prediction explanation creation into smaller batch sizes for lower memory consumption.
        """
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
        """Generate explanations for individual predictions, in the form of SHAPley scores for supplied test data.
        One SHAP score per individual prediction per CpG, and more if multiclass/output.
        For multiclass/multivariate outcomes, scores are generated for each outcome class.

        Parameters
        ----------
        test_methyl_array : type
            Testing MethylationArray.
        n_samples : type
            Number of SHAP score samples to produce. More SHAP samples provides convergence to correct score.
        n_outputs : type
            Number of outcome classes.
        shap_sample_batch_size : type
            If not None, break up SHAP score sampling into batches of this size to be averaged.
        top_outputs : type
            For deep explainer, limit number of output classes due to memory consumption.


        Returns
        -------
        type
            Description of returned object.

        """
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
        """Assigns the SHAP scores to a SHAPley data object, populating nested dictionaries (containing class-individual information) with SHAPley information and "top CpGs".

        Parameters
        ----------
        test_methyl_array : type
            Testing MethylationArray.
        n_top_features : type
            Number of top SHAP scores to use for a subdict called "top CpGs"
        interest_col : type
            Column in pheno sheet from which class names reside.
        prediction_classes : type
            User supplied prediction classes.

        """

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
        """Assigns the SHAP scores to a SHAPley data object, populating nested dictionaries (containing multioutput and single output regression-individual information) with SHAPley information and "top CpGs".

        Parameters
        ----------
        test_methyl_array : type
            Testing MethylationArray.
        n_top_features : type
            Number of top SHAP scores to use for a subdict called "top CpGs"
        cell_names : type
            If multi-output regression, create separate regression classes to access for all individuals.

        """
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
                    shapley_data.add_class(class_name, shap_df, cpgs, n_top_features, add_top_abs=True)
                    #shapley_data.add_class('{}_pos'.format(class_name), shap_df, cpgs, n_top_features)
                    #shapley_data.add_class('{}_neg'.format(class_name), shap_df, cpgs, n_top_features, add_top_negative=True)
        else: # regression tasks
            shap_df = pd.DataFrame(self.shap_values,index=test_methyl_array.beta.index,columns=cpgs)
            shapley_data.add_class('regression' if not cell_names else '{}'.format(cell_names[0]), shap_df, cpgs, n_top_features, add_top_abs=True)
            #shapley_data.add_class('regression_pos' if not cell_names else '{}_pos'.format(cell_names[0]), shap_df, cpgs, n_top_features)
            #shapley_data.add_class('regression_neg' if not cell_names else '{}_neg'.format(cell_names[0]), shap_df, cpgs, n_top_features, add_top_negative=True)
        self.shapley_data = shapley_data

    def feature_select(self, methyl_array, n_top_features):
        """Perform feature selection based on the best overall SHAP scores across all samples.

        Parameters
        ----------
        methyl_array : type
            MethylationArray to run feature selection on.
        n_top_features : type
            Number of CpGs to select.

        Returns
        -------
        MethylationArray
            Subsetted by top overall CpGs, overall positive contributions to prediction. May need to update.

        """
        cpgs = methyl_array.return_cpgs()
        print(cpgs)
        cpgs=cpgs[np.argsort(self.cpg_global_shapley_scores*-1)[:n_top_features]]
        print(cpgs)
        return methyl_array.subset_cpgs(cpgs)

    def return_shapley_predictions(self, test_methyl_array, sample_name, interest_col, encoder=None):
        """Method in development or may be deprecated."""
        prediction_class = test_methyl_array['pheno'].loc[sample_name,interest_col]
        prediction_class_labelled = None
        if encoder != None:
            prediction_class_labelled = encoder.transform(prediction_class)
        return "In development"

    @classmethod
    def from_explainer(explainer, method, cuda):
        """Load custom SHAPley explainer"""
        cpg_explainer = CpGExplainer(cuda=cuda)
        cpg_explainer.explainer = explainer
        return cpg_explainer


class BioInterpreter:
    """Interrogate CpGs found to be important from SHAP through enrichment and overlap tests.

    Parameters
    ----------
    dict_top_cpgs : type
        Dictionary of topCpGs output from ShapleyExplorer object

    Attributes
    ----------
    top_cpgs : type
        Dictionary of top cpgs to be interrogated.

    """
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
        """Run GO, KEGG, GSEA analyses or return nearby genes.

        Parameters
        ----------
        collection : type
            GO, KEGG, GSEA, GENE overlaps?
        allcpgs : type
            All CpGs defines CpG universe, empty if use all CpGs, smaller group of CpGs may yield more significant results.
        length_output : type
            How many lines should the output be, maximum number of results to print.
        gsea_analyses : type
            GSEA analyses/collections to target.
        gsea_pickle : type
            Location of gsea pickle containing gene sets, may need to run download_help_data to acquire.

        Returns
        -------
        Dict
            Dictionary containing results from each test run on all of the keys in top_cpgs.

        """
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
        """Query for CpGs that are within a max_gap of the topCpGs in the genome, helps find cpgs that could be highly correlated and thus also important.

        Parameters
        ----------
        all_cpgs : type
            List of all cpgs in methylationarray.
        max_gap : type
            Max radius to search for nearby CpGs.

        Returns
        -------
        Dict
            Results for each class/indiv's top CpGs, in the form of nearby CpGs to each of these CpGs sets.

        """
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
        """Perform overlap test, overlapping top CpGs with IDOL CpGs, age related cpgs, etc.

        Parameters
        ----------
        set_cpgs : type
            Set of reference cpgs.
        platform : type
            450K or 850K
        all_cpgs : type
            All CpGs to build a universe.
        output_csv : type
            Output CSV for results of overlaps between classes, how much doo they share these CpGs overlapped.
        extract_library : type
            Can extract a list of the CpGs that ended up beinng overlapped with.

        Returns
        -------
        List
            Optional output of overlapped library of CpGs.

        """
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
        """Run LOLA enrichment test.

        Parameters
        ----------
        all_cpgs : type
            CpG universe for LOLA.
        lola_db : type
            Location of LOLA database, can be downloaded using methylnet-interpret. Set to extended or core, and collections correspond to these.
        cores : type
            Number of cores to use.
        collections : type
            LOLA collections to run, leave empty to run all.
        depletion : type
            Set true to look for depleted regions over enriched.

        Returns
        -------
        type
            Description of returned object.

        """
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

def return_shap_values(test_arr, explainer, method, n_samples, additional_opts):
    """Return SHAP values, sampled a number of times, used in CpGExplainer class.

    Parameters
    ----------
    test_arr : type
        Testing MethylationArray.
    explainer : type
        SHAP explainer object.
    method : type
        Method of explaining.
    n_samples : type
        Number of samples to estimate SHAP scores.
    additional_opts : type
        Additional options to be passed into non-kernel/gradient methods.

    Returns
    -------
    np.array/list
        Shapley scores.

    """
    if method == 'kernel' or method == 'gradient': # ranked_outputs=ranked_outputs, add if feature_selection
        svals=(explainer.shap_values(test_arr, nsamples=n_samples, **additional_opts)[0] if (method == 'gradient' and additional_opts['ranked_outputs'] != None) else explainer.shap_values(test_arr, nsamples=n_samples))
        return np.stack(svals,axis=0) if type(svals) == type([]) else svals
    else:
        return (explainer.shap_values(test_arr, **additional_opts)[0] if additional_opts['ranked_outputs'] !=None else explainer.shap_values(test_arr))

def to_tensor(arr):
    """Turn np.array into tensor."""
    return Transformer().generate()(arr)

def return_predict_function(model, cuda):
    """Decorator to build the supplied prediction function, important for kernel explanations.

    Parameters
    ----------
    model : type
        Prediction function with predict method.
    cuda : type
        Run using cuda.

    Returns
    -------
    function
        Predict function.

    """

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
    """Decorator to build dataloader compatible with KernelExplainer for SHAP.

    Parameters
    ----------
    n_workers : type
        Number CPU.
    batch_size : type
        Batch size to load data into pytorch.

    Returns
    -------
    DataLoader
        Pytorch dataloader.

    """
    def construct_data_loader(raw_beta_array):
        raw_beta_dataset=RawBetaArrayDataSet(raw_beta_array,Transformer())
        raw_beta_dataloader=DataLoader(dataset=raw_beta_dataset,
            num_workers=n_workers,
            batch_size=batch_size,
            shuffle=False)
        return raw_beta_dataloader
    return construct_data_loader

def main_prediction_function(n_workers,batch_size, model, cuda):
    """Combine dataloader and prediction function into final prediction function that takes in tensor data, for Kernel Explanations.

    Parameters
    ----------
    n_workers : type
        Number of workers.
    batch_size : type
        Batch size for input data.
    model : type
        Model/predidction function.
    cuda : type
        Running on GPUs?

    Returns
    -------
    function
        Final prediction function.

    """
    dataloader_constructor=return_dataloader_construct(n_workers,batch_size)
    predict_function=return_predict_function(model, cuda)
    def main_predict(raw_beta_array):
        return predict_function(dataloader_constructor(raw_beta_array))
    return main_predict

def plot_lola_output_(lola_csv, plot_output_dir, description_col, cell_types):
    """Plots any LOLA output in the form of a forest plot.

    Parameters
    ----------
    lola_csv : type
        CSV containing lola results.
    plot_output_dir : type
        Plot output directory.
    description_col : type
        Column that will label the points on the plot.
    cell_types : type
        Column containing cell-types, colors plots and are rows to be compared.

    """
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

class DistanceMatrixCompute:
    """From any embeddings, calculate pairwise distances between classes that label embeddings.

    Parameters
    ----------
    methyl_array : MethylationArray
        Methylation array storing beta and pheno data.
    pheno_col : str
        Column name from which to extract sample names from and group by class.

    Attributes
    ----------
    col : pd.DataFrame
        Column of pheno array.
    classes : np.array
        Unique classes of pheno column.
    embeddings : pd.DataFrame
        Embeddings of beta values.
    """

    def __init__(self, methyl_array, pheno_col):
        self.embeddings = methyl_array.beta
        self.col = methyl_array.pheno[pheno_col]
        self.classes = self.col.unique()

    def compute_distances(self, metric='cosine', trim=0.):
        """Compute distance matrix between classes by average distances between points of classes.

        Parameters
        ----------
        metric : str
            Scikit-learn distance metric.

        """
        from itertools import combinations
        from sklearn.metrics import pairwise_distances
        from scipy.stats import trim_mean
        distance_calculator = lambda x,y: trim_mean(pairwise_distances(x,y,metric=metric).flatten(),trim)#.mean()
        self.distances = pd.DataFrame(0,index=self.classes,columns=self.classes)

        for i,j in combinations(self.classes,r=2):
            x1=self.embeddings.loc[self.col[self.col==i].index]
            x2=self.embeddings.loc[self.col[self.col==j].index]
            self.distances.loc[i,j]=distance_calculator(x1,x2)
            self.distances.loc[j,i]=self.distances.loc[i,j]

    def calculate_p_values(self):
        """Compute pairwise p-values between different clusters using manova."""
        from statsmodels.multivariate.manova import MANOVA
        from itertools import combinations
        from sklearn.preprocessing import OneHotEncoder
        test_id = 0 # returns wilk's lambda
        self.p_values = pd.DataFrame(1,index=self.classes,columns=self.classes)

        for i,j in combinations(self.classes,r=2):
            if i!=j:
                cluster_labels = self.col[np.isin(self.col,np.array([i,j]))]
                embeddings = np.array(self.embeddings.loc[cluster_labels.index].values)
                cluster_labels = OneHotEncoder().fit_transform(cluster_labels.values[:,np.newaxis]).todense().astype(int)
                #print(embeddings,cluster_labels)
                test_results = MANOVA(embeddings, cluster_labels)
                #print(test_results)
                p_val = test_results.mv_test().results['x0']['stat'].values[test_id, 4]
                self.p_values.loc[i,j]=p_val
                self.p_values.loc[j,i]=self.p_values.loc[i,j]
                #cluster_labels = cluster_labels.map({v:k for k,v in enumerate(self.col.unique().tolist())})


    def return_distances(self):
        """Return the distance matrix

        Returns
        -------
        pd.DataFrame
            Distance matrix between classes.

        """
        return self.distances

    def return_p_values(self):
        """Return the distance matrix

        Returns
        -------
        pd.DataFrame
            MANOVA values between classes.

        """
        return self.p_values
