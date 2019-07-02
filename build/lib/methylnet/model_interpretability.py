#https://www.bioconductor.org/packages/devel/bioc/vignettes/missMethyl/inst/doc/missMethyl.html#gene-ontology-analysis
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
from methylnet.interpretation_classes import *


# after attaching classifier to VAE

# use SHAP to extract CpGs

# use gometh on extracted CpGs for each class in classifier!

# output to file and analyze


CONTEXT_SETTINGS = dict(help_option_names=['-h','--help'], max_content_width=90)

@click.group(context_settings= CONTEXT_SETTINGS)
@click.version_option(version='0.1')
def interpret():
    pass

@interpret.command() # FIXME add abs or just -1, positive contributions or any contributions
@click.option('-i', '--train_pkl', default='./train_val_test_sets/train_methyl_array.pkl', help='Input database for beta and phenotype data. Use ./predictions/vae_mlp_methyl_arr.pkl or ./embeddings/vae_methyl_arr.pkl for vae interpretations.', type=click.Path(exists=False), show_default=True)
@click.option('-v', '--val_pkl', default='./train_val_test_sets/val_methyl_array.pkl', help='Val database for beta and phenotype data. Use ./predictions/vae_mlp_methyl_arr.pkl or ./embeddings/vae_methyl_arr.pkl for vae interpretations.', type=click.Path(exists=False), show_default=True)
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
@click.option('-dsd', '--dump_shap_data', is_flag=True, help='Dump raw SHAP data to numpy file, warning, may be large file.', show_default=True)
def produce_shapley_data(train_pkl, val_pkl, test_pkl, model_pickle, n_workers, batch_size, cuda, n_samples, n_top_features, output_dir, method, shap_sample_batch_size, n_random_representative, interest_col, n_random_representative_test, categorical_encoder, cross_class, feature_selection, top_outputs, vae_interpret, pred_class, results_csv, individual, residual_cutoff, cell_names, dump_shap_data):
    """Explanations (coefficients for CpGs) for every individual prediction.
    Produce SHAPley scores for each CpG for each individualized prediction, and then aggregate them across coarser classes.
    Store CpGs with top SHAPley scores as well for quick access.
    Store in Shapley data object."""
    os.makedirs(output_dir,exist_ok=True)
    if not pred_class:
        pred_class = None
    if not top_outputs or not feature_selection:
        top_outputs = None
    if os.path.exists(categorical_encoder):
        categorical_encoder=pickle.load(open(categorical_encoder,'rb'))
        if categorical_encoder:
            prediction_classes=list(categorical_encoder.categories_[0])
        else:
            prediction_classes = None
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
                residual_df.loc[:,col] = (residual_df[col] - residual_df[col].mean())/residual_df[col].std(ddof=0)
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
    if dump_shap_data:
        np.save(file='raw_shap_data.npy',arr=cpg_explainer.shap_values)
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
    """Download core and extended LOLA databases for enrichment tests."""
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
@click.option('-gsp', '--gsea_pickle', default='./data/gsea_collections.p', help='Gene set enrichment analysis to choose from.', type=click.Path(exists=False), show_default=True)
@click.option('-d', '--depletion', is_flag=True, help='Run depletion LOLA analysis instead of enrichment.', show_default=True)
@click.option('-ov', '--overlap_test', is_flag=True, help='Run overlap test with IDOL library instead of all other tests.', show_default=True)
@click.option('-cgs', '--cpg_set', default='IDOL', help='Test for clock or IDOL enrichments.', type=click.Choice(['IDOL','epitoc','horvath','hannum']), show_default=True)
@click.option('-g', '--top_global', is_flag=True, help='Look at global top cpgs, overwrites classes and individuals.', show_default=True)
@click.option('-ex', '--extract_library', is_flag=True, help='Extract a library of cpgs to subset future array for inspection or prediction tests.', show_default=True)
def interpret_biology(all_cpgs_pickle,shapley_data_list,output_dir, analysis, n_workers, lola_db, individuals, classes, max_gap, class_intersect, length_output, set_subtraction, intersection, collections, gsea_analyses, gsea_pickle, depletion, overlap_test, cpg_set, top_global, extract_library):
    """Interrogate CpGs with high SHAPley scores for individuals, classes, or overall, for enrichments, genes, GO, KEGG, LOLA, overlap with popular cpg sets, GSEA, find nearby cpgs to top."""
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
        if not top_global:
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
        else:
            top_cpgs = {'global':shapley_data_explorer.return_global_importance_cpgs()}
        bio_interpreter = BioInterpreter(top_cpgs)
        if overlap_test:
            output_library_cpgs=bio_interpreter.return_overlap_score(set_cpgs=cpg_set,all_cpgs=all_cpgs,output_csv=join(output_dir,'{}_overlaps.csv'.format(cpg_set)),extract_library=extract_library)
            if output_library_cpgs is not None:
                pickle.dump(output_library_cpgs,open(join(output_dir,'cpg_library.pkl'),'wb'))
        else:
            if analysis in ['GO','KEGG','GENE']:
                analysis_outputs=bio_interpreter.gometh(analysis, allcpgs=all_cpgs, length_output=length_output)
            elif analysis == 'LOLA':
                analysis_outputs=bio_interpreter.run_lola(all_cpgs=all_cpgs, lola_db=lola_db, cores=n_workers, collections=collections, depletion=depletion)
            elif analysis == 'NEAR_CPGS':
                analysis_outputs=bio_interpreter.get_nearby_cpg_shapleys(all_cpgs=all_cpgs,max_gap=max_gap, class_intersect=class_intersect)
            elif gsea_analyses:
                analysis_outputs=bio_interpreter.gometh('', allcpgs=all_cpgs, length_output=length_output, gsea_analyses=gsea_analyses, gsea_pickle=gsea_pickle)
            for k in analysis_outputs:
                output_csv=join(output_dir,'{}_{}_{}.csv'.format(shapley_data_list[i].split('/')[-1],k,analysis if not gsea_analyses else '_'.join(gsea_analyses)).replace('/','-'))
                analysis_outputs[k].to_csv(output_csv)

@interpret.command()
@click.option('-s', '--shapley_data', default='./interpretations/shapley_explanations/shapley_data.p', help='Pickle containing top CpGs.', type=click.Path(exists=False), show_default=True)
@click.option('-co', '--classes_only', is_flag=True, help='Only take top CpGs from each class.', show_default=True)
@click.option('-o', '--output_dir', default='./interpretations/shapley_explanations/top_cpgs_extracted_methylarr/', help='Output directory for methylation array.', type=click.Path(exists=False), show_default=True)
@click.option('-t', '--test_pkl', default='./train_val_test_sets/test_methyl_array.pkl', help='Pickle containing testing set.', type=click.Path(exists=False), show_default=True)
@click.option('-c', '--col', default='', help='Column to color for output csvs.', show_default=True)
@click.option('-g', '--global_vals', is_flag=True, help='Only take top CpGs globally.', show_default=True)
@click.option('-n', '--n_extract', default=1000, help='Number cpgs to extract.', show_default=True)
def extract_methylation_array(shapley_data, classes_only, output_dir, test_pkl, col, global_vals, n_extract):
    """Subset and write methylation array using top SHAP cpgs."""
    os.makedirs(output_dir,exist_ok=True)
    shapley_data=ShapleyData.from_pickle(shapley_data)
    shapley_data_explorer=ShapleyDataExplorer(shapley_data)
    output_methyl_arr = shapley_data_explorer.extract_methylation_array(MethylationArray.from_pickle(test_pkl),classes_only,global_vals=global_vals,n_extract=n_extract)
    output_methyl_arr.write_pickle(os.path.join(output_dir,'extracted_methyl_arr.pkl'))
    if col:
        output_methyl_arr.beta[col] = output_methyl_arr.pheno[col]
    output_methyl_arr.write_csvs(output_dir)

@interpret.command()
@click.option('-l', '--head_lola_dir', default='interpretations/biological_explanations/', help='Location of lola output csvs.', type=click.Path(exists=False), show_default=True)
@click.option('-o', '--plot_output_dir', default='./interpretations/biological_explanations/lola_plots/', help='Output directory for interpretations.', type=click.Path(exists=False), show_default=True)
@click.option('-d', '--description_col', default='description', help='Description column for categorical variables', type=click.Path(exists=False),show_default=True)
@click.option('-c', '--cell_types', default=[''], multiple=True, help='Cell types.', show_default=True)
@click.option('-ov', '--overwrite', is_flag=True, help='Overwrite existing plots.', show_default=True)
def plot_all_lola_outputs(head_lola_dir,plot_output_dir,description_col,cell_types,overwrite):
    """Iterate through all LOLA csv results and plot forest plots of all enrichment scores."""
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
    """Plot LOLA results via forest plot for one csv file."""
    plot_lola_output_(lola_csv, plot_output_dir, description_col, cell_types)

@interpret.command()
@click.option('-s', '--shapley_data', default='./interpretations/shapley_explanations/shapley_data.p', help='Pickle containing top CpGs.', type=click.Path(exists=False), show_default=True)
@click.option('-o', '--output_dir', default='./interpretations/shapley_explanations/shapley_data_by_methylation/', help='Output directory for hypo/hyper shap data.', type=click.Path(exists=False), show_default=True)
@click.option('-t', '--test_pkl', default='./train_val_test_sets/test_methyl_array.pkl', help='Pickle containing testing set.', type=click.Path(exists=False), show_default=True)
@click.option('-thr', '--threshold', default='original', help='Pickle containing testing set.', type=click.Choice(['original','mean']), show_default=True)
def split_hyper_hypo_methylation(shapley_data, output_dir, test_pkl, threshold):
    """Split SHAPleyData object by methylation type (needs to change; but low methylation is 0.5 beta value and below) and output to file."""
    os.makedirs(output_dir,exist_ok=True)
    shapley_data=ShapleyData.from_pickle(shapley_data)
    shapley_data_explorer=ShapleyDataExplorer(shapley_data)
    test_methyl_array=MethylationArray.from_pickle(test_pkl)
    shap_data_methylation=shapley_data_explorer.return_shapley_data_by_methylation_status(test_methyl_array, threshold)
    shap_data_methylation['hypo'].to_pickle(join(output_dir,'hypo_shapley_data.p'))
    shap_data_methylation['hyper'].to_pickle(join(output_dir,'hyper_shapley_data.p'))

@interpret.command()
@click.option('-s', '--shapley_data', default='./interpretations/shapley_explanations/shapley_data.p', help='Pickle containing top CpGs.', type=click.Path(exists=False), show_default=True)
@click.option('-t', '--test_pkl', default='./train_val_test_sets/test_methyl_array.pkl', help='Pickle containing testing set.', type=click.Path(exists=False), show_default=True)
@click.option('-c', '--col', default='age', help='Column to turn into bins.', type=click.Path(exists=False),show_default=True)
@click.option('-n', '--n_bins', default=10, help='Number of bins.',show_default=True)
@click.option('-ot', '--output_test_pkl', default='./train_val_test_sets/test_methyl_array_shap_binned.pkl', help='Binned shap pickle for further testing.', type=click.Path(exists=False), show_default=True)
@click.option('-os', '--output_shap_pkl', default='./interpretations/shapley_explanations/shapley_binned.p', help='Pickle containing top CpGs, binned phenotype.', type=click.Path(exists=False), show_default=True)
def bin_regression_shaps(shapley_data, test_pkl,col,n_bins,output_test_pkl,output_shap_pkl):
    """Take aggregate individual scores from regression results, that normally are not categorized, and aggregate across new category created from continuous data."""
    os.makedirs(output_test_pkl[:output_test_pkl.rfind('/')],exist_ok=True)
    os.makedirs(output_shap_pkl[:output_shap_pkl.rfind('/')],exist_ok=True)
    shapley_data=ShapleyData.from_pickle(shapley_data)
    shapley_data_explorer=ShapleyDataExplorer(shapley_data)
    test_methyl_array=MethylationArray.from_pickle(test_pkl)
    new_col_name = test_methyl_array.bin_column(col,n_bins)
    print(test_methyl_array.categorical_breakdown(new_col_name))
    shapley_data = shapley_data_explorer.return_binned_shapley_data(col, test_methyl_array.pheno[new_col_name], add_top_negative=False)
    print(ShapleyDataExplorer(shapley_data).list_classes())
    test_methyl_array.write_pickle(output_test_pkl)
    shapley_data.to_pickle(output_shap_pkl)

# create force plots for each sample??? use output shapley values and output explainer
@interpret.command()
@click.option('-s', '--shapley_data', default='./interpretations/shapley_explanations/shapley_data.p', help='Pickle containing top CpGs.', type=click.Path(exists=False), show_default=True)
@click.option('-c', '--class_names', default=[''], multiple=True, help='Class names.', show_default=True)
@click.option('-o', '--output_dir', default='./interpretations/shapley_explanations/top_cpgs_jaccard/', help='Output directory for cpg jaccard_stats.', type=click.Path(exists=False), show_default=True)
@click.option('-ov', '--overall', is_flag=True, help='Output overall similarity.', show_default=True)
@click.option('-i', '--include_individuals', is_flag=True, help='Output individuals.', show_default=True)
@click.option('-co', '--cooccurrence', is_flag=True, help='Output cooccurrence instead jaccard.', show_default=True)
@click.option('-opt', '--optimize_n_cpgs', is_flag=True, help='Search for number of top CpGs to use.', show_default=True)
def shapley_jaccard(shapley_data,class_names, output_dir, overall, include_individuals, cooccurrence, optimize_n_cpgs):
    """Plot Shapley Jaccard Similarity Matrix to demonstrate sharing of Top CpGs between classes/groupings of individuals."""
    os.makedirs(output_dir,exist_ok=True)
    shapley_data=ShapleyData.from_pickle(shapley_data)
    shapley_data_explorer=ShapleyDataExplorer(shapley_data)
    outfilename=join(output_dir,'{}_jaccard.csv'.format('_'.join(class_names)))
    shapley_data_explorer.jaccard_similarity_top_cpgs(class_names,include_individuals,overall, cooccurrence).to_csv(outfilename)
    if optimize_n_cpgs:
        from skopt import gp_minimize
        shapley_data_explorer2 = ShapleyDataExplorer(shapley_data)
        def f(x):
            shapley_data_explorer2.shapley_data = shapley_data_explorer.limit_number_top_cpgs(n_top_cpgs=x[0])
            jaccard_matrix = shapley_data_explorer2.jaccard_similarity_top_cpgs(class_names,include_individuals,overall, cooccurrence).values
            jaccard_mean=jaccard_matrix.mean()
            print(x[0],jaccard_mean)
            return -jaccard_mean
        results=gp_minimize(f,[(100,shapley_data_explorer.shapley_data.top_cpgs['overall'].shape[0])],n_calls=25,n_jobs=8)
        print("{} number CpGs produces most similar jaccard similarity matrix.".format(results.x))
        print(dict(zip(results.x_iters.flatten().tolist(), (results.func_vals*-1).tolist())))

@interpret.command()
@click.option('-i', '--input_csv', default='./interpretations/shapley_explanations/top_cpgs_jaccard/all_jaccard.csv', help='Output directory for cpg jaccard_stats.', type=click.Path(exists=False), show_default=True)
@click.option('-t', '--test_pkl', default='./train_val_test_sets/test_methyl_array.pkl', help='Pickle containing testing set.', type=click.Path(exists=False), show_default=True)
@click.option('-c', '--col', default='disease', help='Column to sort on.', show_default=True)
@click.option('-o', '--output_csv', default='./interpretations/shapley_explanations/top_cpgs_jaccard/all_jaccard_sorted.csv', help='Output directory for cpg jaccard_stats.', type=click.Path(exists=False), show_default=True)
@click.option('-sym', '--symmetric', is_flag=True, help='Is symmetric?', show_default=True)
def order_results_by_col(input_csv, test_pkl, col, output_csv, symmetric):
    """Order results that produces some CSV by phenotype column, alphabetical ordering to show maybe that results group together. Plot using pymethyl-visualize."""
    os.makedirs(output_csv[:output_csv.rfind('.')],exist_ok=True)
    df=pd.read_csv(input_csv,index_col=0)
    if os.path.exists(test_pkl):
        test_arr=MethylationArray.from_pickle(test_pkl) # temporary solution
        test_arr_idx=test_arr.pheno.sort_values(col).index.values
        if np.all(np.vectorize(lambda x: x.startswith(col))(df.index.values)):
            test_arr_idx = np.vectorize(lambda x: '{}_{}'.format(col,x))(test_arr_idx)
        test_arr_idx=test_arr_idx[np.isin(test_arr_idx,df.index.values)] # temporary fix
    else:
        test_arr_idx = np.sort(df.index.values)
    df=df.loc[test_arr_idx,test_arr_idx] if symmetric else df[test_arr_idx]
    df.to_csv(output_csv)


@interpret.command()
@click.option('-s', '--shapley_data', default='./interpretations/shapley_explanations/shapley_data.p', help='Pickle containing top CpGs.', type=click.Path(exists=False), show_default=True)
@click.option('-i', '--individuals', default=[''], multiple=True, help='Individuals.', show_default=True)
@click.option('-o', '--output_dir', default='./interpretations/shapley_explanations/top_cpgs_methylation/', help='Output directory for cpg methylation shap.', type=click.Path(exists=False), show_default=True)
@click.option('-t', '--test_pkl', default='./train_val_test_sets/test_methyl_array.pkl', help='Pickle containing testing set.', type=click.Path(exists=False), show_default=True)
def view_methylation_top_cpgs(shapley_data,individuals, output_dir, test_pkl):
    """Write the Top CpGs for each class/individuals to file along with methylation information."""
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
    """List individuals that have ShapleyData. Not all made the cut from the test dataset."""
    shapley_data=ShapleyData.from_pickle(shapley_data)
    shapley_data_explorer=ShapleyDataExplorer(shapley_data)
    print(shapley_data_explorer.list_individuals())

@interpret.command()
@click.option('-s', '--shapley_data', default='./interpretations/shapley_explanations/shapley_data.p', help='Pickle containing top CpGs.', type=click.Path(exists=False), show_default=True)
def list_classes(shapley_data):
    """List classes/multioutput regression cell-types that have SHAPley data to be interrogated."""
    shapley_data=ShapleyData.from_pickle(shapley_data)
    print(list(shapley_data.top_cpgs['by_class'].keys()))

@interpret.command()
@click.option('-s', '--shapley_data', default='./interpretations/shapley_explanations/shapley_data.p', help='Pickle containing top CpGs.', type=click.Path(exists=False), show_default=True)
@click.option('-nf', '--n_top_features', default=500, show_default=True, help='Top features to select for shap outputs.')
@click.option('-o', '--output_pkl', default='./interpretations/shapley_explanations/shapley_reduced_data.p', help='Pickle containing top CpGs, reduced number.', type=click.Path(exists=False), show_default=True)
def reduce_top_cpgs(shapley_data,n_top_features,output_pkl):
    """Reduce set of top cpgs."""
    os.makedirs(output_pkl[:output_pkl.rfind('/')],exist_ok=True)
    shapley_data=ShapleyData.from_pickle(shapley_data)
    shapley_data_explorer=ShapleyDataExplorer(shapley_data)
    shapley_data_explorer.limit_number_top_cpgs(n_top_cpgs=n_top_features).to_pickle(output_pkl)

@interpret.command()
@click.option('-s', '--shapley_data', default='./interpretations/shapley_explanations/shapley_data.p', help='Pickle containing top CpGs.', type=click.Path(exists=False), show_default=True)
@click.option('-nf', '--n_top_features', default=500, show_default=True, help='Top features to select for shap outputs.')
@click.option('-o', '--output_pkl', default='./interpretations/shapley_explanations/shapley_reduced_data.p', help='Pickle containing top CpGs, reduced number.', type=click.Path(exists=False), show_default=True)
@click.option('-a', '--abs_val', is_flag=True, help='Top CpGs found using absolute value.')
@click.option('-n', '--neg_val', is_flag=True, help='Return top CpGs that are making negative contributions.')
def regenerate_top_cpgs(shapley_data,n_top_features,output_pkl, abs_val, neg_val):
    """Increase size of Top CpGs using the original SHAP scores."""
    os.makedirs(output_pkl[:output_pkl.rfind('/')],exist_ok=True)
    shapley_data=ShapleyData.from_pickle(shapley_data)
    shapley_data_explorer=ShapleyDataExplorer(shapley_data)
    shapley_data_explorer.regenerate_individual_shap_values(n_top_cpgs=n_top_features,abs_val=abs_val, neg_val=neg_val).to_pickle(output_pkl)

@interpret.command()
@click.option('-s', '--shapley_data', default='./interpretations/shapley_explanations/shapley_data.p', help='Pickle containing top CpGs.', type=click.Path(exists=False), show_default=True)
@click.option('-o', '--output_dir', default='./interpretations/shap_outputs/', help='Output directory for output plots.', type=click.Path(exists=False), show_default=True)
@click.option('-i', '--individuals', default=[''], multiple=True, help='Individuals to evaluate.', show_default=True)
@click.option('-c', '--classes', default=[''], multiple=True, help='Classes to evaluate.', show_default=True)
@click.option('-hist', '--output_histogram', is_flag=True, help='Whether to output a histogram for each class/individual of their SHAP scores.')
@click.option('-abs', '--absolute', is_flag=True, help='Use sums of absolute values in making computations.')
@click.option('-log', '--log_transform', is_flag=True, help='Log transform final results for plotting.')
def return_shap_values(shapley_data,output_dir,individuals,classes, output_histogram, absolute, log_transform):
    """Return matrix of shapley values per class, with option to, classes/individuals are columns, CpGs are rows, option to plot multiple histograms/density plots."""
    from sklearn.metrics import pairwise_distances
    os.makedirs(output_dir,exist_ok=True)
    shapley_data=ShapleyData.from_pickle(shapley_data)
    shapley_data_explorer=ShapleyDataExplorer(shapley_data)
    individuals=list(filter(None,individuals))
    classes=list(filter(None,classes))
    if absolute:
        shapley_data_explorer.make_shap_scores_abs()
    if classes and classes[0]=='all':
        classes = shapley_data_explorer.list_classes()
    if individuals and individuals[0]=='all':
        individuals = shapley_data_explorer.list_individuals(return_list=True)
    concat_list=[]
    if classes:
        for class_name in classes:
            concat_list.append(shapley_data_explorer.extract_class(class_name,get_shap_values=True))
    if individuals:
        for individual in individuals:
            concat_list.append(shapley_data_explorer.extract_individual(individual,get_shap_values=True))
    df=pd.concat(concat_list,axis=1,keys=classes+individuals)
    df.to_csv(join(output_dir,'returned_shap_values.csv'))
    pd.DataFrame(pairwise_distances(df.T.values,metric='correlation'),index=list(df),columns=list(df)).to_csv(join(output_dir,'returned_shap_values_corr_dist.csv'))
    #df.corr().to_csv(join(output_dir,'returned_shap_values_corr_dist.csv'))
    if output_histogram:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set()
        for entity in (classes+individuals):
            shap_scores = df[entity]
            plt.figure()
            ax=sns.distplot(shap_scores, kde=(log_transform==False))
            if log_transform:
                ax.set_xscale('symlog')
            plt.xlabel('{} SHAP value'.format("Log" if log_transform else ""))
            plt.ylabel('Frequency')
            plt.savefig(join(output_dir,'{}_shap_values.png'.format(entity)),dpi=300)
            top_10_shaps = shap_scores.abs().sort_values(ascending=False).iloc[:10]
            top_10_shaps=pd.DataFrame({'cpgs':top_10_shaps.index, '|SHAP|':top_10_shaps.values})
            plt.figure()
            ax=sns.barplot('|SHAP|','cpgs',orient='h',data=top_10_shaps)
            ax.tick_params(labelsize=4)
            plt.savefig(join(output_dir,'{}_top_shap_values.png'.format(entity)),dpi=300)
    if not absolute:
        print("All saved values reflect absolute values of sums, not sums of absolute values, if CpG SHAP scores are opposite signs across individuals, this will reduce the score of the resulting SHAP estimate.")

@interpret.command()
@click.option('-s', '--shapley_data', default='./interpretations/shapley_explanations/shapley_data.p', help='Pickle containing top CpGs.', type=click.Path(exists=False), show_default=True)
@click.option('-o', '--output_dir', default='./interpretations/output_plots/', help='Output directory for output plots.', type=click.Path(exists=False), show_default=True)
@click.option('-i', '--individuals', default=[''], multiple=True, help='Individuals to evaluate.', show_default=True)
@click.option('-c', '--classes', default=[''], multiple=True, help='Classes to evaluate.', show_default=True)
def plot_top_cpgs(shapley_data,output_dir,individuals,classes):
    """Plot the top cpgs locations in the genome using circos."""
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

@interpret.command()
@click.option('-i', '--embedding_methyl_array_pkl', default='./embeddings/vae_methyl_arr.pkl', help='Use ./predictions/vae_mlp_methyl_arr.pkl or ./embeddings/vae_methyl_arr.pkl for vae interpretations.', type=click.Path(exists=False), show_default=True)
@click.option('-c', '--pheno_col', default='disease', help='Column to separate on.', type=click.Path(exists=False), show_default=True)
@click.option('-m', '--metric', default='cosine', help='Distance metric to compare classes.', type=click.Path(exists=False), show_default=True)
@click.option('-t', '--trim', default=0.0, help='Trim outlier distances. Number 0-0.5.', show_default=True)
@click.option('-o', '--output_csv', default='./results/class_embedding_differences.csv', help='Distances between classes.', type=click.Path(exists=False), show_default=True)
@click.option('-op', '--output_pval_csv', default='', help='If specify a CSV file, outputs pairwise manova tests between embeddings for clusters.', type=click.Path(exists=False), show_default=True)
def interpret_embedding_classes(embedding_methyl_array_pkl, pheno_col, metric, trim, output_csv, output_pval_csv):
    """Compare average distance between classes in embedding space."""
    os.makedirs(output_csv[:output_csv.rfind('/')],exist_ok=True)
    methyl_array = MethylationArray.from_pickle(embedding_methyl_array_pkl)
    distance_compute = DistanceMatrixCompute(methyl_array, pheno_col)
    distance_compute.compute_distances(metric, trim)
    distance_compute.return_distances().to_csv(output_csv)
    if output_pval_csv:
        distance_compute.calculate_p_values()
        distance_compute.return_p_values().to_csv(output_pval_csv)

#################

if __name__ == '__main__':
    interpret()
