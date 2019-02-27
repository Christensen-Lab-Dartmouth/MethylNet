Dataset: GSE87571

Directory: /dartfs-hpc/rc/lab/C/ChristensenB/users/jlevy/projects/blood/

include_col.txt (tab delimited):
age:ch1 Age
gender:ch1  Sex
tissue:ch1  Tissue

* cd /dartfs-hpc/rc/lab/C/ChristensenB/users/jlevy/projects/blood && module load python/3-Anaconda && source activate methylnet_pro2

Preprocessing:

* nohup python preprocess.py download_geo -g GSE87571 &
* wget https://raw.githubusercontent.com/Christensen-Lab-Dartmouth/data-processing/master/data_sets/johansson_lifespan_aging/results/%20johansson%20_cell_type_estimates.csv?token=ASyRZ1DQBCRXZ3hGTQkMXyTDJgvPaKMeks5cbBGHwA%3D%3D
* mv *== cell_type_estimates.csv # =*
* nano include_col.txt
* python preprocess.py create_sample_sheet -is ./geo_idats/GSE87571_clinical_info.csv -s geo -i geo_idats/ -os geo_idats/samplesheet.csv -d "disease state:ch1" -c include_col.txt
* mkdir backup_clinical && mv ./geo_idats/GSE87571_clinical_info.csv backup_clinical
* python -c "import pandas as pd,numpy as np; df=pd.read_csv('cell_type_estimates.csv',index_col=0);df['Basename']=np.vectorize(lambda x: 'geo_idats/'+x)(list(df.index));df.reset_index(drop=True).to_csv('cell_type_estimates_adjusted.csv')"
* python preprocess.py merge_sample_sheets -nd -s1 geo_idats/samplesheet.csv -s2 cell_type_estimates_adjusted.csv -os geo_idats/samplesheet_merged.csv
* mv ./geo_idats/samplesheet.csv backup_clinical
* nohup pymethyl-preprocess preprocess_pipeline -i geo_idats/ -p minfi -noob -qc &
* nohup python preprocess.py preprocess_pipeline -i geo_idats/ -p minfi -noob -u & # add moving jpg files
* mkdir qc_report && mv *.jpg qc_report #=*
* python utils.py print_number_sex_cpgs -i preprocess_outputs/methyl_array.pkl #
* pymethyl-utils remove_sex -i preprocess_outputs/methyl_array.pkl
* python preprocess.py na_report -i autosomal/methyl_array.pkl -o na_report/ # NA Rate is on average: 0.706484934739793%
* nohup pymethyl-preprocess imputation_pipeline -i ./autosomal/methyl_array.pkl -s fancyimpute -m KNN -k 15 -st 0.05 -ct 0.05 &
* pymethyl-preprocess feature_select -n 300000
* mkdir visualizations
* nohup pymethyl-visualize transform_plot -o visualizations/pre_vae_umap.html -c Age -nn 8 &
* nohup python visualizations.py transform_plot -o visualizations/pre_vae_umap_sex.html -c Sex -nn 8 &
* nohup python visualizations.py transform_plot -o visualizations/pre_vae_umap_CD4T.html -c CD4T -nn 8 &
* nohup python visualizations.py transform_plot -o visualizations/pre_vae_umap_CD8T.html -c CD8T -nn 8 &
* nohup python visualizations.py transform_plot -o visualizations/pre_vae_umap_NK.html -c NK -nn 8 &
* nohup python visualizations.py transform_plot -o visualizations/pre_vae_umap_Bcell.html -c Bcell -nn 8 &
* nohup python visualizations.py transform_plot -o visualizations/pre_vae_umap_gMDSC.html -c gMDSC -nn 8 &

* pymethyl-utils train_test_val_split -tp .8 -vp .125


MethylNet Commands:

* mkdir embeddings
* python embedding.py launch_hyperparameter_scan -sc Age -t -mc 0.84 -b 1. -g -j 20
* python embedding.py launch_hyperparameter_scan -sc Age -t -g -n 1 -b 1.
* pymethyl-visualize transform_plot -i embeddings/vae_methyl_arr.pkl -nn 8 -c Age
* python predictions.py launch_hyperparameter_scan -ic Age -t -mc 0.84 -g -j 200
* python predictions.py launch_hyperparameter_scan -ic Age -t -g -n 1
* pymethyl-visualize transform_plot -i predictions/vae_mlp_methyl_arr.pkl -nn 8 -c Age
* python model_interpretability.py produce_shapley_data_torque -c "python model_interpretability.py produce_shapley_data -mth gradient -ssbs 30 -ns 300 -bs 100 -rc 4. -r 0 -rt 0 -cn Age -nf 4000 -c"
* python model_interpretability.py regenerate_top_cpgs -nf 4000 -a
* python predictions.py regression_report
* python model_interpretability.py split_hyper_hypo_methylation -s ./interpretations/shapley_explanations/shapley_reduced_data.p
* python model_interpretability.py shapley_jaccard -c all -i -s ./interpretations/shapley_explanations/shapley_data_by_methylation/hypo_shapley_data.p -o ./interpretations/shapley_explanations/top_cpgs_jaccard/hypo/ && python model_interpretability.py order_results_by_col -c Age -i ./interpretations/shapley_explanations/top_cpgs_jaccard/hypo/all_jaccard.csv -o ./interpretations/shapley_explanations/top_cpgs_jaccard/hypo/all_jaccard.sorted.csv
* python model_interpretability.py shapley_jaccard -c all -i -s ./interpretations/shapley_explanations/shapley_data_by_methylation/hyper_shapley_data.p -o ./interpretations/shapley_explanations/top_cpgs_jaccard/hyper/ && python model_interpretability.py order_results_by_col -c Age -i ./interpretations/shapley_explanations/top_cpgs_jaccard/hyper/all_jaccard.csv -o ./interpretations/shapley_explanations/top_cpgs_jaccard/hyper/all_jaccard.sorted.csv
* pymethyl-visualize plot_heatmap -c -m similarity -fs .4 -i ./interpretations/shapley_explanations/top_cpgs_jaccard/hypo/all_jaccard.sorted.csv -o ./interpretations/shapley_explanations/top_cpgs_jaccard/hypo/all_hypo_jaccard.png
* pymethyl-visualize plot_heatmap -c -m similarity -fs .4 -i ./interpretations/shapley_explanations/top_cpgs_jaccard/hyper/all_jaccard.sorted.csv -o ./interpretations/shapley_explanations/top_cpgs_jaccard/hyper/all_hyper_jaccard.png
* python model_interpretability.py bin_regression_shaps -c Age
* python model_interpretability.py shapley_jaccard -c all -s ./interpretations/shapley_explanations/shapley_binned.p  -o ./interpretations/shapley_explanations/top_cpgs_jaccard/ -ov
* python model_interpretability.py order_results_by_col -c Age -t null -i ./interpretations/shapley_explanations/top_cpgs_jaccard/all_jaccard.csv -o ./interpretations/shapley_explanations/top_cpgs_jaccard/all_jaccard.sorted.csv &
* pymethyl-utils counts -i train_val_test_sets/test_methyl_array_shap_binned.pkl -k Age_binned
* pymethyl-visualize plot_heatmap -m similarity -fs .7 -i ./interpretations/shapley_explanations/top_cpgs_jaccard/all_jaccard.sorted.csv -o ./interpretations/shapley_explanations/top_cpgs_jaccard/all_jaccard.png -x -y -c &
* python visualizations_methylnet.py plot_training_curve -t embeddings/training_val_curve.p -vae -o results/embed_training_curve.png -thr 2e8
* python visualizations_methylnet.py plot_training_curve -thr 2e6
* python model_interpretability.py interpret_biology -ov -c all
* python model_interpretability.py interpret_biology -ov -c all -s interpretations/shapley_explanations/shapley_binned.p -cgs clock -g
* pymethyl-visualize plot_heatmap -m similarity -fs .7 -i ./interpretations/biological_explanations/clock_overlaps.csv -o ./interpretations/biological_explanations/clock_overlaps.png -x -y -a &

# plot heatmap of top cpgs vs samples, reduce count to 1000, hclustered
#

# get library using bio_interpreter or extract_methylation_array
# Then visualize using subset_array (extract_ already does this), to_csv and then plot_heatmap
* python model_interpretability.py extract_methylation_array -col Age_binned -s  interpretations/shapley_explanations/shapley_binned.p  -c
* nohup python model_interpretability.py interpret_biology -ov -c all -s interpretations/shapley_explanations/shapley_binned.p -cgs horvath -ex &
* pymethyl-visualize plot_heatmap -fs .7 -i ./interpretations/shapley_explanations/top_cpgs_extracted_methylarr/beta.csv -o ./interpretations/biological_explanations/beta.png -c &


pymethyl-utils subset_array -i train_val_test_sets/test_methyl_array.pkl -c ./interpretations/biological_explanations/cpg_library.pkl
pymethyl-utils pkl_to_csv -i subset/methyl_array.pkl -o subset/ -col Age_binned
pymethyl-visualize plot_heatmap -fs .7 -i ./subset/beta.csv -o ./subset/beta.png -c -x &
pymethyl-utils set_part_array_zeros -i train_val_test_sets/test_methyl_array.pkl -c ./interpretations/biological_explanations/cpg_library.pkl # only set top 1k overall or intersected horvath
# set them to background mean instead!!!!!
CUDA_VISIBLE_DEVICES="0" python predictions.py make_new_predictions -tp removal/methyl_array.pkl -c -ic Age
python predictions.py regression_report -r new_predictions/results.p -o new_results/
# or run predictions with library omitted
# set_part_array_zeros then make_new_predictions, then classification/regression report

# test external set:
# download
# preprocess
# pymethyl-utils create_external_validation_set
# make_new_predictions then classification/regression report

# to-do search for missing cpgs, do same for other studies
