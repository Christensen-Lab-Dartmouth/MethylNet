Dataset: GSE42861

Directory: /dartfs-hpc/rc/lab/C/ChristensenB/users/jlevy/projects/smoke/

include_col.txt (tab delimited):
age:ch1	Age
gender:ch1	Sex
cell type:ch1 Cell_Type
subject:ch1 Subject_Type
disease state:ch1 Disease_State

* cd /dartfs-hpc/rc/lab/C/ChristensenB/users/jlevy/projects/smoke && module load python/3-Anaconda && source activate methylnet_pro2

Preprocessing:

* nohup python preprocess.py download_geo -g GSE42861 &
* python preprocess.py create_sample_sheet -is ./geo_idats/GSE42861_clinical_info.csv -s geo -i geo_idats/ -os geo_idats/samplesheet.csv -d "smoking status:ch1" -c include_col.txt
* mkdir backup_clinical && mv ./geo_idats/GSE42861_clinical_info.csv backup_clinical
* python preprocess.py meffil_encode -is geo_idats/samplesheet.csv -os geo_idats/samplesheet.csv
* nohup time python preprocess.py preprocess_pipeline -n 30 -m -qc  -pc -1 -bns 0.05 -pds 0.05 -bnc 0.05 -pdc 0.05 -sc -2 -sd 5 -i /dartfs-hpc/rc/lab/C/ChristensenB/users/jlevy/projects/smoke/geo_idats/ -o /dartfs-hpc/rc/lab/C/ChristensenB/users/jlevy/projects/smoke/preprocess_outputs/methyl_array.pkl & # 13:51.16
* nohup time python preprocess.py preprocess_pipeline -n 14 -m -u -pc -1 -bns 0.05 -pds 0.05 -bnc 0.05 -pdc 0.05 -sc -2 -sd 5 -i /dartfs-hpc/rc/lab/C/ChristensenB/users/jlevy/projects/smoke/geo_idats/ -o /dartfs-hpc/rc/lab/C/ChristensenB/users/jlevy/projects/smoke/preprocess_outputs/methyl_array.pkl & # 25:29.04,  110 GB
* python utils.py remove_sex -i preprocess_outputs/methyl_array.pkl
* python preprocess.py na_report -i autosomal/methyl_array.pkl -o na_report/ # 0.1928996540098538%
* nohup python preprocess.py imputation_pipeline -i ./autosomal/methyl_array.pkl -s fancyimpute -m KNN -k 5 -st 0.05 -ct 0.05 &
* python preprocess.py feature_select -n 300000
* mkdir visualizations
* nohup python visualizations.py transform_plot -o visualizations/pre_vae_umap.html -c disease -nn 8 &
* nohup python visualizations.py transform_plot -o visualizations/pre_vae_umap_sex.html -c Sex -nn 8 &
* nohup python visualizations.py transform_plot -o visualizations/pre_vae_umap_age.html -c Age -nn 8 &
* nohup python visualizations.py transform_plot -o visualizations/pre_vae_umap_sex_disease.html -c Disease_State -nn 8 &
* python utils.py train_test_val_split -tp .8 -vp .125 -cat -k Disease_State

MethylNet Commands:

* python embedding.py launch_hyperparameter_scan -sc Disease_State -t -mc 0.84 -b 1. -g -j 20
* python embedding.py launch_hyperparameter_scan -sc Disease_State -t -g -n 1 -b 1.
* python predictions.py launch_hyperparameter_scan -ic Disease_State -cat -t -g -mc 0.84 -j 80
* python predictions.py launch_hyperparameter_scan -ic Disease_State -cat -t -g -n 1
* python model_interpretability.py produce_shapley_data_torque -c "python model_interpretability.py produce_shapley_data -mth gradient -ssbs 30 -ns 300 -bs 100 -col Disease_State -r 0 -rt 30 -nf 4000 -c"
* python model_interpretability.py shapley_jaccard -c all -i -ov
* pymethyl-visualize plot_heatmap -m similarity -fs .2 -i ./interpretations/shapley_explanations/top_cpgs_jaccard/all_jaccard.csv -o ./interpretations/shapley_explanations/top_cpgs_jaccard/all_jaccard.png -x -y -c &
* python model_interpretability.py reduce_top_cpgs -nf 1000 && python model_interpretability.py split_hyper_hypo_methylation -s ./interpretations/shapley_explanations/shapley_reduced_data.p
* python model_interpretability.py shapley_jaccard -c all -i -ov -s ./interpretations/shapley_explanations/shapley_data_by_methylation/hypo_shapley_data.p -o ./interpretations/shapley_explanations/top_cpgs_jaccard/hypo/
* pymethyl-visualize plot_heatmap -m similarity -fs .2 -i ./interpretations/shapley_explanations/top_cpgs_jaccard/hypo/all_jaccard.csv -o ./interpretations/shapley_explanations/top_cpgs_jaccard/all_hypo_jaccard.png -x -y -c &
* python model_interpretability.py shapley_jaccard -c all -i -ov -s ./interpretations/shapley_explanations/shapley_data_by_methylation/hyper_shapley_data.p -o ./interpretations/shapley_explanations/top_cpgs_jaccard/hyper/
* pymethyl-visualize plot_heatmap -m similarity -fs .2 -i ./interpretations/shapley_explanations/top_cpgs_jaccard/hyper/all_jaccard.csv -o ./interpretations/shapley_explanations/top_cpgs_jaccard/all_hyper_jaccard.png -x -y -c &
* python predictions.py classification_report
* python visualizations_methylnet.py plot_training_curve -t embeddings/training_val_curve.p -vae -o results/embed_training_curve.png
* python visualizations_methylnet.py plot_training_curve
* python visualizations_methylnet.py plot_roc_curve
