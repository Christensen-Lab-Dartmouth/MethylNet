Dataset: GSE112179

Directory: /dartfs-hpc/rc/lab/C/ChristensenB/users/jlevy/projects/850k/

include_col.txt (tab delimited):
age:ch1 Age
cell.type:ch1   Cell_Type
pmi:ch1 pmi
race:ch1        Race
Sex:ch1 Sex
tissue:ch1	Tissue

* cd /dartfs-hpc/rc/lab/C/ChristensenB/users/jlevy/projects/850k && module load python/3-Anaconda && source activate methylnet_pro2

Preprocessing:

* nohup pymethyl-preprocess download_geo -g GSE112179 &
* nano include_col.txt
* pymethyl-preprocess create_sample_sheet -is ./geo_idats/GSE112179_clinical_info.csv -s geo -i geo_idats/ -os geo_idats/samplesheet.csv -d "dist.dx:ch1" -c include_col.txt
* mkdir backup_clinical && mv ./geo_idats/GSE112179_clinical_info.csv backup_clinical
* pymethyl-preprocess meffil_encode -is geo_idats/samplesheet.csv -os geo_idats/samplesheet.csv
* nohup time pymethyl-preprocess preprocess_pipeline -i geo_idats/ -m -n 30 -i ./geo_idats/ -o preprocess_outputs/methyl_array.pkl &
* pymethyl-utils print_number_sex_cpgs -a epic -i preprocess_outputs/methyl_array.pkl #
* pymethyl-utils remove_sex -i preprocess_outputs/methyl_array.pkl -a epic
* pymethyl-preprocess na_report -i autosomal/methyl_array.pkl -o na_report/ # NA Rate is on average: 0.24249205298696708%
* nohup pymethyl-preprocess imputation_pipeline -i ./autosomal/methyl_array.pkl -s fancyimpute -m MICE -k 7 -st 0.05 -ct 0.05 &
* pymethyl-preprocess feature_select -n 500000
* mkdir visualizations
* nohup pymethyl-visualize transform_plot -o visualizations/pre_vae_umap.html -c disease -nn 8 &

* python utils.py train_test_val_split -tp .8 -vp .125


MethylNet Commands:
