Dataset: GSE81961

Directory: /dartfs-hpc/rc/lab/C/ChristensenB/users/jlevy/projects/age_external_validation

include_col.txt (tab delimited):
age (yr):ch1 Age
gender:ch1  Sex
tissue:ch1  Tissue

* cd /dartfs-hpc/rc/lab/C/ChristensenB/users/jlevy/projects/age_external_validation && module load python/3-Anaconda && source activate methylnet_pro2

Preprocessing:

* nohup python preprocess.py download_geo -g GSE81961 &
* python preprocess.py create_sample_sheet -is ./geo_idats/GSE81961_clinical_info.csv -s geo -i geo_idats/ -os geo_idats/samplesheet.csv -d "disease state:ch1" -c include_col.txt
* mkdir backup_clinical && mv ./geo_idats/GSE81961_clinical_info.csv backup_clinical
* python preprocess.py meffil_encode -is geo_idats/samplesheet.csv -os geo_idats/samplesheet.csv
* python preprocess.py preprocess_pipeline -n 35 -m  -pc -1 -bns 0.05 -pds 0.05 -bnc 0.05 -pdc 0.05 -sc -2 -sd 5 -i /dartfs-hpc/rc/lab/C/ChristensenB/users/jlevy/projects/crohn/geo_idats/ -o /dartfs-hpc/rc/lab/C/ChristensenB/users/jlevy/projects/crohn/preprocess_outputs/methyl_array.pkl # time elapsed 4:47.46 , 4 minutes
* pymethyl-utils remove_sex -i preprocess_outputs/methyl_array.pkl
* python preprocess.py na_report -i autosomal/methyl_array.pkl -o na_report/ # 0.17111708849559365%
* nohup python preprocess.py imputation_pipeline -i ./autosomal/methyl_array.pkl -s fancyimpute -m KNN -k 5 -st 0.05 -ct 0.05 &
* python preprocess.py feature_select -n 300000

MethylNet Commands:
* mkdir methyl_array && mv final_preprocessed/methyl_array.pkl methyl_array
* scp ../blood/train_val_test_sets/train_methyl_array.pkl methyl_array
* pymethyl-utils create_external_validation_set -t methyl_array/train_methyl_array.pkl -q methyl_array/methyl_array.pkl
* python predictions.py make_new_predictions -tp external_validation/methyl_array.pkl -ic Age
* python predictions.py regression_report -r new_predictions/results.p -o new_results/
# 27k CpGs were replaced with 0.5, which lead to low R2 score... Can impute from test array from internal valid cohort?
# simulation packages quantroSim and intersim
