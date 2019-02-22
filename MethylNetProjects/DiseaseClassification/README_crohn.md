Dataset: GSE81961

Directory: /dartfs-hpc/rc/lab/C/ChristensenB/users/jlevy/projects/crohn/

include_col.txt (tab delimited):
age (yr):ch1 Age
gender:ch1  Sex
tissue:ch1  Tissue

* cd /dartfs-hpc/rc/lab/C/ChristensenB/users/jlevy/projects/crohn && module load python/3-Anaconda && source activate methylnet_pro2

Preprocessing:

* nohup python preprocess.py download_geo -g GSE81961 &
* python preprocess.py create_sample_sheet -is ./geo_idats/GSE81961_clinical_info.csv -s geo -i geo_idats/ -os geo_idats/samplesheet.csv -d "disease state:ch1" -c include_col.txt
* mkdir backup_clinical && mv ./geo_idats/GSE81961_clinical_info.csv backup_clinical
* python preprocess.py meffil_encode -is geo_idats/samplesheet.csv -os geo_idats/samplesheet.csv
* python preprocess.py preprocess_pipeline -n 35 -m  -pc -1 -bns 0.05 -pds 0.05 -bnc 0.05 -pdc 0.05 -sc -2 -sd 5 -i /dartfs-hpc/rc/lab/C/ChristensenB/users/jlevy/projects/crohn/geo_idats/ -o /dartfs-hpc/rc/lab/C/ChristensenB/users/jlevy/projects/crohn/preprocess_outputs/methyl_array.pkl # time elapsed 4:47.46 , 4 minutes
* python preprocess.py na_report -i preprocess_outputs/methyl_array.pkl -o na_report/ # 0.17111708849559365%
* nohup python preprocess.py imputation_pipeline -i ./preprocess_outputs/methyl_array.pkl -s fancyimpute -m KNN -k 5 -st 0.05 -ct 0.05 &
* python preprocess.py feature_select -n 300000
* mkdir visualizations
* nohup python visualizations.py transform_plot -o visualizations/pre_vae_umap.html -c disease -nn 8 &
* nohup python visualizations.py transform_plot -o visualizations/pre_vae_umap_age.html -c Age -nn 8 &
* nohup python visualizations.py transform_plot -o visualizations/pre_vae_umap_sex.html -c Sex -nn 8 &

OLD:
* python preprocess.py split_preprocess_input_by_subtype -i geo_idats/samplesheet.csv -o preprocess_outputs/
* time python preprocess.py  batch_deploy_preprocess -n 17 -m -r -c 2
Preprocess time: All processes have completed

real	4m59.066s
user	49m0.946s
sys	6m40.525s
* python preprocess.py na_report -i preprocess_outputs/ -o na_report/ -r
preprocess_outputs/Control/methyl_array.pkl NA Rate is on average: 0.15274262480892614%
preprocess_outputs/Crohn/methyl_array.pkl NA Rate is on average: 0.07682075275304943%
* python preprocess.py combine_methylation_arrays -d ./preprocess_outputs/
* nohup python preprocess.py imputation_pipeline -i ./combined_outputs/methyl_array.pkl -ss -s fancyimpute -m KNN -k 15 -st 0.05 -ct 0.05 &
* python preprocess.py feature_select -n 300000
* mkdir visualizations && nohup python visualizations.py transform_plot -o visualizations/pre_vae_umap.html -c disease -nn 8 &

MethylNet Commands:
