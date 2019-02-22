Dataset: GSE69138

Directory: /dartfs-hpc/rc/lab/C/ChristensenB/users/jlevy/projects/stroke/

include_col.txt (tab delimited):
age:ch1	Age
gender:ch1	Sex
sample type:ch1	Tissue

* cd /dartfs-hpc/rc/lab/C/ChristensenB/users/jlevy/projects/stroke && module load python/3-Anaconda && source activate methylnet_pro2

Preprocessing:

* nohup python preprocess.py download_geo -g GSE69138 &
* python preprocess.py create_sample_sheet -is ./geo_idats/GSE69138_clinical_info.csv -s geo -i geo_idats/ -os geo_idats/samplesheet.csv -d "stroke subtype:ch1" -c include_col.txt
* mkdir backup_clinical && mv ./geo_idats/GSE69138_clinical_info.csv backup_clinical
* python preprocess.py meffil_encode -is geo_idats/samplesheet.csv -os geo_idats/samplesheet.csv
* nohup time python preprocess.py preprocess_pipeline -n 35 -m  -pc -1 -bns 0.05 -pds 0.05 -bnc 0.05 -pdc 0.05 -sc -2 -sd 5 -i /dartfs-hpc/rc/lab/C/ChristensenB/users/jlevy/projects/stroke/geo_idats/ -o /dartfs-hpc/rc/lab/C/ChristensenB/users/jlevy/projects/stroke/preprocess_outputs/methyl_array.pkl & # time elapsed 11:59.47
* python utils.py remove_sex -i preprocess_outputs/methyl_array.pkl
* python preprocess.py na_report -i autosomal/methyl_array.pkl -o na_report/ # 0.16576042324017323%
* nohup python preprocess.py imputation_pipeline -i ./autosomal/methyl_array.pkl -ss -s fancyimpute -m KNN -k 5 -st 0.05 -ct 0.05 &
* python preprocess.py feature_select -n 300000
* mkdir visualizations
* nohup python visualizations.py transform_plot -o visualizations/pre_vae_umap.html -c disease -nn 8 &
* nohup python visualizations.py transform_plot -o visualizations/pre_vae_umap_sex.html -c Sex -nn 8 &
# NOTE: Before proceeding, get number of non-autosomal sites for pan and brain

OLD
* nohup python visualizations.py transform_plot -i autosomal/methyl_array.pkl -o visualizations/pre_vae_umap_no_sex.html -c Sex -nn 8 &
* nohup python visualizations.py transform_plot -i autosomal/methyl_array.pkl -o visualizations/pre_vae_umap_ns_disease.html -c disease -nn 8 &

MethylNet Commands:
