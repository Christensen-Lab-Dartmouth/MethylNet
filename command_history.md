1. Pancancer  
    * python preprocess.py create_sample_sheet  
    * python preprocess.py get_categorical_distribution -k disease -k case_control | awk -F':' '{sum+=$2} END {print sum}'
    * Samples should be over 10000, only 19628/2 downloaded, and above command yielded 8901
    * clinical_info contains 11161 samples, so how many are redundant?
    * clinical info > 11k, ~9.8k downloaded, ~8.9k samples actually in pheno file
2. Brain Cancer
    * python preprocess.py meffil_encode -is geo_idats/geo_merged.csv -os geo_idats/geo_merged.csv
    * nohup python preprocess.py preprocess_pipeline -i ./geo_idats/ -n 15 -ss -d -m &
