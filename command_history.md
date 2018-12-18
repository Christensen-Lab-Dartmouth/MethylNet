1. Pancancer  
    * python preprocess.py create_sample_sheet  
    * python preprocess.py meffil_encode
    * python preprocess.py get_categorical_distribution -k disease -k case_control | awk -F':' '{sum+=$2} END {print sum}'
    * Samples should be over 10000, only 19628/2 downloaded, and above command yielded 8901
    * clinical_info contains 11161 samples, so how many are redundant?
    * clinical info > 11k, ~9.8k downloaded, ~8.9k samples actually in pheno file
    * nohup python preprocess.py preprocess_pipeline -n 25 -ss -m &
    * python preprocess.py combine_methylation_arrays -d ./preprocess_outputs/ -e OV
    * nohup python preprocess.py imputation_pipeline -i ./combined_outputs/methyl_array.pkl -o final_preprocessed/methyl_array.pkl -n 200000 -s simple -m Mean &
    * nohup python visualizations.py transform_plot  -o prevae_visual.html -nn 10 &
    * nohup python embedding.py perform_embedding -n 300 -hlt 500 -kl 27 -b 4. -s warm_restarts -lr 1e-4 -bce -e 200 &
    * nohup python visualizations.py transform_plot -o vae_latest.html -i ./embeddings/vae_methyl_arr.pkl -nn 8 -d 0.1  &
2. Brain Cancer
    * python preprocess.py merge_sample_sheets -s1 ../../methylation_analyses/geo_pheno_data_backup/geo_minfi.csv -s2 ../../methylation_analyses/geo_pheno_data_backup/geo_concat.csv -os ./geo_idats/geo_merged.csv  
    * python preprocess.py meffil_encode -is geo_idats/geo_merged.csv -os geo_idats/geo_merged.csv
    * nohup python preprocess.py preprocess_pipeline -i ./geo_idats/geo_merged.csv -n 40 -ss -d -m &
    nohup python preprocess.py imputation_pipeline -i ./preprocess_outputs/methyl_array.pkl -o final_preprocessed/methyl_array.pkl -n 200000 -s simple -m Mean &

    1510  nohup python visualizations.py transform_plot  -o prevae_visual.html -nn 10 &
    1511  nohup python visualizations.py transform_plot  -o prevae_visual_supervised.html -nn 10 -s &
    1514  nohup python visualizations.py transform_plot  -o prevae_visual_supervised_disease_only.html -nn 10 -s -c disease_only &

    python embedding.py perform_embedding -lr 1e-4 -wd 0.01 -hlt 500 -n 100 -kl 20
    python visualizations.py transform_plot -o vae.html -i ./embeddings/vae_methyl_arr.pkl
    nohup python embedding.py perform_embedding -lr 5e-5 -wd 0.01 -hlt 500 -n 100 -kl 20 -b 5. &
3. python preprocess.py create_sample_sheet -is clinical_info.csv  && python preprocess.py meffil_encode && python preprocess.py split_preprocess_input_by_subtype && python preprocess.py  batch_deploy_preprocess -n 1 -i preprocess_outputs/ -m
3. Setup
    * brew tap caskroom/versions
brew cask install java8
    * docker build . -t methylnet:0.1
4. embed
        * python embedding.py perform_embedding -n 300 -hlt 500 -kl 5 -b 4. -s warm_restarts -lr 1e-4 -bce
        * nohup python embedding.py perform_embedding -n 70 -hlt 200 -kl 300 -b 3. -s warm_restarts -lr 1e-3 -bce -e 3000 >nohup2.out&
        * nohup python embedding.py perform_embedding -n 60 -hlt 300 -kl 1000 --t_mult 1.1 -b 5. -s warm_restarts -lr 1e-3 -bce -e 2500 >nohup2.out&
