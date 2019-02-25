Pancancer:
pymethyl-preprocess split_preprocess_input_by_subtype -i ../pancancer/final_samplesheet.csv -d
nohup time pymethyl-preprocess batch_deploy_preprocess -n 6 -c 5 -r -m & 4 hours and 15 minutes


Braincancer:
pymethyl-preprocess split_preprocess_input_by_subtype -i ../brain_cancer/final_samplesheet.csv -d
nohup time pymethyl-preprocess batch_deploy_preprocess -n 6 -c 5 -r -m & # 2 hours and 20 minutes
