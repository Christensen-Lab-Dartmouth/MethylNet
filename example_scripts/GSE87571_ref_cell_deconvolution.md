Dataset: GSE87571

**Install Instructions**
See README.md

**Preprocessing**
Run commands from: https://github.com/Christensen-Lab-Dartmouth/PyMethylProcess/blob/master/example_scripts/GSE87571.md

**Reference-Based Estimation of Cell Type Proportions**

```
pymethyl-utils ref_estimate_cell_counts -ro geo_idats/ -a IDOL
```
Formatting output:
```
python
>>> df=pd.read_csv('added_cell_counts/cell_type_estimates.csv')
  df.pivot('Var1','Var2','Freq').to_csv('added_cell_counts/cell_types_adjusted.csv')
>>> exit()
scp added_cell_counts/cell_types_adjusted.csv .
```
Overwrite pheno data in train, test, val Methylation Arrays with cell-type proportions:
```
pymethyl-utils overwrite_pheno_data -i old_train_val_test_sets/train_methyl_array.pkl -o train_val_test_sets/train_methyl_array.pkl --input_formatted_sample_sheet cell_types_adjusted.csv
pymethyl-utils overwrite_pheno_data -i old_train_val_test_sets/val_methyl_array.pkl -o train_val_test_sets/val_methyl_array.pkl --input_formatted_sample_sheet cell_types_adjusted.csv
pymethyl-utils overwrite_pheno_data -i old_train_val_test_sets/test_methyl_array.pkl -o train_val_test_sets/test_methyl_array.pkl --input_formatted_sample_sheet cell_types_adjusted.csv
```
Visualize Results:
```
pymethyl-visualize plot_cell_type_results -i cell_types_adjusted.csv -o cell_types_adjusted.png
nohup pymethyl-visualize transform_plot -o visualizations/pre_vae_umap.html -c Age -nn 8 &
nohup pymethyl-visualize transform_plot -o visualizations/pre_vae_umap_sex.html -c Sex -nn 8 &
nohup pymethyl-visualize transform_plot -o visualizations/pre_vae_umap_CD4T.html -c CD4T -nn 8 &
nohup pymethyl-visualize transform_plot -o visualizations/pre_vae_umap_CD8T.html -c CD8T -nn 8 &
nohup pymethyl-visualize transform_plot -o visualizations/pre_vae_umap_NK.html -c NK -nn 8 &
nohup pymethyl-visualize transform_plot -o visualizations/pre_vae_umap_Bcell.html -c Bcell -nn 8 &
nohup pymethyl-visualize transform_plot -o visualizations/pre_vae_umap_gMDSC.html -c gMDSC -nn 8 &
nohup pymethyl-visualize transform_plot -o visualizations/pre_vae_umap_Neu.html -c Neu -nn 8 &
```

**Embedding using VAE**
Run 200 job hyperparameter scan for learning embeddings on torque (remove -t option to run local, same for prediction jobs below):  
```
methylnet-embed launch_hyperparameter_scan -cu -sc Age -t -mc 0.84 -b 1. -g -j 200 -a "module load python/3-Anaconda && source activate methylnet_pro2"
```
Rerun top performing run to get final embeddings:
```
methylnet-embed launch_hyperparameter_scan -cu -sc Age -t -g -n 1 -b 1. -a "module load python/3-Anaconda && source activate methylnet_pro2"
```

**Predictions using Transfer Learning**
Run 200 job hyperparameter scan for learning predictions on torque:
```
methylnet-predict launch_hyperparameter_scan -ic Bcell -ic CD4T -ic CD8T -ic Mono -ic NK -ic Neu -t -mc 0.84 -g -cu -j 200 -a "module load python/3-Anaconda && source activate methylnet_pro2"
```
Rerun top performing run to get final predictions:
```
methylnet-predict launch_hyperparameter_scan -ic Bcell -ic CD4T -ic CD8T -ic Mono -ic NK -ic Neu -t -mc 0.84 -cu -t -g -n 1 -a "module load python/3-Anaconda && source activate methylnet_pro2"
```

**Plot Embedding and Prediction Results**
```
pymethyl-visualize plot_cell_type_results -i predictions/results.csv -o cell_types_pred_vs_true.png
```

**Plot results:**
```
methylnet-predict regression_report
methylnet-visualize plot_training_curve -t embeddings/training_val_curve.p -vae -o results/embed_training_curve.png -thr 2e8
methylnet-visualize plot_training_curve -thr 2e6
```

**MethylNet Interpretations**
If using torque:  
```
methylnet-torque run_torque_job -c "methylnet-interpret produce_shapley_data produce_shapley_data -mth gradient -ssbs 30 -ns 300 -bs 100 -rc 4. -r 0 -rt 0 -cn Bcell -cn CD4T -cn CD8T -cn Mono -cn NK -cn Neu -nf 4000" -a "module load python/3-Anaconda && source activate methylnet_pro2"
```
Else (running with GPU 0):  
```
CUDA_VISIBLE_DEVICES=0 methylnet-interpret produce_shapley_data produce_shapley_data -mth gradient -ssbs 30 -ns 300 -bs 100 -rc 4. -r 0 -rt 0 -cn Bcell -cn CD4T -cn CD8T -cn Mono -cn NK -cn Neu -nf 4000
```

Extract spreadsheet of top overall CpGs:
```
methylnet-interpret return_shap_values -c all -hist  -o  interpretations/shap_results/ -s interpretations/shapley_explanations/shapley_data.pkl -log &
methylnet-interpret return_shap_values -c all -hist -abs -o interpretations/abs_shap_results/ -log -s interpretations/shapley_explanations/shapley_data.pkl &
```

Plot bar chart of top CpGs:
```
pymethyl-visualize plot_heatmap -m distance -fs .6 -i interpretations/shap_results/returned_shap_values_corr_dist.csv -o ./interpretations/shap_results/distance_cpgs.png -x -y -c &
pymethyl-visualize plot_heatmap -m distance -fs .6 -i interpretations/abs_shap_results/returned_shap_values_corr_dist.csv -o ./interpretations/abs_shap_results/distance_cpgs.png -x -y -c &
```

Plot bar chart of top CpGs:
```
pymethyl-visualize plot_heatmap -m distance -fs .6 -i interpretations/shap_results/returned_shap_values_corr_dist.csv -o ./interpretations/shap_results/distance_cpgs.png -x -y -c &
pymethyl-visualize plot_heatmap -m distance -fs .6 -i interpretations/abs_shap_results/returned_shap_values_corr_dist.csv -o ./interpretations/abs_shap_results/distance_cpgs.png -x -y -c &
```

**Creation Test Data for Test Pipeline:**
```
from pymethylprocess.MethylationDataTypes import MethylationArray
sample_p = 0.35
methyl_array=MethylationArray.from_pickle("train_methyl_array.pkl")
methyl_array.subsample("Age",frac=None,n_samples=int(methyl_array.pheno.shape[0]*sample_p)).write_pickle("train_methyl_array_subsampled.pkl")
methyl_array=MethylationArray.from_pickle("val_methyl_array.pkl")
methyl_array.subsample("Age",frac=None,n_samples=int(methyl_array.pheno.shape[0]*sample_p)).write_pickle("val_methyl_array_subsampled.pkl")
methyl_array=MethylationArray.from_pickle("test_methyl_array.pkl")
methyl_array.subsample("Age",frac=None,n_samples=int(methyl_array.pheno.shape[0]*sample_p)).write_pickle("test_methyl_array_subsampled.pkl")
pymethyl-utils feature_select_train_val_test -n 25000
```
