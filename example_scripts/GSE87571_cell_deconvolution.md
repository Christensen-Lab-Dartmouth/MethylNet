Dataset: GSE87571

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
methylnet-embed launch_hyperparameter_scan -sc Age -t -mc 0.84 -b 1. -g -j 200
```
Rerun top performing run to get final embeddings:
```
methylnet-embed launch_hyperparameter_scan -sc Age -t -g -n 1 -b 1.
```
Visualize VAE-Embedding:
```
pymethyl-visualize transform_plot -i embeddings/vae_methyl_arr.pkl -nn 8 -c Age
```

**Predictions using Transfer Learning**
Run 200 job hyperparameter scan for learning predictions on torque:
```
methylnet-predict launch_hyperparameter_scan -ic Age -t -mc 0.84 -g -j 200
```
Rerun top performing run to get final predictions:
```
methylnet-predict launch_hyperparameter_scan -ic Age -t -g -n 1
```
Visualize embeddings after training prediction model:
```
pymethyl-visualize transform_plot -i predictions/vae_mlp_methyl_arr.pkl -nn 8 -c Age
```

**Plot Embedding and Prediction Results**
```
pymethyl-visualize plot_cell_type_results -i predictions/results.csv -o cell_types_pred_vs_true.png
```

**MethylNet Interpretations**
If using torque:  
```
methylnet-torque run_torque_job -c "methylnet-interpret produce_shapley_data -mth gradient -ssbs 30 -ns 300 -bs 100 -rc 4. -r 0 -rt 0 -cn Age -nf 4000 -c" -gpu -a "source activate methylnet" -q gpuq -t 4 -n 1
```
Else (running with GPU 0):  
```
CUDA_VISIBLE_DEVICES=0 methylnet-interpret produce_shapley_data -mth gradient -ssbs 30 -ns 300 -bs 100 -rc 4. -r 0 -rt 0 -cn Age -nf 4000 -c
```

Extract spreadsheet of top overall CpGs:
```

```

Plot bar chart of top CpGs:
```

```

Find genomic context of these CpGs:
```

```

Run enrichment test with LOLA:
```

```

Plot results:
```

```




MethylNet Commands:


* python model_interpretability.py regenerate_top_cpgs -nf 4000 -a
* python predictions.py regression_report
* python model_interpretability.py split_hyper_hypo_methylation -s ./interpretations/shapley_explanations/shapley_reduced_data.p
* python model_interpretability.py shapley_jaccard -c all -i -s ./interpretations/shapley_explanations/shapley_data_by_methylation/hypo_shapley_data.p -o ./interpretations/shapley_explanations/top_cpgs_jaccard/hypo/ && python model_interpretability.py order_results_by_col -c Age -i ./interpretations/shapley_explanations/top_cpgs_jaccard/hypo/all_jaccard.csv -o ./interpretations/shapley_explanations/top_cpgs_jaccard/hypo/all_jaccard.sorted.csv
* python model_interpretability.py shapley_jaccard -c all -i -s ./interpretations/shapley_explanations/shapley_data_by_methylation/hyper_shapley_data.p -o ./interpretations/shapley_explanations/top_cpgs_jaccard/hyper/ && python model_interpretability.py order_results_by_col -c Age -i ./interpretations/shapley_explanations/top_cpgs_jaccard/hyper/all_jaccard.csv -o ./interpretations/shapley_explanations/top_cpgs_jaccard/hyper/all_jaccard.sorted.csv
* pymethyl-visualize plot_heatmap -c -m similarity -fs .4 -i ./interpretations/shapley_explanations/top_cpgs_jaccard/hypo/all_jaccard.sorted.csv -o ./interpretations/shapley_explanations/top_cpgs_jaccard/hypo/all_hypo_jaccard.png
* pymethyl-visualize plot_heatmap -c -m similarity -fs .4 -i ./interpretations/shapley_explanations/top_cpgs_jaccard/hyper/all_jaccard.sorted.csv -o ./interpretations/shapley_explanations/top_cpgs_jaccard/hyper/all_hyper_jaccard.png
* python model_interpretability.py bin_regression_shaps -c Age -n 16
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
* python model_interpretability.py extract_methylation_array -t train_val_test_sets/test_methyl_array_shap_binned.pkl -c Age_binned -s  interpretations/shapley_explanations/shapley_binned.p

* pymethyl-visualize plot_heatmap -fs .7 -i ./interpretations/shapley_explanations/top_cpgs_extracted_methylarr/beta.csv -o ./interpretations/biological_explanations/beta.png -c -col Age_binned &

* N_CPG=1000 # 100, 2000, 4000 # -g -n $N_CPG
* python model_interpretability.py extract_methylation_array -t train_val_test_sets/test_methyl_array_shap_binned.pkl -c Age_binned -s  interpretations/shapley_explanations/shapley_binned.p -co -n 4000 && pymethyl-utils write_cpgs -i interpretations/shapley_explanations/top_cpgs_extracted_methylarr/extracted_methyl_arr.pkl -c removal_cpgs/cpgs_library.pkl && pymethyl-utils set_part_array_background -i train_val_test_sets/test_methyl_array_shap_binned.pkl -c removal_cpgs/cpgs_library.pkl -o removal_cpgs/methyl_array.pkl && python predictions.py make_new_predictions -tp removal_cpgs/methyl_array.pkl -ic Age && python predictions.py regression_report -r new_predictions/results.p -o new_results/
# maybe extract cpgs for one class at a time and notice changes


* nohup python model_interpretability.py interpret_biology -ov -c all -s interpretations/shapley_explanations/shapley_binned.p -cgs horvath -ex & # hannum

pymethyl-utils subset_array -i train_val_test_sets/test_methyl_array_shap_binned.pkl -c ./interpretations/biological_explanations/cpg_library.pkl
pymethyl-utils pkl_to_csv -i subset/methyl_array.pkl -o subset/ -c Age_binned
pymethyl-visualize plot_heatmap -fs .7 -i ./subset/beta.csv -o ./subset/beta.png -c -x -col Age_binned &
xpymethyl-utils set_part_array_background -i train_val_test_sets/test_methyl_array.pkl -c ./interpretations/biological_explanations/cpg_library.pkl # only set top 1k overall or intersected horvath
# set them to background mean instead!!!!!
xpython predictions.py make_new_predictions -tp removal/methyl_array.pkl -ic Age
xpython predictions.py regression_report -r new_predictions/results.p -o new_results/

python model_interpretability.py interpret_biology -ov -c all -s interpretations/shapley_explanations/shapley_binned.p -cgs hannum -ex
# or run predictions with library omitted
# set_part_array_zeros then make_new_predictions, then classification/regression report

# test external set:
# download
# preprocess
# pymethyl-utils create_external_validation_set
# make_new_predictions then classification/regression report

# to-do search for missing cpgs, do same for other studies
# check overlap with different blood types
(13.92,22.0] top cpgs overlap with 0.0% of hannum cpgs
(22.0,30.0] top cpgs overlap with 0.0% of hannum cpgs
(30.0,38.0] top cpgs overlap with 1.45% of hannum cpgs
(38.0,46.0] top cpgs overlap with 4.35% of hannum cpgs
(46.0,54.0] top cpgs overlap with 39.13% of hannum cpgs
(62.0,70.0] top cpgs overlap with 78.26% of hannum cpgs
(70.0,78.0] top cpgs overlap with 79.71% of hannum cpgs
(78.0,86.0] top cpgs overlap with 82.61% of hannum cpgs
(86.0,94.0] top cpgs overlap with 65.22% of hannum cpgs
This cohort was around this age distribution...

Maybe look at horvath and epitoc age distribution for cohort.

(54.0,62.0] shared cpgs: 41/41.0
(70.0,78.0] shared cpgs: 55/55.0
(62.0,70.0] shared cpgs: 54/54.0
(22.0,30.0] shared cpgs: 0/0.0
(78.0,86.0] shared cpgs: 57/57.0
(38.0,46.0] shared cpgs: 3/3.0
(13.92,22.0] shared cpgs: 0/0.0
(30.0,38.0] shared cpgs: 1/1.0
(46.0,54.0] shared cpgs: 27/27.0
(86.0,94.0] shared cpgs: 45/45.0


nohup pymethyl-utils est_age -a epitoc -a horvath -a hannum -ac Age  &
pymethyl-utils concat_csv -i1 age_estimation/output_age_estimations.csv -i2 predictions/results.csv -o age_estimation/age_results.csv -a 1

pymethyl-utils rate_regression -i age_estimation/age_results.csv -c1 Age_pred -c2 Age_true
y_true: Age_true
y_pred: MethylNet
Mean Squared Error: 19.784308742853167
Mean Absolute Error: 3.4389054444444445
R2: 0.95581130721779

pymethyl-utils rate_regression -i age_estimation/age_results.csv -c1 Horvath.Est -c2 Age_true
y_true: Age_true
y_pred: Horvath.Est
Mean Squared Error: 33.310758037960525
Mean Absolute Error: 4.411414892854086
R2: 0.9255996824345102

pymethyl-utils rate_regression -i age_estimation/age_results.csv -c1 EpiTOC.Est -c2 Age_true
y_true: Age_true
y_pred: EpiTOC.Est
Mean Squared Error: 2618.6396767114074
Mean Absolute Error: 46.602052498991526
R2: -4.848789850861311

pymethyl-utils rate_regression -i age_estimation/age_results.csv -c1 Hannum.Est -c2 Age_true
y_true: Age_true
y_pred: Hannum.Est
Mean Squared Error: 45.395446876712896
Mean Absolute Error: 5.651159485416533
R2: 0.8986082616371
