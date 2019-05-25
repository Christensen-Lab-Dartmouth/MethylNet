Dataset: GSE87571

Dataset: GSE87571

**Install Instructions**
See README.md

**Preprocessing**
Run commands from: https://github.com/Christensen-Lab-Dartmouth/PyMethylProcess/blob/master/example_scripts/GSE87571.md

**Embedding using VAE**
Run 200 job hyperparameter scan for learning embeddings on torque (remove -t option to run local, same for prediction jobs below):  
```
methylnet-embed launch_hyperparameter_scan -cu -sc Age -t -mc 0.84 -b 1. -g -j 400
```
Rerun top performing run to get final embeddings:
```
methylnet-embed launch_hyperparameter_scan -cu -sc Age -t -g -n 1 -b 1.
```
Visualize VAE-Embedding:
```
pymethyl-visualize transform_plot -i embeddings/vae_methyl_arr.pkl -nn 8 -c Age
```

**Predictions using Transfer Learning**
Run 200 job hyperparameter scan for learning predictions on torque:
```
methylnet-predict launch_hyperparameter_scan -cu -ic Age -t -mc 0.84 -g -j 200 -a "module load python/3-Anaconda && source activate methylnet_pro2"
```
Rerun top performing run to get final predictions:
```
methylnet-predict launch_hyperparameter_scan -cu -ic Age -t -g -n 1 -a "module load python/3-Anaconda && source activate methylnet_pro2"
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

Bin Ages:
```
methylnet-interpret bin_regression_shaps -c Age -n 5
```

Extract spreadsheet of top overall CpGs:
```
methylnet-interpret return_shap_values -log -c all -hist -s interpretations/shapley_explanations/shapley_binned.p -o  interpretations/shap_results/ &
methylnet-interpret return_shap_values -log -c all -hist -abs -o interpretations/abs_shap_results/ -s interpretations/shapley_explanations/shapley_binned.p &

```

Overlap with Clock CpGs:

```
methylnet-interpret interpret_biology -ov -c all -s interpretations/shapley_explanations/shapley_binned.p -cgs hannum

python
import pandas as pd, matplotlib, matplotlib.pyplot as plt, seaborn as sns, numpy as np
sns.set(font_scale=0.7)
clock_cpg_results = list(map(lambda line: [w.replace('(','').replace(']','').replace(',','-').replace('%','') for w in line.split() if w[-1] in [']','%']],"""(13.92,24.0] top cpgs overlap with 0.0% of hannum cpgs
(24.0,34.0] top cpgs overlap with 0.0% of hannum cpgs
(34.0,44.0] top cpgs overlap with 1.45% of hannum cpgs
(44.0,54.0] top cpgs overlap with 34.78% of hannum cpgs
(54.0,64.0] top cpgs overlap with 68.12% of hannum cpgs
(64.0,74.0] top cpgs overlap with 79.71% of hannum cpgs
(74.0,84.0] top cpgs overlap with 82.61% of hannum cpgs
(84.0,94.0] top cpgs overlap with 73.91% of hannum cpgs""".splitlines()))
plt.figure()
df=pd.DataFrame(clock_cpg_results,columns=['Age','Percent Hannum CpGs'])
df.iloc[:,1]=df.iloc[:,1].astype(float)
sns.barplot('Age','Percent Hannum CpGs',data=df, palette="Blues_d") # np.arange(df.shape[0])
plt.savefig('hannum_overlap.png',dpi=300)
```

Plot bar chart of top CpGs:
```
pymethyl-visualize plot_heatmap -m distance -fs .6 -i interpretations/shap_results/returned_shap_values_corr_dist.csv -o ./interpretations/shap_results/distance_cpgs.png -x -y -c &
pymethyl-visualize plot_heatmap -m distance -fs .6 -i interpretations/abs_shap_results/returned_shap_values_corr_dist.csv -o ./interpretations/abs_shap_results/distance_cpgs.png -x -y -c &
```

**Plot results:**
```
methylnet-predict regression_report
methylnet-visualize plot_training_curve -t embeddings/training_val_curve.p -vae -o results/embed_training_curve.png -thr 2e8
methylnet-visualize plot_training_curve -thr 2e6
```

**Estimate Age/Rates Using Other Clocks:**
```
nohup pymethyl-utils est_age -a epitoc -a horvath -a hannum -ac Age  -i for_curtis/test_set/test_methyl_array.pkl &
pymethyl-utils concat_csv -i1 age_estimation/output_age_estimations.csv -i2 predictions/results.csv -o age_estimation/age_results.csv -a 1
```

**Score Regression:**
```
pymethyl-utils rate_regression -i age_estimation/age_results.csv -c1 Age_pred -c2 Age_true
pymethyl-utils rate_regression -i age_estimation/age_results.csv -c1 Horvath.Est -c2 Age_true
pymethyl-utils rate_regression -i age_estimation/age_results.csv -c1 Hannum.Est -c2 Age_true
```
