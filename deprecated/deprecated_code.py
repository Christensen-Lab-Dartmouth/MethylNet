# from model model_interpretability
        if feature_selection:
            self.cpg_global_shapley_scores = np.abs(shap_values).mean(0).mean(0)
        else:
            top_cpgs=[]

            if test_methyl_array.pheno.shape[0] == 1 and prediction_classes != None:
                sample=list(test_arr.pheno.index)[0]
                prediction_class_idx = np.flatnonzero((np.array(prediction_classes) == test_methyl_array.pheno.loc[sample,interest_col]).astype(int))
                top_feature_idx = np.argsort(shap_values[prediction_class_idx, ...]*-1)[:n_top_features]
                top_cpgs.append(np.vstack([cpgs[top_feature_idx],np.abs(shap_values[prediction_class_idx, ...])[top_feature_idx]]).T)
                if summary_plot_file:
                    plt.figure()
                    shap.force_plot(self.explainer.expected_value,shap_values[prediction_class_idx,...],test_methyl_array.beta,show=False)
                    plt.savefig('{}_{}_{}.png'.format(summary_plot_file[:summary_plot_file.rfind('.')],sample,test_arr.pheno.loc[sample,interest_col]))
            else:
                for i in range(len(shap_values)): # list of top cpgs, one per class
                    if pred_class != None:
                        if prediction_classes != None and prediction_classes[i]==pred_class:
                            shap_values_idx = np.flatnonzero((test_methyl_array.pheno[interest_col].values == prediction_classes[i]).astype(int))
                            top_feature_idx = np.argsort(shap_values[i, shap_values_idx, :].mean(0)*-1)[:n_top_features]
                            top_cpgs.append(np.vstack([cpgs[top_feature_idx],np.abs(shap_values[i, ...]).mean(0)[top_feature_idx]]).T) # -np.abs(shap_values) # should I only consider the positive cases
                    else:
                        if prediction_classes != None:
                            shap_values_idx = np.flatnonzero((test_methyl_array.pheno[interest_col].values == prediction_classes[i]).astype(int))
                            top_feature_idx = np.argsort(shap_values[i, shap_values_idx, :].mean(0)*-1)[:n_top_features]
                        else:
                            top_feature_idx=np.argsort(np.abs(shap_values[i, ...]).mean(0)*-1)[:n_top_features]
                        top_cpgs.append(np.vstack([cpgs[top_feature_idx],np.abs(shap_values[i, ...]).mean(0)[top_feature_idx]]).T) # -np.abs(shap_values) # should I only consider the positive cases
            self.top_cpgs = top_cpgs # return shapley values
            self.shapley_values = [pd.DataFrame(shap_values[i, ...],index=test_methyl_array.beta.index,columns=cpgs) for i in range(shap_values.shape[0])]

        if summary_plot_file and not (test_arr.pheno.shape[0] == 1):
            import matplotlib.pyplot as plt
            if feature_selection:
                plt.figure()
                cpg_idx=np.argsort(self.cpg_global_shapley_scores*-1)[:40]
                shap_values_reduced = [shap_values[i, :, cpg_idx] for i in range(shap_values.shape[0])]
                shap.summary_plot(shap_values_reduced, test_methyl_array.beta.iloc[:,cpg_idx], plot_type='bar', max_display=min(test_methyl_array.beta.shape[1],40), class_names=prediction_classes if (prediction_classes != None and not feature_selection) else None, show=False)
                plt.savefig('{}_feature_selection.png'.format(summary_plot_file[:summary_plot_file.rfind('.')]))
            elif pred_class != None and prediction_classes != None:
                plt.figure()
                cpg_idx = self.top_cpgs[0][:,0]
                if 40 < len(cpg_idx):
                    cpg_idx = cpg_idx[:40]
                cpg_idx = np.isin(cpgs,cpg_idx)
                shap_values_idx = np.flatnonzero((test_methyl_array.pheno[interest_col].values == prediction_classes[i]).astype(int))
                shap_values_reduced = [shap_values[shap_values_idx, :, cpg_idx]]
                shap.summary_plot(shap_values_reduced, test_methyl_array.beta.iloc[:,cpg_idx], plot_type='bar', class_names=[pred_class], show=False)
                plt.savefig('{}_{}.png'.format(summary_plot_file[:summary_plot_file.rfind('.')],pred_class))
            else:
                for i in range(len(self.top_cpgs)):
                    plt.figure()
                    cpg_idx = self.top_cpgs[i][:,0]
                    if 40 < len(cpg_idx):
                        cpg_idx = cpg_idx[:40]
                    cpg_idx = np.isin(cpgs,cpg_idx)
                    shap_values_reduced = [shap_values[i, :, cpg_idx] for i in range(shap_values.shape[0])]
                    shap.summary_plot(shap_values_reduced, test_methyl_array.beta.iloc[:,cpg_idx], plot_type='bar', class_names=prediction_classes if prediction_classes != None else None, show=False)
                    plt.savefig('{}_{}.png'.format(summary_plot_file[:summary_plot_file.rfind('.')],prediction_classes[i] if prediction_classes != None else i))
