from pymethylprocess.general_machine_learning import MachineLearning
from pymethylprocess.MethylationDataTypes import MethylationArray, MethylationArrays
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


p = argparse.ArgumentParser()
p.add_argument('-t', '--github', type=str, help='Install from github.', required=True)
p.add_argument('-b', '--bioconductor', type=str, help='Install from bioconductor.', required=True)
p.add_argument('-p','--packages', type=str, help='List of packages.', required=True)
p.add_argument('-p','--packages', type=str, help='List of packages.', required=True)
args=p.parse_args()

train_pkl=''
val_pkl=''
test_pkl=''
train_methyl_array, val_methyl_array, test_methyl_array = MethylationArray.from_pickle(train_pkl), MethylationArray.from_pickle(val_pkl), MethylationArray.from_pickle(test_pkl)

model = MachineLearning(RandomForestClassifier,options={},grid=dict(
                n_estimators=[10,25,50,75,100,125,150,175,200],
                criterion=['gini','entropy'],
                max_features = ['auto','sqrt'],
                max_depth = [None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                min_samples_leaf=[2,5,10],
                min_samples_leaf=[1,2,4],
                bootstrap = [True,False]
                ))

model.fit(train_methyl_array,val_methyl_array,'Disease_State')


y_pred = model.predict(test_methyl_array)

original, std_err, (low_ci,high_ci) = model.return_outcome_metric(test_methyl_array, 'Disease_State', accuracy_score, run_bootstrap=True)

results={accuracy:original,'Standard Error':std_err, '0.95 CI Low':low_ci, '0.95 CI High':high_ci}

print('\n'.join(['{}:{}'.format(k,v) for k,v in results.items()]))
