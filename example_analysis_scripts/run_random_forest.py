from pymethylprocess.general_machine_learning import MachineLearning
from pymethylprocess.MethylationDataTypes import MethylationArray, MethylationArrays
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
import argparse

def run_rand_forest(train_pkl, val_pkl, test_pkl, classify=True, outcome_col='Disease_State', num_random_search=0):
    train_methyl_array, val_methyl_array, test_methyl_array = MethylationArray.from_pickle(train_pkl), MethylationArray.from_pickle(val_pkl), MethylationArray.from_pickle(test_pkl)
    model = RandomForestClassifier if classify else RandomForestRegressor
    model = MachineLearning(model,options={},grid=dict(n_estimators=[10,25,50,75,100,125,150,175,200],
                                                       criterion=['gini','entropy'],
                                                       max_features = ['auto','sqrt'],
                                                       max_depth = [None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                                       min_samples_split=[2,5,10],
                                                       min_samples_leaf=[1,2,4],
                                                       bootstrap = [True,False]),
                            n_eval=num_random_search)

    model.fit(train_methyl_array,val_methyl_array,outcome_col)

    y_pred = model.predict(test_methyl_array)

    original, std_err, (low_ci,high_ci) = model.return_outcome_metric(test_methyl_array, 'Disease_State', accuracy_score if classify else r2_score, run_bootstrap=True)

    results={'score':original,'Standard Error':std_err, '0.95 CI Low':low_ci, '0.95 CI High':high_ci}

    print('\n'.join(['{}:{}'.format(k,v) for k,v in results.items()]))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-tr', '--train_pkl', type=str, help='Train methylarray.', default='./train_val_test_sets/train_methyl_array.pkl', required=True)
    p.add_argument('-v', '--val_pkl', type=str, help='Val methylarray.', default='./train_val_test_sets/val_methyl_array.pkl', required=True)
    p.add_argument('-tt','--test_pkl', type=str, help='Test methylarray.', default='./train_val_test_sets/test_methyl_array.pkl', required=True)
    p.add_argument('-o','--outcome_col', type=str, help='Outcome column to train on.', default='Disease_State', required=True)
    p.add_argument('-c','--classify', action='store_true', help='Whether to perform classification.', required=False)
    p.add_argument('-n','--num_random_search', type=int, help='Number of random hyperparameter jobs.', default=0, required=False)
    args=p.parse_args()
    run_rand_forest(args.train_pkl,args.val_pkl,args.test_pkl,args.classify,args.outcome_col, args.num_random_search)
