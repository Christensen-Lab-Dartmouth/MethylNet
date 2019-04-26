from pymethylprocess.general_machine_learning import MachineLearning
from pymethylprocess.MethylationDataTypes import MethylationArray, MethylationArrays
from sklearn.linear_model import SGDClassifier,ElasticNet
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd, numpy as np
import argparse
import pickle

def run_elasticnet(train_pkl, val_pkl, test_pkl, series=False, outcome_col='Disease_State', num_random_search=0):
    train_methyl_array, val_methyl_array, test_methyl_array = MethylationArray.from_pickle(train_pkl), MethylationArray.from_pickle(val_pkl), MethylationArray.from_pickle(test_pkl)
    model = SGDClassifier
    model = MachineLearning(model,options={'penalty':'elasticnet','verbose':3,'n_jobs':35},grid={"max_iter": [20,50,100,200,500],
                      "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                      "l1_ratio": np.arange(0.0, 1.0, 0.1)},
                            n_eval=num_random_search,
                            series=series,
                            labelencode=True,
                            verbose=True)

    sklearn_model=model.fit(train_methyl_array,val_methyl_array,outcome_col)

    y_pred = model.predict(test_methyl_array)

    original, std_err, (low_ci,high_ci) = model.return_outcome_metric(test_methyl_array, 'Disease_State', accuracy_score if classify else r2_score, run_bootstrap=True)

    results={'score':original,'Standard Error':std_err, '0.95 CI Low':low_ci, '0.95 CI High':high_ci}

    pd.DataFrame(y_pred[:,np.newaxis],columns=['ElasticNetPredictions']).to_csv('ElasticNetPredictions.csv')

    print('\n'.join(['{}:{}'.format(k,v) for k,v in results.items()]))

    pickle.dump(sklearn_model,open('sklearn_model.p','wb'))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('-tr', '--train_pkl', type=str, help='Train methylarray.', default='./train_val_test_sets/train_methyl_array.pkl', required=True)
    p.add_argument('-v', '--val_pkl', type=str, help='Val methylarray.', default='./train_val_test_sets/val_methyl_array.pkl', required=True)
    p.add_argument('-tt','--test_pkl', type=str, help='Test methylarray.', default='./train_val_test_sets/test_methyl_array.pkl', required=True)
    p.add_argument('-o','--outcome_col', type=str, help='Outcome column to train on.', default='disease', required=True)
    p.add_argument('-s','--series', action='store_true', help='Disable multiprocessing.', required=False)
    p.add_argument('-n','--num_random_search', type=int, help='Number of random hyperparameter jobs.', default=0, required=False)
    args=p.parse_args()
    run_elasticnet(args.train_pkl,args.val_pkl,args.test_pkl,args.series,args.outcome_col, args.num_random_search)

if __name__ == '__main__':
    main()
