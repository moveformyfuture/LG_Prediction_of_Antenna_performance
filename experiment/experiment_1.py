import os
import sys
import numpy as np
import pandas as pd

import utils.preprocessing as preprocessing
import utils.utils as utils
from utils.inference import inference_individual

from hyperopt import fmin, hp, tpe
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score


result_preds = pd.DataFrame()

def single_objective(params):
    params = {
        'n_estimators': int(params['n_estimators']),
        'max_depth': int(params['max_depth']),
        'num_leaves': int(params['num_leaves']),
        'min_child_samples': int(params['min_child_samples']),
        'colsample_bytree': '{:.5f}'.format(params['colsample_bytree']),
        'subsample': '{:.5f}'.format(params['subsample']),
        'min_split_gain': '{:.5f}'.format(params['min_split_gain']),
        'scale_pos_weight': '{:.5f}'.format(params['scale_pos_weight']),
        'reg_alpha': '{:.5f}'.format(params['reg_alpha']),
        'reg_lambda': '{:.5f}'.format(params['reg_lambda']),
        'learning_rate': '{:.5f}'.format(params['learning_rate']),   
    }

    model = LGBMRegressor(
        n_jobs = -1,
        random_state = 1,
        **params
    )

    losses = np.sqrt(-cross_val_score(model, train_x, train_y['Y_01'], cv=10, scoring='neg_mean_squared_error')) ## Plz Change Column name (Y_01 ~ Y_14)
    losses = losses / np.mean(np.abs(train_y['Y_01'])) ## Plz Change Column name (Y_01 ~ Y_14)
    print("NRMSE Loss {:.5f} params {}".format(losses.mean(), params))
    return losses.mean()

space = {
    'n_estimators' : hp.quniform('n_estimators', 100, 1500, 1),
    'max_depth': hp.quniform('max_depth', 5, 250, 1),
    'num_leaves': hp.quniform('num_leaves', 20, 200, 5),
    'min_child_samples': hp.quniform('min_child_samples', 10, 150, 5),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
    'subsample': hp.uniform('subsample', 0.3, 1.0),
    'min_split_gain': hp.uniform('min_split_gain', 0, 0.7),
    'scale_pos_weight': hp.uniform('scale_pos_weight', 1, 10),
    'reg_alpha': hp.uniform('reg_alpha', 0, 500),
    'reg_lambda': hp.uniform('reg_lambda', 0, 500),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.5)),
}

utils.seed_everything(42)
train_x, train_y, test_df = preprocessing.load_data('./train.csv', './test.csv')

best = fmin(fn = single_objective,
            space = space,
            algo = tpe.suggest,
            max_evals = 200)

test_preds = inference_individual(train_x, train_y, test_df, 'Y_01', best) ## Plz Change Column name (Y_01 ~ Y_14)
result_preds = pd.concat([result_preds, test_preds], axis=1)