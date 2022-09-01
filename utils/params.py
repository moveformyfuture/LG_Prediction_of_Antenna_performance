# 기본 modules
import pandas as pd
import random
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import tqdm

# 머신러닝 modules
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold

from lightgbm import LGBMRegressor
from ngboost import NGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Lasso, Ridge
from catboost import CatBoostRegressor, Pool
from skranger.ensemble import RangerForestRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from ngboost.scores import LogScore

from hyperopt import fmin, hp, tpe
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance

# 모듈화된 함수 사용
import utils.preprocessing as preprocessing
import utils.utils as utils
import utils.stacking as stk

# Parameter Setting
def lgbm_objective(params):
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
        verbose = 100,
        **params
    )

    losses = np.sqrt(-cross_val_score(model, train_x, train_y['Y_01'], cv=utils.Config.cv, scoring='neg_mean_squared_error')) # cross_val_score : 교차검증
    losses = losses / np.mean(np.abs(train_y['Y_01']))
    print("NRMSE Loss {:.5f} params {}".format(losses.mean(), params))
    return losses.mean()

def xgb_objective(params):
    params = {

    }

    model = XGBRegressor(
        n_jobs = -1,
        verbose = 100,
        random_state = 1,
        **params
    )

    losses = np.sqrt(-cross_val_score(model, train_x, train_y['Y_01'], cv=Config.cv, scoring='neg_mean_squared_error'))
    losses = losses / np.mean(np.abs(train_y['Y_01']))
    print("NRMSE Loss {:.5f} params {}".format(losses.mean(), params))
    return losses.mean()

def cat_objective(params):
    params = {
        'n_estimators': int(params['n_estimators']),
        'depth': int(params['depth']),
        'learning_rate': params['learning_rate'],   
        'l2_leaf_reg': params['l2_leaf_reg'],
        'max_bin': int(params['max_bin']),
        'min_data_in_leaf': int(params['min_data_in_leaf']),
        'random_strength': params['random_strength'],
        'fold_len_multiplier': params['fold_len_multiplier'],
        
    }

    model = CatBoostRegressor(
        logging_level='Silent',
        **params
    )

    losses = np.sqrt(-cross_val_score(model, train_x, train_y['Y_01'], cv=utils.Config.cv, scoring='neg_mean_squared_error'))
    losses = losses / np.mean(np.abs(train_y['Y_01']))
    print("NRMSE Loss {:.5f} params {}".format(losses.mean(), params))
    return losses.mean()

def extra_objective(params):
    params = {
        'n_estimators': int(params['n_estimators']),
        'max_depth': int(params['max_depth']),
        'min_samples_split': int(params['min_samples_split']),
        'min_samples_leaf': int(params['min_samples_leaf']),
        'min_weight_fraction_leaf': params['min_weight_fraction_leaf'],
        'max_features': params['max_features'],
        'max_leaf_nodes': int(params['max_leaf_nodes']),
        'min_impurity_decrease': params['min_impurity_decrease'],
        'bootstrap': params['bootstrap'],
        'ccp_alpha': params['ccp_alpha'],  
    }

    model = ExtraTreesRegressor(
        n_jobs = -1,
        verbose = 0,
        random_state = 1,
        **params
    )

    losses = np.sqrt(-cross_val_score(model, train_x, train_y['Y_01'], cv=Config.cv, scoring='neg_mean_squared_error'))
    losses = losses / np.mean(np.abs(train_y['Y_01']))
    print("NRMSE Loss {:.5f} params {}".format(losses.mean(), params))
    return losses.mean()

def ngbr_objective(params):
    params = {
        'n_estimators': int(params['n_estimators']),
        'learning_rate': params['learning_rate'],
        'natural_gradient': params['natural_gradient'],
        'col_sample': float(params['col_sample']),
        'minibatch_frac': float(params['minibatch_frac']),
        'tol': float(params['tol']),
    }

    model = NGBRegressor(
        verbose = 100,
        random_state = 1,
        **params
    )

    losses = np.sqrt(-cross_val_score(model, train_x, train_y['Y_01'], cv=Config.cv, scoring='neg_mean_squared_error'))
    losses = losses / np.mean(np.abs(train_y['Y_01']))
    print("NRMSE Loss {:.5f} params {}".format(losses.mean(), params))
    return losses.mean()

