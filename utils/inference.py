import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pytorch_tabnet.tab_model import TabNetRegressor 
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor

from utils.utils import lg_individual_nrmse

def inference_individual(train_x, train_y, test_x, col, best):
    kf = KFold(n_splits=10, random_state = 1, shuffle=True)
    preds, cv_loss = [], []
    for train_idx, test_idx in kf.split(train_x):
        X_train, X_valid = train_x.loc[train_idx], train_x.loc[test_idx]
        y_train, y_valid = train_y[col].loc[train_idx], train_y[col].loc[test_idx]
        model = LGBMRegressor(n_jobs = -1, random_state = 1, **best)

        model.fit(X_train, y_train)
        valid_pred = model.predict(X_valid)
        cv_loss.append(lg_individual_nrmse(y_valid, valid_pred))
        
        test_pred = model.predict(test_x)
        preds.append(np.array(test_pred))

    print(f"Cross Validation Loss :{np.mean(cv_loss)}")
    test_preds = list(np.array(preds).sum(aixs=0)/10)

    return pd.DataFrame(test_preds)

def inference_tabnet(features, targets, test, params, args):
    kfold = KFold(n_splits = args.folds, random_state = args.seed, shuffle = True)
    oof_predictions = np.zeros((39608, 14))
    test_predictions = np.zeros((39608, 14))

    for fold, (trn_ind, val_ind) in enumerate(kfold.split(features)):
        print('##################################################')
        print(f'\t\tTraining fold {fold + 1}\t\t')
        print('##################################################')
        X_train, X_val = features.iloc[trn_ind].values, features.iloc[val_ind].values
        y_train, y_val = targets.iloc[trn_ind].values, targets.iloc[val_ind].values

        clf =  TabNetRegressor(**params)

        clf.fit(
            X_train, 
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric = ['mse'],
            loss_fn = nn.MSELoss(),
            max_epochs = args.max_epochs,
            patience = args.patience,
            batch_size = args.batch_size, 
            virtual_batch_size = args.virtual_batch_size,
            num_workers = args.num_workers,
            drop_last = False,
                      )
        
    saved_filepath = clf.save_model(f'./fold_{fold + 1}')
    oof_predictions[val_ind] = clf.predict(X_val)
    test_predictions+=clf.predict(test.values)/args.folds

    return test_predictions