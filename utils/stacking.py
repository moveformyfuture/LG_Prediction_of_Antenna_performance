import numpy as np
from sklearn.model_selection import train_test_split, KFold
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from ngboost import NGBRegressor
from lightgbm import LGBMRegressor

def get_stacking_base_datasets(model, train_x, train_y, col,test, params):
    kf = KFold(n_splits=10, shuffle=False)
    train_fold_pred = np.zeros((train_x.shape[0],1))
    test_pred = np.zeros((test.shape[0],10))
    
    
    for folder_counter, (train_index, valid_index) in enumerate(kf.split(train_x)):
        print('Fold : ', folder_counter, ' Start')
        X_tr = train_x.loc[train_index]
        y_tr = train_y[col].loc[train_index]
        X_te = train_x.loc[valid_index] 
        
        if model == 'cat':
          model = CatBoostRegressor(random_state=1,
                                    **params)
        
        elif model == 'extra':
          model = ExtraTreesRegressor(random_state=1, 
                                      **params)

        elif model == 'ngbr':
          model = NGBRegressor(random_state = 1)
        
        elif model == 'lgbm':
          model = LGBMRegressor(random_state=1, n_jobs=-1, 
                                **params)

        model.fit(X_tr, y_tr)
        train_fold_pred[valid_index, :] = model.predict(X_te).reshape(-1,1) 
        test_pred[:, folder_counter] = model.predict(test) 
        
    test_pred_mean = np.mean(test_pred, axis=1).reshape(-1,1)
    
    return train_fold_pred, test_pred_mean 