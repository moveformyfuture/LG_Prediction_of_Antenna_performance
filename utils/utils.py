import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# seed 고정
def seed_everything(seed): 
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

class Config:
    seed = 42
    epochs = 200
    cv=10
    test_size = 0.2

# Y_Feature별 NRMSE의 총합
def lg_nrmse(gt, preds):
    """
    @Description: Metric used in this project
    @Params1: gt, pandas dataframe
    @Param2: preds, pandas dataframe
    @Return: nrmse score
    """
    preds = pd.DataFrame(preds)
    all_nrmse = []
    for idx in range(0,14):
        rmse = mean_squared_error(gt.iloc[:,idx], preds.iloc[:,idx], squared=False)
        nrmse = rmse/np.mean(np.abs(gt.iloc[:,idx]))
        all_nrmse.append(nrmse)
    score = 1.2 * np.sum(all_nrmse[:8]) + 1.0 * np.sum(all_nrmse[8:15]) # Y_01 ~ Y_08 까지 20% 가중치 부여
    return score

# Y_Feature 개별 NRMSE 계산
def lg_individual_nrmse(gt, preds):
    """
    @Description: Metric used in this project (individual)
    @Params1: gt, pandas dataframe
    @Param2: preds, pandas dataframe
    @Return: nrmse score
    """
    rmse = mean_squared_error(gt, preds, squared=False)
    nrmse = rmse/np.mean(np.abs(gt))
    return nrmse