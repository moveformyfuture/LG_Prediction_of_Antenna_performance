import argparse
import pandas as pd
from utils.utils import seed_everything
from utils.preprocessing import load_data
from lightgbm import LGBMRegressor

if __name__ == '__main__':
    def parser():
        parser = argparse.ArgumentParser(description='tabnet experiment')
        parser.add_argument('--train', type=str, default='./train.csv')
        parser.add_argument('--test', type=str, default='./test.csv')
        parser.add_argument('--sub', type=str, default='./submssion.csv')
        parser.add_argument('--seed', type = int, default=42)
        args = parser.parse_args()

        return args

    args = parser()   
    seed_everything(args.seed)
    
    train_x, train_y, test_x = load_data(args.train, args.test)

    params = []

    lg_1 = {'colsample_bytree': 0.572280100273023, 'learning_rate': 0.010283635038627429, 'max_depth': 180, 'min_child_samples': 135, 'min_split_gain': 0.04511227284338413, 'n_estimators': 900, 'num_leaves': 70, 'reg_alpha': 4.406681827912319, 'reg_lambda': 20.4785600448913, 'scale_pos_weight': 8.302374117433086, 'subsample': 0.1688669888026464}
    lg_2 =  {'colsample_bytree': 0.7641322280477741, 'learning_rate': 0.010977205425053654, 'max_depth': 90, 'min_child_samples': 75, 'min_split_gain': 0.13379952895779884, 'n_estimators': 900, 'num_leaves': 80, 'reg_alpha': 1.9214119194170154, 'reg_lambda': 14.454450236504218, 'scale_pos_weight': 2.171961031806387, 'subsample': 0.9552593593877317}
    lg_3 = {'colsample_bytree': 0.5504769098255781,  'learning_rate': 0.019653385015120244, 'max_depth': 220, 'min_child_samples': 25, 'min_split_gain': 0.1273611040963466, 'n_estimators': 470, 'num_leaves': 160, 'reg_alpha': 3.5549669150756706, 'reg_lambda': 39.88636182674132, 'scale_pos_weight': 12.46696320152359, 'subsample': 0.7590007450921917}
    lg_4 = {'colsample_bytree': 0.5597537952569402, 'learning_rate': 0.02374663979814546, 'max_depth': 32, 'min_child_samples': 100, 'min_split_gain': 0.12211426885216736, 'n_estimators': 1263, 'num_leaves': 200, 'reg_alpha': 14.606693962963451, 'reg_lambda': 299.52278825209424, 'scale_pos_weight': 7.7785016838070735, 'subsample': 0.6254745287838821}
    lg_5 = {'colsample_bytree': 0.4311015575880258, 'learning_rate': 0.01749725932551278, 'max_depth': 53, 'min_child_samples': 15, 'min_split_gain': 0.2820951740673634, 'n_estimators': 974, 'num_leaves': 165, 'reg_alpha': 9.604623064885754, 'reg_lambda': 12.314490508636432, 'scale_pos_weight': 6.6422956907936825, 'subsample': 0.7390190399971659}
    lg_6 = {'colsample_bytree': 0.6889745043181079, 'learning_rate': 0.06146161938790444, 'max_depth': 89, 'min_child_samples': 10, 'min_split_gain': 0.669592868575692, 'n_estimators': 1169, 'num_leaves': 175, 'reg_alpha': 11.405277636150856, 'reg_lambda': 112.37954230084294, 'scale_pos_weight': 5.932435783263877, 'subsample': 0.8265223228903998}  
    lg_7 = {'colsample_bytree': 0.8663251864650988, 'learning_rate': 0.018110306887688978, 'max_depth': 166, 'min_child_samples': 50, 'min_split_gain': 0.025403061552667243, 'n_estimators': 1080, 'num_leaves': 100, 'reg_alpha': 2.0131018839563666, 'reg_lambda': 63.56640846106552, 'scale_pos_weight': 1.8584564419776715, 'subsample': 0.7643028435523616}
    lg_8 = {'colsample_bytree': 0.8970390757241629, 'learning_rate': 0.03571726260659087, 'max_depth': 164, 'min_child_samples': 30, 'min_split_gain': 0.2863362850926679, 'n_estimators': 740, 'num_leaves': 100, 'reg_alpha': 1.1167159754886287, 'reg_lambda': 280.9798636389436, 'scale_pos_weight': 4.75867892931176, 'subsample': 0.681716202670263}
    lg_9 = {'n_estimators': 900, 'max_depth': 86, 'num_leaves': 150, 'min_child_samples': 85, 'colsample_bytree': 0.90507, 'subsample': 0.62362, 'min_split_gain': 0.21034, 'scale_pos_weight': 8.77311, 'reg_alpha': 0.07069, 'reg_lambda': 499.10672, 'learning_rate': 0.04679}
    lg_10 = {'colsample_bytree': 0.8350973419202665, 'learning_rate': 0.03134966396365972, 'max_depth': 114, 'min_child_samples': 20, 'min_split_gain': 0.24406788869557822, 'n_estimators': 454, 'num_leaves': 115, 'reg_alpha': 1.0870546166564243, 'reg_lambda': 346.21163772786895, 'scale_pos_weight': 5.81617865285278, 'subsample': 0.45612075761336973}
    lg_11 = {'colsample_bytree': 0.7285829045071064, 'learning_rate': 0.019839273085108612, 'max_depth': 71, 'min_child_samples': 50, 'min_split_gain': 0.35567737788276876, 'n_estimators': 970, 'num_leaves': 140, 'reg_alpha': 0.27353134227182774, 'reg_lambda': 157.85749037224548, 'scale_pos_weight': 5.956126991298146, 'subsample': 0.7509931500532172}
    lg_12 = {'colsample_bytree': 0.6115826698158419, 'learning_rate': 0.010052927231718068, 'max_depth': 71, 'min_child_samples': 85, 'min_split_gain': 0.12003011548878659, 'n_estimators': 1300, 'num_leaves': 120, 'reg_alpha': 1.3013867029804251, 'reg_lambda': 269.3915696845848, 'scale_pos_weight': 5.290961082236748, 'subsample': 0.7542724715058367}
    lg_13 = {'colsample_bytree': 0.9511047907962863, 'learning_rate': 0.023257873709858216, 'max_depth': 58, 'min_child_samples': 80, 'min_split_gain': 0.21488153574891886, 'n_estimators': 1300, 'num_leaves': 150, 'reg_alpha': 0.33761852089148814, 'reg_lambda': 57.05291849099506, 'scale_pos_weight': 2.0801436555772854, 'subsample': 0.5580106548214563}
    lg_14 = {'colsample_bytree': 0.8851122740930837, 'learning_rate': 0.013136814152245062, 'max_depth': 249, 'min_child_samples': 65, 'min_split_gain': 0.2072264172906347, 'n_estimators': 450, 'num_leaves': 135, 'reg_alpha': 0.642890771203696, 'reg_lambda': 45.624663648443345, 'scale_pos_weight': 6.400746088779947, 'subsample': 0.30084274480143686}

    model = LGBMRegressor(n_jobs=-1, random_state=1, **lg_1)
    model.fit(train_x, train_y['Y_01'])
    pred1 = model.predict(test_x)

    model = LGBMRegressor(n_jobs=-1, random_state=1, **lg_2)
    model.fit(train_x, train_y['Y_02'])
    pred2 = model.predict(test_x)

    model = LGBMRegressor(n_jobs=-1, random_state=1, **lg_3)
    model.fit(train_x, train_y['Y_03'])
    pred3 = model.predict(test_x)

    model = LGBMRegressor(n_jobs=-1, random_state=1, **lg_4)
    model.fit(train_x, train_y['Y_04'])
    pred4 = model.predict(test_x)

    model = LGBMRegressor(n_jobs=-1, random_state=1, **lg_5)
    model.fit(train_x, train_y['Y_05'])
    pred5 = model.predict(test_x)

    model = LGBMRegressor(n_jobs=-1, random_state=1, **lg_6)
    model.fit(train_x, train_y['Y_06'])
    pred6 = model.predict(test_x)


    model = LGBMRegressor(n_jobs=-1, random_state=1, **lg_7)
    model.fit(train_x, train_y['Y_07'])
    pred7 = model.predict(test_x)


    model = LGBMRegressor(n_jobs=-1, random_state=1, **lg_8)
    model.fit(train_x, train_y['Y_08'])
    pred8 = model.predict(test_x)

    model = LGBMRegressor(n_jobs=-1, random_state=1, **lg_9)
    model.fit(train_x, train_y['Y_09'])
    pred9 = model.predict(test_x)

    model = LGBMRegressor(n_jobs=-1, random_state=1, **lg_10)
    model.fit(train_x, train_y['Y_10'])
    pred10 = model.predict(test_x)

    model = LGBMRegressor(n_jobs=-1, random_state=1, **lg_11)
    model.fit(train_x, train_y['Y_11'])
    pred11 = model.predict(test_x)

    model = LGBMRegressor(n_jobs=-1, random_state=1, **lg_12)
    model.fit(train_x, train_y['Y_12'])
    pred12 = model.predict(test_x)

    model = LGBMRegressor(n_jobs=-1, random_state=1, **lg_13)
    model.fit(train_x, train_y['Y_13'])
    pred13 = model.predict(test_x)

    model = LGBMRegressor(n_jobs=-1, random_state=1, **lg_14)
    model.fit(train_x, train_y['Y_14'])
    pred14 = model.predict(test_x)


    sub = pd.read_csv(args.sub)
    sub['Y_01'] = pred1
    sub['Y_02'] = pred2
    sub['Y_03'] = pred3
    sub['Y_04'] = pred4
    sub['Y_05'] = pred5
    sub['Y_06'] = pred6
    sub['Y_07'] = pred7
    sub['Y_08'] = pred8
    sub['Y_09'] = pred9
    sub['Y_10'] = pred10
    sub['Y_11'] = pred11
    sub['Y_12'] = pred12
    sub['Y_13'] = pred13
    sub['Y_14'] = pred14

    sub.to_csv('./best.csv',index=False)