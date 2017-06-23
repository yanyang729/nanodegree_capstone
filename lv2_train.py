import pandas as pd
import numpy as np
from glob2 import glob
from lv1_preprocessing import prepare_lv1_data
import xgboost as xgb
from sklearn.ensemble import BaggingRegressor

def prepare_lv2_data(change_price=False):
    """
    read lv1 data and ensemble to make submission
    """
    train_files = glob('./models/level1_model_files/train/*')
    test_files = glob('./models/level1_model_files/test/*')
    num_feat = len(train_files)
    nrow = pd.read_csv(train_files[0]).shape[0]
    X_train = np.zeros((nrow, num_feat))
    num_feat = len(train_files)
    X_test = np.zeros((7662, num_feat))

    for i, path in enumerate(train_files):
        X_train[:, i] = pd.read_csv(path).drop(['index', 'reponse'], axis=1).values.reshape(-1)

    for i, train_path in enumerate(train_files):
        model_name = train_path.split('{')[0].split('/')[-1]

        for test_path in test_files:
            if model_name in test_path:
                print((model_name))
                X_test[:, i] = pd.read_csv(test_path).price_doc.values

    y_train = pd.read_csv(train_files[0]).reponse.values
    # print(pd.DataFrame(X_train).corr(),pd.DataFrame(X_test).corr())
    return X_train, X_test, y_train

###############################################################
X_train, X_test, y_train = prepare_lv2_data(change_price=False)

bg = BaggingRegressor(xgb.XGBRegressor(max_depth=4,learning_rate=0.01,subsample=0.7,colsample_bytree=1,
                                       n_estimators=1000),n_estimators=5,max_samples=0.8)
bg.fit(X_train,y_train)
pred = bg.predict(X_test)

# make submit
sub = pd.read_csv('./input/sample_submission.csv')
sub.price_doc = pred
####################change sub name here######################
sub.to_csv('./sub/6models_drop_first_few_years_pca_counts_with_under_sample.csv',index=False)