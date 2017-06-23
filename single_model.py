import pandas as pd
from lv1_preprocessing import prepare_lv1_data
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor, AdaBoostRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import numpy as np
import xgboost as xgb
from datetime import datetime
import logging
logging.basicConfig(filename='log_15.log',level=logging.INFO)
from time import time
logging.info('='*33+'start at {}'.format(datetime.now().strftime('%d-%H-%M'))+'='*33)
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization

def rmlse(y,pred):
    return np.sqrt(np.mean(np.square(np.log(y+1) - np.log(pred + 1))))

# manually get the best params
def make_sub(clf,test,log=True):
    pred = clf.predict(test)
    sub = pd.read_csv('./output/sub_base.csv')
    # log transform
    if log:
        sub.price_doc = np.exp(pred) - 1
    else:
        sub.price_doc = pred
    logging.info('Done at {}'.format(datetime.now().strftime('%d-%H-%M')))
    sub.to_csv('./output/sub_{}.csv'.format(datetime.now().strftime('%d-%H-%M')),index=False)


def xgb_cv(X,y):
    """
    tuning xgb
    :return: best clf
    """
    params = {
        'learning_rate': [0.03],
        'max_depth': [5],
        'subsample': [0.7],
        'colsample_bytree': [0.7],
        'objective': ['reg:linear'],
        'n_estimators': [1000],
        'gamma':[0],
    }

    print('start training...')
    tic = time()
    gs_xgb = GridSearchCV(estimator=xgb.XGBRegressor(seed=123), param_grid=params, cv=3,
                          scoring='neg_mean_squared_error')
    gs_xgb.fit(X, y)

    logging.info('start tuning XGB')
    logging.info('search grid {}'.format(str(params)))
    logging.info('best score on valid set' + str(gs_xgb.best_score_))
    logging.info('best params' + str(gs_xgb.best_params_))
    logging.info('Run time: {}min'.format(str(int((time() - tic)/60))))
    print('done tuning')

    final_model = xgb.XGBRegressor(**gs_xgb.best_params_,seed=123).fit(X,y)
    return final_model


def rft_cv(X,y):
    params={
        'n_estimators':[1000],
        'max_features':[0.5,0.6,0.7],
        'max_depth':[4],
    }

    print('start training...')
    tic = time()
    gs_rft = GridSearchCV(estimator=RandomForestRegressor(random_state=123),param_grid=params,cv=3,
                          scoring='neg_mean_squared_error')
    gs_rft.fit(X,y)

    logging.info('start tuning RFT')
    logging.info('search grid {}'.format(str(params)))
    logging.info('best score on valid set' + str(gs_rft.best_score_))
    logging.info('best params' + str(gs_rft.best_params_))
    logging.info('Run time: {}min'.format(str(int((time() - tic)/60))))
    print('done tuning')

    final_model = RandomForestRegressor(**gs_rft.best_params_,random_state=123,n_jobs=-1).fit(X,y)
    return final_model

def et_cv(X,y):
    params={
        'n_estimators':[1000],
        'max_features':[0.9],
        'max_depth':[6],
    }

    print('start training...')
    tic = time()
    gs_et = GridSearchCV(estimator=ExtraTreesRegressor(random_state=123),param_grid=params,cv=3,
                          scoring='neg_mean_squared_error')
    gs_et.fit(X,y)

    logging.info('start tuning ET')
    logging.info('search grid {}'.format(str(params)))
    logging.info('best score on valid set' + str(gs_et.best_score_))
    logging.info('best params' + str(gs_et.best_params_))
    logging.info('Run time: {}min'.format(str(int((time() - tic)/60))))
    print('done tuning')

    final_model = RandomForestRegressor(**gs_et.best_params_,random_state=123,n_jobs=-1).fit(X,y)
    return final_model

def ada_cv(X,y):
    params={
        'n_estimators':[200],
        'learning_rate':[0.03,0.01,0.05],
    }

    print('start training...')
    tic = time()
    gs_ada = GridSearchCV(estimator=AdaBoostRegressor(random_state=123),param_grid=params,cv=3,
                          scoring='neg_mean_squared_error')
    gs_ada.fit(X,y)

    logging.info('start tuning ada')
    logging.info('search grid {}'.format(str(params)))
    logging.info('best score on valid set' + str(gs_ada.best_score_))
    logging.info('best params' + str(gs_ada.best_params_))
    logging.info('Run time: {}min'.format(str(int((time() - tic)/60))))
    print('done tuning')

    final_model = AdaBoostRegressor(**gs_ada.best_params_,random_state=123).fit(X,y)
    return final_model

def nn(X,y):
    EPOCHS = 400
    BATCH_SIZE = 256
    num_nodes = (500,100,50)
    dropout = (0.4,0.4,0.4)
    batch_norm =False
    tic = time()
    nn = Sequential()
    nn.add(Dense(num_nodes[0], input_shape=(X.shape[1],), activation='relu'))
    if batch_norm:
        nn.add(BatchNormalization())
    nn.add(Dropout(dropout[0]))
    nn.add(Dense(num_nodes[1], activation='relu'))
    if batch_norm:
        nn.add(BatchNormalization())
    nn.add(Dropout(dropout[1]))
    nn.add(Dense(num_nodes[2], activation='relu'))
    nn.add(Dropout(dropout[2]))
    nn.add(Dense(1))
    nn.compile(loss='mean_squared_logarithmic_error', optimizer='adam')
    nn.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1,validation_split=0.33)

    logging.info('start tuning NN')
    logging.info('epochs: {}, batch_size:{}, num_nodes:{},dropour:{},batch_norm:{}'.format(str(EPOCHS),str(BATCH_SIZE),
                                                                    str(num_nodes),str(dropout),str(batch_norm)))
    logging.info('Run time: {}min'.format(str(int((time() - tic)/60))))
    print('done tuning')


    return nn

if __name__ =='__main__':
    LOG_TRANS = False # if log transform target
    NA_HOW = 'median_no_dummy' # '-99' /'nothing'/'median_dummuy_col'/ 'median_no_dummy'
    REMOVE_OUTLIER_LOWER = True # outlier form IQR
    MERGE_MACRO = False # merge 1 col afte PCA
    SCALING = False

    logging.info('LOG_TRANS：{}'.format(str(LOG_TRANS)))
    logging.info('NA_HOW：{}'.format(str(NA_HOW)))
    logging.info('remove lower outlier in training set：{}'.format(str(REMOVE_OUTLIER_LOWER)))
    logging.info('merge macro：{}'.format(str(MERGE_MACRO)))
    logging.info('sacling: %s' % str(SCALING))

    # prepare data that can directly feed to model
    X_train, test,y_train = prepare_lv1_data(
                                      log=LOG_TRANS,na_how=NA_HOW,
        scaling=SCALING)

    final_model = xgb_cv(X_train, y_train)
    # final_model = rft_cv(X_train,y_train)
    # final_model = nn(X_train,y_train)
    # final_model = et_cv(X_train,y_train)
    # final_model = ada_cv(X_train,y_train)
    make_sub(final_model, test,log=LOG_TRANS)