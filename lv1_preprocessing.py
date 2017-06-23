from  sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import xgboost as xgb

def prepare_lv1_data(log=True,na_how='nothing',remove_outlier_upper=False,remove_outlier_lower=True,merge_macro=False,
                     scaling=False,feat_selection=None,drop_strange=True,price_change=False):
    """
    :return: ready for modeling (X_train,X_test,y_train)
    """
    train = pd.read_csv('./input/train.csv', parse_dates=['timestamp'])
    test = pd.read_csv('./input/test.csv', parse_dates=['timestamp'])
    macro = pd.read_csv('./input/macro.csv', parse_dates=['timestamp'])

    # simply drop first few samples because of gap between test&train set
    train.drop([x for x in range(9000)],axis=0,inplace=True)

    ##########################CLEANING###########################################################################
    NROW = train.shape[0]
    X = pd.concat((train,test),axis=0)# all the data

    # full_sq
    X.ix[X[(X.full_sq > 260) & (X.price_doc < 3e7)].index, 'full_sq'] = np.NaN
    X.ix[X[X.full_sq < 10].index, 'full_sq'] = np.NaN
    # life_sq
    X.ix[X[(X.life_sq > 600) | (X.life_sq < 10)].index, 'life_sq'] = np.NaN
    
    bad_index = X[X.life_sq > X.full_sq].index
    X.ix[bad_index, "life_sq"] = np.NaN

    bad_index = X[X['kitch_sq'] > 500].index
    X.ix[bad_index, 'kitch_sq'] = np.NaN

    bad_index = X[X.kitch_sq >= X.life_sq].index
    X.ix[bad_index, 'kitch_sq'] = np.NaN
    bad_index = X[X.kitch_sq >= X.full_sq].index
    X.ix[bad_index, 'kitch_sq'] = np.NaN

    bad_index = X[(X.kitch_sq == 0).values + (X.kitch_sq == 1).values].index
    X.ix[bad_index, "kitch_sq"] = np.NaN

    bad_index = X[(X.num_room == 0) | (X.num_room>=10)].index
    X.ix[bad_index, "num_room"] = np.NaN

    bad_index = X[(X.floor == 0).values * (X.max_floor == 0).values].index
    X.ix[bad_index, ["max_floor", "floor"]] = np.NaN
    bad_index = X[X.floor == 0].index
    X.ix[bad_index, "floor"] = np.NaN
    bad_index = X[X.max_floor == 0].index
    X.ix[bad_index, "max_floor"] = np.NaN
    bad_index = X[X.floor > X.max_floor].index
    X.ix[bad_index, "max_floor"] = np.NaN

    bad_index = X[X.state == 33].index
    X.ix[bad_index, "state"] = np.NaN

    bad_index = X[X.build_year > 2020].index
    X.ix[bad_index, "build_year"] = np.NaN
    bad_index = X[X.build_year < 1500].index
    X.ix[bad_index, "build_year"] = np.NaN


    ###########################CLEANING SEPRATELY########################################################
    # FOR TRAIN
    # drop more rows
    train = X.iloc[:NROW,:]
    train = train[~(train.life_sq > 7000)]
    train = train[train.price_doc / train.full_sq <= 600000]
    train = train[train.price_doc / train.full_sq >= 10000]
    train = train[train.price_doc != 111111112]

    # downsample
    if remove_outlier_lower:
        trainsub = train[train.timestamp < '2015-01-01']
        trainsub = trainsub[trainsub.product_type == "Investment"]

        ind_1m = trainsub[trainsub.price_doc <= 1000000].index
        ind_2m = trainsub[trainsub.price_doc == 2000000].index
        ind_3m = trainsub[trainsub.price_doc == 3000000].index

        train_index = set(train.index.copy())

        for ind, gap in zip([ind_1m, ind_2m, ind_3m], [10, 3, 2]):
            ind_set = set(ind)
            ind_set_cut = ind.difference(set(ind[::gap]))

            train_index = train_index.difference(ind_set_cut)

        train = train.loc[train_index]
    else:
        trainsub = train[train.timestamp < '2015-01-01']
        trainsub = trainsub[trainsub.product_type == "Investment"]
        ind_1m = trainsub[trainsub.price_doc <= 1000000].index
        train_index = set(train.index.copy())
        for ind, gap in zip([ind_1m], [2]):
            ind_set_cut = ind.difference(set(ind[::gap]))
            train_index = train_index.difference(ind_set_cut)
        train = train.loc[train_index]

    bad_index = [23584]
    train.ix[bad_index, "floor"] = np.NaN

    kitch_is_build_year = [13117]
    train.ix[kitch_is_build_year, "build_year"] = train.ix[kitch_is_build_year, "kitch_sq"]
    train.ix[kitch_is_build_year, "kitch_sq"] = np.NaN

    bad_index = train[(train.full_sq > 210) & (train.life_sq / train.full_sq < 0.3)].index
    train.ix[bad_index, "full_sq"] = np.NaN

    bad_index = [10076, 11621, 17764, 19390, 24007, 26713, 29172]
    bad_index = [i for i in bad_index if i in train.index]
    train.ix[bad_index, "num_room"] = np.NaN

    # FOR TEST
    test = X.iloc[NROW:,:].reset_index(drop=True).drop('price_doc',axis=1)

    bad_index = test[(test.full_sq > 150) & (test.life_sq / test.full_sq < 0.3)].index
    test.ix[bad_index, "life_sq"] = np.NaN

    bad_index = [3174, 7313]
    test.ix[bad_index, "num_room"] = np.NaN

    test.product_type.fillna('Investment', inplace=True)

    ##########################FEATURE#############################################################################
    X = pd.concat((train,test),axis=0).reset_index(drop=True)

    # Add month-year
    month_year = (X.timestamp.dt.month + X.timestamp.dt.year * 100)
    month_year_cnt_map = month_year.value_counts().to_dict()
    X['month_year_cnt'] = month_year.map(month_year_cnt_map)

    # Add week-year count
    week_year = (X.timestamp.dt.weekofyear + X.timestamp.dt.year * 100)
    week_year_cnt_map = week_year.value_counts().to_dict()
    X['week_year_cnt'] = week_year.map(week_year_cnt_map)

    # Add month and day-of-week
    X['year'] = X.timestamp.dt.year
    X['month'] = X.timestamp.dt.month

    # other features
    X['rel_floor'] = X['floor'] / X['max_floor'].astype(float)
    X['rel_kitch_sq'] = X['kitch_sq'] / X['full_sq'].astype(float)
    X.apartment_name = X.sub_area + X['metro_km_avto'].astype(str)

    # # # combine count features
    feat_count = [x for x in X.columns if 'count' in x and 'build_count' not in x]
    pca = PCA(n_components=2).fit(X[feat_count])
    X['count_feature_pca1'] = pca.transform(X[feat_count])[:,0]
    X['count_feature_pca2'] = pca.transform(X[feat_count])[:,1]
    X.drop(feat_count,axis=1,inplace=True)


    # adjust data by macro effect
    rate_2016_q2 = 1
    rate_2016_q1 = rate_2016_q2 / .99903
    rate_2015_q4 = rate_2016_q1 / .9831
    rate_2015_q3 = rate_2015_q4 / .9834
    rate_2015_q2 = rate_2015_q3 / .9815
    rate_2015_q1 = rate_2015_q2 / .9932
    rate_2014_q4 = rate_2015_q1 / 1.0112
    rate_2014_q3 = rate_2014_q4 / 1.0169
    rate_2014_q2 = rate_2014_q3 / 1.0086
    rate_2014_q1 = rate_2014_q2 / 1.0126
    rate_2013_q4 = rate_2014_q1 / 0.9902
    rate_2013_q3 = rate_2013_q4 / 1.0041
    rate_2013_q2 = rate_2013_q3 / 1.0044
    rate_2013_q1 = rate_2013_q2 / 1.0104
    rate_2012_q4 = rate_2013_q1 / 0.9832
    rate_2012_q3 = rate_2012_q4 / 1.0277
    rate_2012_q2 = rate_2012_q3 / 1.0279
    rate_2012_q1 = rate_2012_q2 / 1.0279
    rate_2011_q4 = rate_2012_q1 / 1.076
    rate_2011_q3 = rate_2011_q4 / 1.0236
    rate_2011_q2 = rate_2011_q3 / 1
    rate_2011_q1 = rate_2011_q2 / 1.011
    X['average'] = 1
    X_2016_q2_index = X.loc[X['timestamp'].dt.year == 2016].loc[X['timestamp'].dt.month >= 4].loc[
        X['timestamp'].dt.month <= 7].index
    X.loc[X_2016_q2_index, 'average_q_price'] = rate_2016_q2
    X_2016_q1_index = X.loc[X['timestamp'].dt.year == 2016].loc[X['timestamp'].dt.month >= 1].loc[
        X['timestamp'].dt.month < 4].index
    X.loc[X_2016_q1_index, 'average_q_price'] = rate_2016_q1
    X_2015_q4_index = X.loc[X['timestamp'].dt.year == 2015].loc[X['timestamp'].dt.month >= 10].loc[
        X['timestamp'].dt.month < 12].index
    X.loc[X_2015_q4_index, 'average_q_price'] = rate_2015_q4
    X_2015_q3_index = X.loc[X['timestamp'].dt.year == 2015].loc[X['timestamp'].dt.month >= 7].loc[
        X['timestamp'].dt.month < 10].index
    X.loc[X_2015_q3_index, 'average_q_price'] = rate_2015_q3

    X_2015_q2_index = X.loc[X['timestamp'].dt.year == 2015].loc[X['timestamp'].dt.month >= 4].loc[
        X['timestamp'].dt.month < 7].index
    X.loc[X_2015_q2_index, 'average_q_price'] = rate_2015_q2

    X_2015_q1_index = X.loc[X['timestamp'].dt.year == 2015].loc[X['timestamp'].dt.month >= 1].loc[
        X['timestamp'].dt.month < 4].index
    X.loc[X_2015_q1_index, 'average_q_price'] = rate_2015_q1

    # X 2014
    X_2014_q4_index = X.loc[X['timestamp'].dt.year == 2014].loc[X['timestamp'].dt.month >= 10].loc[
        X['timestamp'].dt.month <= 12].index
    X.loc[X_2014_q4_index, 'average_q_price'] = rate_2014_q4

    X_2014_q3_index = X.loc[X['timestamp'].dt.year == 2014].loc[X['timestamp'].dt.month >= 7].loc[
        X['timestamp'].dt.month < 10].index
    X.loc[X_2014_q3_index, 'average_q_price'] = rate_2014_q3

    X_2014_q2_index = X.loc[X['timestamp'].dt.year == 2014].loc[X['timestamp'].dt.month >= 4].loc[
        X['timestamp'].dt.month < 7].index
    X.loc[X_2014_q2_index, 'average_q_price'] = rate_2014_q2

    X_2014_q1_index = X.loc[X['timestamp'].dt.year == 2014].loc[X['timestamp'].dt.month >= 1].loc[
        X['timestamp'].dt.month < 4].index
    X.loc[X_2014_q1_index, 'average_q_price'] = rate_2014_q1

    # X 2013
    X_2013_q4_index = X.loc[X['timestamp'].dt.year == 2013].loc[X['timestamp'].dt.month >= 10].loc[
        X['timestamp'].dt.month <= 12].index
    X.loc[X_2013_q4_index, 'average_q_price'] = rate_2013_q4

    X_2013_q3_index = X.loc[X['timestamp'].dt.year == 2013].loc[X['timestamp'].dt.month >= 7].loc[
        X['timestamp'].dt.month < 10].index
    X.loc[X_2013_q3_index, 'average_q_price'] = rate_2013_q3

    X_2013_q2_index = X.loc[X['timestamp'].dt.year == 2013].loc[X['timestamp'].dt.month >= 4].loc[
        X['timestamp'].dt.month < 7].index
    X.loc[X_2013_q2_index, 'average_q_price'] = rate_2013_q2

    X_2013_q1_index = X.loc[X['timestamp'].dt.year == 2013].loc[X['timestamp'].dt.month >= 1].loc[
        X['timestamp'].dt.month < 4].index
    X.loc[X_2013_q1_index, 'average_q_price'] = rate_2013_q1

    # X 2012
    X_2012_q4_index = X.loc[X['timestamp'].dt.year == 2012].loc[X['timestamp'].dt.month >= 10].loc[
        X['timestamp'].dt.month <= 12].index
    X.loc[X_2012_q4_index, 'average_q_price'] = rate_2012_q4

    X_2012_q3_index = X.loc[X['timestamp'].dt.year == 2012].loc[X['timestamp'].dt.month >= 7].loc[
        X['timestamp'].dt.month < 10].index
    X.loc[X_2012_q3_index, 'average_q_price'] = rate_2012_q3

    X_2012_q2_index = X.loc[X['timestamp'].dt.year == 2012].loc[X['timestamp'].dt.month >= 4].loc[
        X['timestamp'].dt.month < 7].index
    X.loc[X_2012_q2_index, 'average_q_price'] = rate_2012_q2

    X_2012_q1_index = X.loc[X['timestamp'].dt.year == 2012].loc[X['timestamp'].dt.month >= 1].loc[
        X['timestamp'].dt.month < 4].index
    X.loc[X_2012_q1_index, 'average_q_price'] = rate_2012_q1

    # X 2011
    X_2011_q4_index = X.loc[X['timestamp'].dt.year == 2011].loc[X['timestamp'].dt.month >= 10].loc[
        X['timestamp'].dt.month <= 12].index
    X.loc[X_2011_q4_index, 'average_q_price'] = rate_2011_q4

    X_2011_q3_index = X.loc[X['timestamp'].dt.year == 2011].loc[X['timestamp'].dt.month >= 7].loc[
        X['timestamp'].dt.month < 10].index
    X.loc[X_2011_q3_index, 'average_q_price'] = rate_2011_q3

    X_2011_q2_index = X.loc[X['timestamp'].dt.year == 2011].loc[X['timestamp'].dt.month >= 4].loc[
        X['timestamp'].dt.month < 7].index
    X.loc[X_2011_q2_index, 'average_q_price'] = rate_2011_q2

    X_2011_q1_index = X.loc[X['timestamp'].dt.year == 2011].loc[X['timestamp'].dt.month >= 1].loc[
        X['timestamp'].dt.month < 4].index
    X.loc[X_2011_q1_index, 'average_q_price'] = rate_2011_q1

    X['price_doc'] = X['price_doc'] * X['average_q_price']

    # DROP
    null_avg = X.isnull().sum() / train.shape[0]
    super_null_cols = null_avg[null_avg > 0.5].index
    X.drop(super_null_cols, inplace=True, axis=1)

    X.drop(['id', 'timestamp'], axis=1, inplace=True)

    ####################ENCODING#################################################################

    area_median_price = train.groupby('sub_area').median().price_doc.sort_values()
    mapper_sub_area = {area: price / 1000000 for area, price in zip(area_median_price.index, area_median_price.values)}
    X.sub_area = X.sub_area.map(mapper_sub_area)

    X = pd.get_dummies(X, columns=['ecology'])
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = LabelEncoder().fit_transform(list(X[col].values))


    #####################OTHERS#########################################################

    if na_how == 'median_no_dummy':
        # do not create dummy columns to represent nulls
        y_temp = X.price_doc.copy()
        X.drop('price_doc',axis=1,inplace=True)
        X= X.groupby(
            'sub_area').transform(lambda x: x.fillna(x.median()))
        # still have NaNs, so:
        X.fillna(X.median(),inplace=True)
        X = pd.concat((X,y_temp),axis=1)
    else:
        pass

    if scaling:
        for col in X.columns:
            if col not in 'price_doc':
                X[col] = MinMaxScaler().fit_transform(X[col])

    if merge_macro:
        ts = macro.timestamp.values
        macro.child_on_acc_pre_school = macro.child_on_acc_pre_school.map(
            {'45,713': 45713, '7,311': 7311, '16,765': 16765, '#!': np.NaN})
        macro.drop(['modern_education_share', 'old_education_build_share'], axis=1, inplace=True)
        macro.fillna(macro.median(), inplace=True)
        macro.drop('timestamp', axis=1, inplace=True)

        effects = PCA(n_components=1).fit_transform(macro)
        macro_new = pd.DataFrame({'timestamp': ts, 'macro_effect': effects.flatten()})
        X = X.merge(macro_new, on='timestamp', how='left')

    ########################MAGIC Y####################################################################

    X_train = X[X.price_doc > 0]
    y_train = X_train.price_doc
    if price_change:
        y_train = 0.969 * y_train + 10
    X_train.drop('price_doc',axis=1,inplace=True)

    if log:
        y_train = np.log(y_train + 1)

    test = X[~(X.price_doc > 0)].reset_index(drop=True).drop('price_doc',axis=1)

    if feat_selection:
        if isinstance(feat_selection,int):
            gbm = xgb.XGBRegressor(learning_rate=0.03, max_depth=5, subsample=0.7, colsample_bytree=0.7,
                                   objective='reg:linear', n_estimators=300, gamma=0)
            gbm.fit(X_train, y_train)
            feature_importances = sorted(gbm.feature_importances_)[::4]
            fs = SelectFromModel(gbm,threshold=feature_importances[-feat_selection],prefit=True) # chose thres
            X_train =fs.transform(X_train)
            test = fs.transform(test)
            return X_train,test,y_train.values
        if isinstance(feat_selection,str):
            print('pca transform')
            pca = PCA(n_components=200)
            X = pd.concat((X_train,test),axis=0)
            X = pca.fit_transform(X)
            X_train = X[:train.shape[0], :].astype(float)
            test = X[train.shape[0]:, :].astype(float)
            return X_train, test, y_train.values

    return X_train.values,test.values,y_train.values