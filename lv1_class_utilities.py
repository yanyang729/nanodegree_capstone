from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
import os
import xgboost as xgb
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor ,ExtraTreesRegressor, AdaBoostRegressor
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from keras.utils import np_utils
from lv1_preprocessing import prepare_lv1_data
from scipy.spatial.distance import cosine
import numpy as np
np.random.seed(123)


class Level1Model():
    """Class to train  1st layer models

    Args:
        d_model = (dict) holds the regressors and their name
        folds = (int) number of CV folds to use
    """
    def __init__(self,d_model):
        self.d_model = d_model
        if not os.path.exists('./models/level1_model_files/train/'):
            os.makedirs('./models/level1_model_files/train/')
        if not os.path.exists('./models/level1_model_files/test/'):
            os.makedirs('./models/level1_model_files/test/')

    def _get_oos_preds(self,X,y,model_name):
        """
        Use cross validation to
        get Out of Sample predictions for the model specified by model_name
        args :
                X  : input data of shape (n_samples, n_features)
                y  : target data of shape (n_samples)
                Id : row identification of shape (n_samples)
                model_name (str) : name of the model to train
        """

        # init output
        y_pred_oos = np.zeros(X.shape[0])

        n_folds = self.d_model[model_name]['n_folds']
        kf = KFold(n_splits=n_folds,shuffle=True)

        valid_results = []
        for icv, (train_indices, oos_indices) in enumerate(kf.split(X)):

            X_train, y_train = X[train_indices], y[train_indices]
            X_oos ,y_valid = X[oos_indices] , y[oos_indices]
            y_train = y_train.reshape(-1,1)


            if "xgb" in model_name:

                param = self.d_model[model_name]["param"]
                num_round = self.d_model[model_name]["num_round"]
                #Train

                dtrain = xgb.DMatrix(X_train,label=y_train)
                dvalid = xgb.DMatrix(X_oos,label=y_valid)
                watchlist=[(dtrain,'train'),(dvalid,'eval')]
                def xg_rmlse(yhat, dtrain):
                    y = dtrain.get_label()
                    return 'RMLSE',np.sqrt(np.mean(np.square(np.log(y + 1) - np.log(yhat + 1))))
                rmlse = lambda y,yhat:np.sqrt(np.mean(np.square(np.log(y + 1) - np.log(yhat + 1))))

                # Construct matrix for test set
                xgmat_oos = xgb.DMatrix(X_oos)
                if self.d_model[model_name]['flag_choice']['log']:
                    bst = xgb.train(param, dtrain, num_round, watchlist,verbose_eval=10,early_stopping_rounds=50)
                    y_pred_oos[oos_indices] = np.exp(bst.predict(xgmat_oos))-1
                    print()
                else:
                    def xg_rmlse(yhat, dtrain):
                        y = dtrain.get_label()
                        return 'RMLSE', np.sqrt(np.mean(np.square(np.log(y + 1) - np.log(yhat + 1))))

                    bst = xgb.train(param, dtrain, num_round, watchlist, feval=xg_rmlse, maximize=False, \
                                    verbose_eval=10, early_stopping_rounds=50)
                    pred = bst.predict(xgmat_oos)
                    y_pred_oos[oos_indices] = pred
                    valid_results.append(rmlse(y_valid,pred))
                    print('score:', str(np.mean(valid_results)), str(np.std(valid_results)))

            # rft
            elif 'rft' in model_name:
                param = self.d_model[model_name]['param']
                rft = RandomForestRegressor(n_jobs=-1,**param)
                rft.fit(X_train,y_train)
                if self.d_model[model_name]['flag_choice']['log']:
                    y_pred_oos[oos_indices] = np.exp(rft.predict(X_oos)) -1
                else:
                    y_pred_oos[oos_indices] = rft.predict(X_oos)

            # et
            elif 'et' in model_name:
                param = self.d_model[model_name]['param']
                et = ExtraTreesRegressor(n_jobs=-1,**param)
                et.fit(X_train,y_train)
                if self.d_model[model_name]['flag_choice']['log']:
                    y_pred_oos[oos_indices] = np.exp(et.predict(X_oos)) -1
                else:
                    y_pred_oos[oos_indices] = et.predict(X_oos)

            # ada
            elif 'ada' in model_name:
                param = self.d_model[model_name]['param']
                ada = AdaBoostRegressor(**param)
                ada.fit(X_train,y_train)
                if self.d_model[model_name]['flag_choice']['log']:
                    y_pred_oos[oos_indices] = np.exp(ada.predict(X_oos)) -1
                else:
                    y_pred_oos[oos_indices] = ada.predict(X_oos)

            elif 'nn' in model_name:
                print('compile and training NN,making oos predict')
                dense1, dense2, dense3 = self.d_model[model_name]['num_nodes']
                drop1, drop2, drop3 = self.d_model[model_name]['dropout']
                batch_norm = self.d_model[model_name]['batch_norm']
                epochs = self.d_model[model_name]['epochs']
                nn = Sequential()
                nn.add(Dense(dense1, input_shape=(X_train.shape[1],), activation='relu'))
                if batch_norm:
                    nn.add(BatchNormalization())
                nn.add(Dropout(drop1))
                nn.add(Dense(dense2, activation='relu'))
                if batch_norm:
                    nn.add(BatchNormalization())
                nn.add(Dropout(drop2))
                nn.add(Dense(dense3, activation='relu'))
                if batch_norm:
                    nn.add(BatchNormalization())
                nn.add(Dropout(drop3))
                nn.add(Dense(1))
                nn.compile(loss='mean_squared_logarithmic_error', optimizer='adam')

                nn.fit(X_train, y_train, nb_epoch=epochs, batch_size=256, verbose=1)
                if self.d_model[model_name]['flag_choice']['log']:
                    y_pred_oos[oos_indices] = np.exp(nn.predict(X_oos, verbose=0)) - 1
                else:
                    y_pred_oos[oos_indices] = nn.predict(X_oos,verbose=0)

        return y_pred_oos, model_name

    def save_oos_pred(self):
        """
        For all the models specified in self.d_model,
        get OOS predictions to get LV1 train features
        Save them to a csv file
        """
        # Loop over estimators
        for model_name in self.d_model.keys():
            # Only train if desired
            flag_choice = self.d_model[model_name]['flag_choice']
            # log = True, na_how = 'nothing', remove_outlier = False, merge_macro = False
            X_train, _, y_train = prepare_lv1_data(**flag_choice)
            print("Compute OOS pred for model: ", model_name)

            y_pred_oos, _ = self._get_oos_preds(X_train,y_train,model_name)
            # +-------+---------------+
            # + index + stacked pred  +
            # +-------+---------------+
            df = pd.DataFrame({model_name:y_pred_oos,'reponse':y_train}).reset_index()
            out_name = "./models/level1_model_files/train/%s_%s_train.csv" % (model_name, str(flag_choice))
            df.to_csv(out_name,index=False)

    def _get_test_preds(self,X_train, y_train, test, model_name):
        """
        Train model specified by model_name on training data
        And apply to test data to get test data level2 features
        args :
                X  : input data of shape (n_samples, n_features) for the train and test samples
                y  : target data of shape (n_samples) only train sample
                Id : row identification of shape (n_samples) for the train and test_sample
                model_name (str) : name of the model to train
        """
        if "xgb" in model_name:
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(test)
            param = self.d_model[model_name]['param']
            num_round = self.d_model[model_name]['num_round']
            bst = xgb.train(param, dtrain, num_round)
            if self.d_model[model_name]['flag_choice']['log']:
                y_pred_test = np.exp(bst.predict(dtest)) - 1
            else:
                y_pred_test = bst.predict(dtest)

        # rft
        elif 'rft' in model_name:
            param = self.d_model[model_name]['param']
            rft = RandomForestRegressor(n_jobs=-1, **param)
            rft.fit(X_train, y_train)
            if self.d_model[model_name]['flag_choice']['log']:
                y_pred_test = np.exp(rft.predict(test)) - 1
            else:
                y_pred_test = rft.predict(test)

        # et
        elif 'et' in model_name:
            param = self.d_model[model_name]['param']
            et = ExtraTreesRegressor(n_jobs=-1, **param)
            et.fit(X_train, y_train)
            if self.d_model[model_name]['flag_choice']['log']:
                y_pred_test = np.exp(et.predict(test)) - 1
            else:
                y_pred_test = et.predict(test)

        # ada
        elif 'ada' in model_name:
            param = self.d_model[model_name]['param']
            ada = AdaBoostRegressor(**param)
            ada.fit(X_train, y_train)
            if self.d_model[model_name]['flag_choice']['log']:
                y_pred_test = np.exp(ada.predict(test)) - 1
            else:
                y_pred_test = ada.predict(test)

        elif 'nn' in model_name:

            print('compile and training NN, making test predict')
            dense1,dense2,dense3 = self.d_model[model_name]['num_nodes']
            drop1,drop2,drop3 = self.d_model[model_name]['dropout']
            batch_norm = self.d_model[model_name]['batch_norm']
            epochs = self.d_model[model_name]['epochs']
            nn = Sequential()
            nn.add(Dense(dense1, input_shape=(X_train.shape[1],), activation='relu'))
            if batch_norm:
                nn.add(BatchNormalization())
            nn.add(Dropout(drop1))
            nn.add(Dense(dense2, activation='relu'))
            if batch_norm:
                nn.add(BatchNormalization())
            nn.add(Dropout(drop2))
            nn.add(Dense(dense3, activation='relu'))
            if batch_norm:
                nn.add(BatchNormalization())
            nn.add(Dropout(drop3))
            nn.add(Dense(1))
            nn.compile(loss='mean_squared_logarithmic_error', optimizer='adam')

            nn.fit(X_train, y_train, nb_epoch=epochs, batch_size=256, verbose=1)
            if self.d_model[model_name]['flag_choice']['log']:
                y_pred_test = np.exp(nn.predict(test, verbose=0)) - 1
            else:
                y_pred_test = nn.predict(test, verbose=0)

        return y_pred_test

    def save_test_preds(self):

        if not os.path.exists('./models/level1_model_files/test/'):
            os.makedirs('./models/level1_model_files/test/')

        for model_name in self.d_model.keys():

            flag_choice = self.d_model[model_name]['flag_choice']
            X_train, test, y_train = prepare_lv1_data(**flag_choice)
            y_pred_test = self._get_test_preds(X_train,y_train,test,model_name)

            out_name = './models/level1_model_files/test/%s_%s_test.csv' % (model_name,str(flag_choice))
            sub = pd.read_csv('./input/sample_submission.csv')
            sub.price_doc = y_pred_test
            sub.to_csv(out_name,index=False)




