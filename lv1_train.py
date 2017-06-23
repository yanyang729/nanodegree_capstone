import numpy as np
from lv1_class_utilities import Level1Model

def train_lv1(d_model):
    """
    Main function to call in order to train
    all level1 models
    args : d_model (dict) stores all the information (hyperparameter, name etc)
    	of level1 models
    """
    lv1_model = Level1Model(d_model)
    lv1_model.save_oos_pred()
    lv1_model.save_test_preds()



if __name__ == '__main__':
    d_model = {}

    # 0.31573
    d_model['xgb_1'] = {
        'param':{
            'learning_rate': 0.03,
            'max_depth': 5,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'objective': 'reg:linear',
            'n_estimators': 1000,
            'gamma': 0,
            'silent':1,
        },
        'n_folds':3,
        'num_round':1000,
        'flag_choice':{
            'log': False, # if log transform target
            'na_how': 'nothing', # '-99' /'nothing'/'median_dummuy_col'/ 'median_no_dummy'
            # https://www.kaggle.com/c/sberbank-russian-housing-market/discussion/32717
            'merge_macro': False,
            'scaling': False,
            'feat_selection':None, # None /1 ,3,4,5...
            'price_change':False,
            'remove_outlier_lower':True
        }
    }

    # 0.31526
    d_model['xgb_2'] = {
        'param':{
            'learning_rate': 0.03,
            'max_depth': 5,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'objective': 'reg:linear',
            'n_estimators': 1000,
            'gamma': 0,
            'silent':1,
        },
        'n_folds':3,
        'num_round':1000,
        'flag_choice':{
            'log': False,
            'na_how': 'median_no_dummy',
            'merge_macro': False,
            'scaling': False,
            'feat_selection': None,
            'price_change': False,
            'remove_outlier_lower':True

        }
    }

    d_model['xgb_3'] = {
        'param':{
            'learning_rate': 0.05,
            'max_depth': 5,
            'subsample': 0.6,
            'colsample_bytree': 1,
            'objective': 'reg:linear',
            'n_estimators': 1000,
            'gamma': 0,
            'silent':1,
        },
        'n_folds':3,
        'num_round':1000,
        'flag_choice':{
            'log': False,
            'na_how': 'median_no_dummy',
            'merge_macro': False,
            'scaling': False,
            'feat_selection': 50, # use n *4 features
            'price_change': False,
            'remove_outlier_lower': True,
        }
    }

    # extra tree
    d_model['et_1']={
        'n_folds':3,
        'param':{
            'n_estimators':1000,
            'max_features':0.9,
            'max_depth':6,
        },
        'flag_choice': {
            'log': False,
            'na_how': 'median_no_dummy',
            'merge_macro': False,
            'scaling': False,
            'feat_selection': None,
            'price_change': False,
            'remove_outlier_lower': True,
        },
    }


    d_model['rft_1']={
        'n_folds':3,
        'param':{
            'n_estimators':1000,
            'max_features':0.6,
            'max_depth':4,
        },
        'flag_choice': {
            'log': False,
            'na_how': 'median_no_dummy',
            'merge_macro': False,
            'scaling': False,
            'feat_selection': None,
            'remove_outlier_lower': True,
        }

    }


    d_model['nn_1'] ={
        'n_folds':3,
        'epochs':800,
        'num_nodes':(500,100,50),
        'batch_norm':False,
        'dropout':(0.4,0.4,0.4),
        'flag_choice':{
            'log': False,
            'na_how': 'median_no_dummy',
            'merge_macro': False,
            'scaling': True,
            'feat_selection': None,
            'price_change':False,
            'remove_outlier_lower': True,
        }
    }


    train_lv1(d_model)
