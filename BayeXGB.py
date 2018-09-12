import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import lightgbm as lgb
import xgboost as xgb
from bayes_opt import BayesianOptimization

import warnings
warnings.filterwarnings("ignore")

def readdata():
    train = pd.read_csv("train.csv")
    print('Shape of train: {}'.format(train.shape))
    test = pd.read_csv("test.csv")
    print('Shape of test: {}'.format(test.shape))
    return train, test


def preparedata():
    train, test = readdata()
    print("Preparing data....")
    print("Log-transforming target....")
    train["SalePrice"] = np.log1p(train["SalePrice"])
    
    print("Combining datasets...")
    trainrow = train.shape[0]
    testrow = test.shape[0]
    
    train_ID = train['Id']
    test_ID = test['Id']
    train.drop('Id', axis=1, inplace=True)
    test.drop('Id', axis = 1, inplace = True)
    
    print("Saving target...")
    target = train.SalePrice.values
    
    all_data = pd.concat((train,test)).reset_index(drop=True)
    all_data.drop(['SalePrice'], axis=1, inplace=True)
    
    print("Combined datasize is : {}".format(all_data.shape))
    
    print("Filling Categorical NA's...")
    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'SaleType','MiscFeature', 'Alley',
            'BsmtExposure', 'BsmtCond','BsmtFinType2', 'BsmtFinType1', 'MasVnrType','MSZoning', 'PoolQC', 'Fence', 'FireplaceQu'):
        all_data[col] = all_data[col].fillna('Unknown')
        
    print("Filling Numerical NA's...")
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea', 'BsmtHalfBath', 'BsmtFullBath', 'BsmtFinSF1',
           'BsmtFinSF1', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF2'):
            all_data[col] = all_data[col].fillna(0)
    
    print("Imputing with median...")
    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
    
    print("Imputing with mode...")
    all_data["Functional"] = all_data["Functional"].fillna("Typ")
    all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
    all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
    all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
    all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
    all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
    
    print("Dropping features...")
    all_data = all_data.drop(['Utilities'], axis=1)
    
    print("Labelencoding Categorical Features...")
    catcols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
    for c in catcols:
        lbl = LabelEncoder() 
        lbl.fit(list(all_data[c].values)) 
        all_data[c] = lbl.transform(list(all_data[c].values))
        
    print("One-hot Encoding Categorical Variables...")
    all_data = pd.get_dummies(all_data)
        
        
    print('Final shape of dataset: {}'.format(all_data.shape))
    print("Splitting dataset and returning train, test and target...")
    train = all_data[:trainrow] 
    test = all_data[trainrow:]
    
    return train, test, target, test_ID

def BayesXGB():
    print("Performing Bayesian Optimization on XGB...")
    xgb_bo = BayesianOptimization(xgb_evaluate,{'max_depth'       : (3,15),
                                                 'gamma'           : (0,5),
                                                'colsample_bytree' : (0.3, 0.9),
                                                'min_child_weight' : (0,25),
                                                 'subsample'       : (0.5, 1),
                                                 'alpha'           : (0, 5)
                                            })
    xgb_bo.maximize(init_points=1, n_iter=2, acq = 'ei')
    print("Identified optimal hyperparameters...")
    print('Maximum value obtained: {}'.format(xgb_bo.res['max']['max_val']))
    print(xgb_bo.res['max']['max_params'])
    params = (xgb_bo.res['max']['max_params'])
    params['max_depth'] = int(params['max_depth'])
    params['min_child_weight'] = int(params['min_child_weight'])
    params['silent'] = 1
    params['eta'] = 0.01
    return params

def xgb_evaluate(max_depth, gamma, colsample_bytree, min_child_weight, subsample, alpha):
    dtrain =xgb.DMatrix(train, label=target)
    params = {
        'eval_metric' : 'rmse',
        'max_depth'   : int(max_depth),
        'subsample'   : max(min(subsample,1),0) ,
        'eta'         : 0.01 ,
        'gamma'       : max(gamma,0),
        'alpha'       : max(alpha, 0),
        'colsample_bytree' : max(min(colsample_bytree,1),0),
        'min_child_weight' : int(min_child_weight),
        'silent' : 1
    }
    cv_result = xgb.cv(params, dtrain, num_boost_round = 2000, early_stopping_rounds = 100, nfold=5, verbose_eval = False)
    #BayesOptimization kan kun maximere og ikke minimere, derfor skal vi gøre RMSE negativt
    return -1 * cv_result['test-rmse-mean'].iloc[-1]
    
def trainxgb(params):
    print("Training XGBoost with found parameters...")
    n_iters = 5
    xgb_preds = []
    
    for i in range(n_iters):
        X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.20, random_state = i)
    
        dtrain = xgb.DMatrix(X_train, label = y_train)
        dvalid = xgb.DMatrix(X_test, label = y_test)
        testxgb   = xgb.DMatrix(test)
        watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    
        xgb_model = xgb.train(params, dtrain, 5000, watchlist, early_stopping_rounds = 150, verbose_eval = False)
        preds = xgb_model.predict(testxgb)
        preds = np.exp(preds) - 1
        xgb_preds.append(preds)
        
    #predictions = pd.DataFrame(list(zip(np.mean(xgb_preds, axis=0))), columns=['xgbpreds'])
    print("Finished training and predicting...")
    return np.mean(xgb_preds, axis=0)

def lgbBayes():
    print("Performing Bayesian Optimization on LGB...")
    lgb_bo = BayesianOptimization(lgb_evaluate, {'num_leaves' : (3,25),
                                                 'feature_fraction' : (0.1, 0.9),
                                                 'bagging_fraction' : (0.1, 0.9),
                                                 'max_depth'        : (3, 15),
                                                 'lambda_l1'        : (0, 5),
                                                 'lambda_l2'        : (0, 3),
                                                 'min_split_gain'   : (0.001, 0.1),
                                                 'min_child_weight' : (1, 25)    
                                                })
    lgb_bo.maximize(init_points=1, n_iter=2, acq ='ei')
    print("Identified optimal hyperparameters...")
    print("Maximum value obtained: {}".format(lgb_bo.res['max']['max_val']))
    print(lgb_bo.res['max']['max_params'])
    params = (lgb_bo.res['max']['max_params'])
    params['num_leaves'] = int(params['num_leaves']) 
    params['min_child_weight'] = int(params['min_child_weight'])
    params['max_depth'] = int(params['max_depth']) 
    params['metric'] = 'rmse'
    params['learning_rate'] = 0.01
    params['silent'] = 1
    return params

def lgb_evaluate(num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight):
    dtrain =lgb.Dataset(train, label=target)
    params = {'application':'regression_l2','num_iterations':2000, 'learning_rate':0.01, 'early_stopping_round':100, 'metric':'rmse', 'silent':1}
    params["num_leaves"] = int(num_leaves)
    params['feature_fraction'] = max(min(feature_fraction, 1), 0)
    params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
    params['max_depth'] = int(max_depth)
    params['lambda_l1'] = max(lambda_l1, 0)
    params['lambda_l2'] = max(lambda_l2, 0)
    params['min_split_gain'] = min_split_gain
    params['min_child_weight'] = min_child_weight
    cv_result = lgb.cv(params, dtrain, nfold=5, stratified=False, metrics=['rmse'], verbose_eval = False)
    return -1 * cv_result['rmse-mean'][-1]

def trainlgb(params):
    print("Training LightGBM with found parameters...")
    n_iters = 5
    lgb_preds = []
    
    for i in range(n_iters):
        X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.20, random_state = i)
    
        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_test, label=y_test, reference=dtrain)
            
        lgb_model = lgb.train(params, dtrain, 5000, valid_sets=dvalid, early_stopping_rounds = 150, verbose_eval = False)
        preds = lgb_model.predict(test)
        preds = np.exp(preds) - 1
        lgb_preds.append(preds)
        
    #predictions = pd.DataFrame(list(zip(np.mean(lgb_preds, axis=0))), columns=['lgbpreds'])
    print("Finished training and predicting...")
    return np.mean(lgb_preds, axis=0)

def combinepreds(lgb_preds, xgb_preds):
    preds = pd.DataFrame(np.column_stack([lgb_preds, xgb_preds]), 
                               columns=['LGB Preds', 'XGB Preds'])
    preds['mean'] = preds.mean(axis=1)
    return preds

def submission(lgb_preds, xgb_preds):
    print("Creating submission...")
    pred = combinepreds(lgb_preds, xgb_preds)
    submissions = pd.DataFrame()
    submissions['Id'] = test_ID
    submissions['SalePrice'] = pred['mean']
    submissions.to_csv('submission.csv', index=False)
    print("Submission saved to csv...")

if __name__ == '__main__':
    train, test, target, test_ID = preparedata()
    params = BayesXGB()
    xgbpreds = trainxgb(params)
    params = lgbBayes()
    lgbpreds = trainlgb(params)
    submission(lgbpreds, xgbpreds)