
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train.head()
#Vanvittig mange variable, allerede før der laves dummies.
#PCA er formentlig en god idé her, efter alt feature engineering er færdig
#train.RoofStyle.value_counts()
#test.RoofStyle.value_counts()
#train.info()


# In[4]:

#test["SalePrice"]
#test(['SalePrice'], axis=1, inplace=True)
#obj_df = df.select_dtypes(include=['object']).copy()
#object_clmns = train.dtypes[train.dtypes != "int64"].index
#print(object_clmns)
#print(train.columns)
print(pd.unique(train['MSZoning']))


# In[5]:

#for col in train.columns:
#    if train[col].dtype != "int64":
#        bla = pd.unique(train[col]) 
#        for blatest in bla:     
#            if len(train.loc[train[col] == blatest,col]) < 50:
#                train.loc[train[col] == blatest, col] = "Test"


# In[6]:

#train1 = [train['MSZoning'] == 'RH']
#mask = df['position'] >= 20
#print(train1)


# In[ ]:




# In[ ]:




# In[7]:

#sns.distplot(train['SalePrice'])
#Herre skæv fordeling


# In[4]:

#Log transofmration af SalePrice log(1+x)
train["SalePrice"] = np.log1p(train["SalePrice"])
#sns.distplot(train['SalePrice'])
#Meget bedre, så kan linære modeller bruges bedre. 


# In[5]:

#Samler train og test set, så vi ikke skal lave rettelser 2 gange. Gemmer 
trainrækker = train.shape[0]
testrækker = test.shape[0]

#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

#Fjerner SalePrice så det ikke skal specificeres at den skal ekskluderes i alle operationer/transofmrationer af datasættet
y_train = train.SalePrice.values
#Resetter index så man kan kalde dem
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))


# In[6]:

antal_NA= all_data.isnull().sum()
#Der er temmelig mange NA's
antal_NA.sort_values(ascending=False)[:20]


# In[7]:

#PoolQC er poolquality
#Skyldes det måske at husene bare ikke har en pool og der derfor bliver rapporteret NA?
len(all_data)
#Det er 2909/2919 variable der er NAN
all_data['PoolQC'].describe() #Den er ordinal/kategorisk
pd.unique(all_data['PoolQC']) 
#Går ud fra at Ex er ekstremt godt, Fa er Fantastic og Gd bare er godt. Mærkeligt at der ikke er nogle i dårlig stand?


# In[8]:

all_data["PoolQC"].head()
all_data["PoolQC"] = all_data["PoolQC"].fillna("Unknown")
pd.unique(all_data['PoolQC']) 


# In[9]:

pd.unique(all_data['MiscFeature']) #Nominel feature
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("Unknown")
pd.unique(all_data['MiscFeature']) 


# In[10]:

pd.unique(all_data['Alley']) 
all_data["Alley"] = all_data["Alley"].fillna("Unknown")
pd.unique(all_data['Alley']) 


# In[11]:

pd.unique(all_data['Fence']) 
all_data["Fence"] = all_data["Fence"].fillna("Unknown")
pd.unique(all_data['Fence']) 


# In[12]:

pd.unique(all_data['FireplaceQu'])
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("Unknown")
pd.unique(all_data['FireplaceQu']) 


# In[13]:

#pd.unique(all_data['LotFrontage'])#integers, enten med median eller mean?
#Grupperer dem efter neighborhood
list(all_data.groupby("Neighborhood")["LotFrontage"])
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))


# In[14]:

#Samler kategoriske/nominale features og erstatter i et loop istedet
#GarageQual GarageCond GarageFinish GarageType BsmtQual BsmtExposure BsmtCond BsmtFinType2 BsmtFinType1 MasVnrType MSZoning
pd.unique(all_data['MSSubClass']) 


# In[15]:

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'SaleType',
            'BsmtExposure', 'BsmtCond','BsmtFinType2', 'BsmtFinType1', 'MasVnrType','MSZoning'):
    all_data[col] = all_data[col].fillna('Unknown')
pd.unique(all_data['Exterior2nd'])


# In[16]:

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea', 'BsmtHalfBath', 'BsmtFullBath', 'BsmtFinSF1',
           'BsmtFinSF1', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF2'):
    all_data[col] = all_data[col].fillna(0)


# In[17]:

all_data = all_data.drop(['Utilities'], axis=1)


# In[18]:

all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])


# In[19]:

antal_NA2= all_data.isnull().sum()
#Ingen NA's tilbage, yay!!!
antal_NA2.sort_values(ascending=False)[:5]


# In[20]:

#for col in all_data.columns:
#    if all_data[col].dtype != "int64":
#        bla = pd.unique(all_data[col]) 
#        for blatest in bla:     
#            if len(all_data.loc[all_data[col] == blatest, col]) < 50:
#                all_data.loc[all_data[col] == blatest, col] = "Test"


# In[21]:

from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
#Looper over kolonner og konverterer kategoriske variable
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('Shape all_data: {}'.format(all_data.shape))


# In[1]:

all_data = pd.get_dummies(all_data)
print(all_data.shape)
pd.unique(all_data['FireplaceQu'])


# In[ ]:




# In[26]:

train = all_data[:trainrækker]
test = all_data[trainrækker:]


# In[27]:

train.columns
#test.columns
#Hvorfor helvede har test og train lige mange kolonner??
#For fanden, husk at fjerne SalePrice ordentligt!!!!


# In[29]:

#Modeller der evt. kan testes:
#OLS, Lasso, Ridge, Gradient Boosting, RandomForest, LightGBM, XGBoost, KNN med forskellige antal k'er?, 
#Måske kunne det være ret sejt at stacke det ind i et NN

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


# In[30]:

#Evalueringen er rmsle, så eval metric'en der skal bruges skal være rmse
#Måske er 5 folds alt for meget? Virkelig et lille datasæt trods alt
n_folds = 5
#Bruger 5kfolds til at træne modellen og give rmse tilbage
def rmse_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(kfoldtrain.values)
    rmse= np.sqrt(-cross_val_score(model, kfoldtrain.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

kfoldtrain = train
#Skal først bruges når der skal stackes, parameterne skal findes først. Evt. genbrug randomsearch du lavede til taxa challenge


# In[31]:

train["SalePrice"] = y_train
Train, Test = train_test_split(train, test_size = 0.3)
X_train = Train.drop(['SalePrice'], axis=1)
Y_train = Train["SalePrice"]
X_test = Test.drop(['SalePrice'], axis=1)
Y_test = Test["SalePrice"]

Y_test = Y_test.reset_index().drop('index',axis = 1)
Y_train = Y_train.reset_index().drop('index',axis = 1)


# In[32]:

#### Iterationer
#Træningssæt, validation sæt og test sæt - Lightgbm skal også have deres egne. Alternativt kan vi lade være med at 
#bruge deres specialdesignet matricer men tror ikke det kan undgås til selve tuningen, når at watchlist skal laves?
dtrain = xgb.DMatrix(X_train, label=Y_train)
dvalid = xgb.DMatrix(X_test, label=Y_test)
#dtest = xgb.DMatrix(test)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]


#VI SKAL EKSPERIMENTERE MED BOOST TYPE OGSÅ!!!!!
#gboost has access to three different boosting methods:
#- "gblinear" (Generalized Linear Model) which is using Shotgun (Parallel Stochastic Gradient Descent);
#- "gbtree" (Gradient Boosted Trees) which is the default boosting method using Decision Trees and Stochastic Gradient Descent;
#- "dart" (Dropout Additive Regression Trees) which is a method employing the Dropout method from Neural Networks.
#Defaults to "gbtree".
Iter = 100
best_score_true = 1

xgb_pars = []
for MCW in [21]:
    #for ETA in [0.05, 0.1, 0.15]:
        for CS in [0.5]:
            for MD in [13]:
                for SS in [0.85]:
                    for LAMBDA in [1.7]:
                        xgb_pars.append({'min_child_weight': MCW, 'eta': 0.01, 
                                         'colsample_bytree':CS, 'max_depth': MD,
                                         'subsample': SS, 'lambda': LAMBDA, 
                                         'nthread': -1, 'booster' : 'gbtree', 'eval_metric': 'rmse',
                                         'silent': 1, 'objective': 'reg:linear', 'seed':42})
#i = 0
print(len(xgb_pars))
for i in range(len(xgb_pars)):
#while i < Iter:
   # xgb_par = np.random.choice(xgb_pars, replace = False)
    xgb_par = xgb_pars[i]
    print(xgb_par)
    model = xgb.train(xgb_par, dtrain, 3000, watchlist, early_stopping_rounds=150,
                      maximize=False, verbose_eval=100)
   # i+= 1
    if model.best_score < best_score_true:
        best_score_true = model.best_score
        xgb_par_best = xgb_par

    print('Modeling RMSLE %.5f\n' % model.best_score)


# In[415]:

print('Modeling RMSLE %.5f\n' % best_score_true, xgb_par_best)

#Modeling RMSLE 0.11743
# {'min_child_weight': 20, 'eta': 0.2, 'colsample_bytree': 0.5, 'max_depth': 12, 
#'subsample': 0.9, 'lambda': 2, 'nthread': -1, 'booster': 'gbtree', 'eval_metric': 'rmse',
#'silent': 1, 'objective': 'reg:linear', 'seed': 42}

#Modeling RMSLE 0.09561
# {'min_child_weight': 22, 'eta': 0.2, 'colsample_bytree': 0.5, 'max_depth': 12, 
#  'subsample': 0.85, 'lambda': 1.8, 'nthread': -1, 'booster': 'gbtree', 'eval_metric': 'rmse',
#  'silent': 1, 'objective': 'reg:linear', 'seed': 42}

#Ser godt ud til vi kan teste højere lambda og max depth faktisk!


# In[33]:

testmatrix = xgb.DMatrix(test)


# In[34]:

train["SalePrice"] = y_train
#x2_train['SalePrice']=y_train


# In[ ]:




# In[ ]:

n_iters = 5
xgb_preds = []
for i in range(n_iters): 
    Train, Test = train_test_split(train, test_size = 0.2, random_state = i)
    X_train = Train.drop(['SalePrice'], axis=1)
    Y_train = Train["SalePrice"]
    X_test = Test.drop(['SalePrice'], axis=1)
    Y_test = Test["SalePrice"]

    Y_test = Y_test.reset_index().drop('index',axis = 1)
    Y_train = Y_train.reset_index().drop('index',axis = 1)
    
    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dvalid = xgb.DMatrix(X_test, label=Y_test)
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

    xgb_model = xgb.train(xgb_par_best, dtrain, 20000, watchlist, early_stopping_rounds=150,
                      maximize=False, verbose_eval=500)
    preds = xgb_model.predict(testmatrix)
    preds = np.exp(preds) - 1
    xgb_preds.append(preds)


# In[304]:

np.mean(xgb_preds, axis=0)


# In[305]:

predsfinal =[]


# In[306]:

predsfinal = np.mean(xgb_preds, axis=0)


# In[307]:

predsfinal


# In[308]:

predictions = pd.DataFrame(list(zip(predsfinal)),
              columns=['xgbpreds'])


# In[ ]:




# In[319]:

predictions.head()


# In[ ]:




# In[ ]:




# In[36]:

#### Iterationer
#Træningssæt, validation sæt og test sæt - Lightgbm skal også have deres egne. Alternativt kan vi lade være med at 
#bruge deres specialdesignet matricer men tror ikke det kan undgås til selve tuningen, når at watchlist skal laves?
X2_train = np.array(X_train)
Y2_train = np.array(Y_train)
X2_test  = np.array(X_test)
Y2_test  = np.array(Y_test)
#Skal gøres 1 dimensionel for at lgb spiser dem
y=Y2_train.ravel()
y2 = Y2_test.ravel()
d_train = lgb.Dataset(X2_train, label=y)
d_valid = lgb.Dataset(X2_test, label=y2)
watchlist = [d_valid]

#VI SKAL EKSPERIMENTERE MED BOOST TYPE OGSÅ!!!!!
#gboost has access to three different boosting methods:
#- "gblinear" (Generalized Linear Model) which is using Shotgun (Parallel Stochastic Gradient Descent);
#- "gbtree" (Gradient Boosted Trees) which is the default boosting method using Decision Trees and Stochastic Gradient Descent;
#- "dart" (Dropout Additive Regression Trees) which is a method employing the Dropout method from Neural Networks.
#Defaults to "gbtree".
#Iter = 100
bestlgb_score_true = 1
n_estimators = 1000
#'early_stopping': 10

lgb_pars = []
for MCW in [20]:
    #for ETA in [0.05, 0.1, 0.15]:
        for CS in [0.85]:
            for MD in [16]:
                for SS in [0.85]:
                    for LAMBDA in [2.1]:
                        lgb_pars.append({'learning_rate': 0.01, 'min_sum_hessian_in_leaf': MCW,
                                         'feature_fraction':CS, 'max_depth': MD,
                                         'bagging_fraction': SS, 'lambda_l2': LAMBDA, 
                                         'boosting_type' : 'gbdt', 'metric': 'rmse',
                                          'objective': 'regression', 
                                           'verbose': 0})
#i = 0
print(len(lgb_pars))
for i in range(len(lgb_pars)):
#while i < Iter:
   # xgb_par = np.random.choice(xgb_pars, replace = False)
    lgb_par = lgb_pars[i]
    print(lgb_par)
    lgb_model = lgb.train(lgb_par, d_train, n_estimators, watchlist, verbose_eval = 500, early_stopping_rounds = 50)
    if int(lgb_model.best_score['valid_0']['rmse']) < bestlgb_score_true:
        bestlgb_score_true = lgb_model.best_score['valid_0']['rmse']
        lgb_par_best = lgb_par
        
    #print('Modeling RMSLE %.5f\n' % bestlgb_score_true, lgb_par_best)


# In[311]:

print(lgb_par)


# In[312]:

print('Modeling RMSLE %.5f\n' % bestlgb_score_true, lgb_par_best)


# In[313]:

lgb_pars_final = []
lgb_pars_final.append({'learning_rate': 0.01, 'min_sum_hessian_in_leaf': 20,
                                         'feature_fraction':0.85, 'max_depth': 16,
                                         'bagging_fraction': 0.85, 'lambda_l2': 2.1, 
                                         'boosting_type' : 'gbdt', 'metric': 'rmse',
                                          'objective': 'regression', 'seed':42, 
                                           'verbose': 0})
print(lgb_pars_final)


# In[ ]:




# In[314]:

n_iters = 5
n_estimators = 20000
lgb_preds = []
for i in range(n_iters): 
    Train, Test = train_test_split(train, test_size = 0.2, random_state = i)
    X_train = Train.drop(['SalePrice'], axis=1)
    Y_train = Train["SalePrice"]
    X_test = Test.drop(['SalePrice'], axis=1)
    Y_test = Test["SalePrice"]

    Y_test = Y_test.reset_index().drop('index',axis = 1)
    Y_train = Y_train.reset_index().drop('index',axis = 1)
    
    X2_train = np.array(X_train)
    Y2_train = np.array(Y_train)
    X2_test  = np.array(X_test)
    Y2_test  = np.array(Y_test)
    #Skal gøres 1 dimensionel for at lgb spiser dem
    y=Y2_train.ravel()
    y2 = Y2_test.ravel()
    d_train = lgb.Dataset(X2_train, label=y)
    d_valid = lgb.Dataset(X2_test, label=y2)
    watchlist = [d_valid]

    lgb_model = lgb.train(lgb_par_best , d_train, n_estimators, watchlist, verbose_eval = 500, early_stopping_rounds = 150)

    preds = lgb_model.predict(test)
    preds = np.exp(preds) - 1
    lgb_preds.append(preds)


# In[315]:

np.mean(lgb_preds[1])


# In[316]:

lgb_predsmean = np.mean(lgb_preds, axis=0)


# In[317]:

lgb_predsmean


# In[318]:

predictions['lgbpreds'] = lgb_predsmean


# In[320]:

predictions.head()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[405]:

from sklearn.linear_model import LassoCV, LassoLarsCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.grid_search import GridSearchCV


# In[ ]:




# In[ ]:




# In[406]:

train.drop(['SalePrice'], axis=1, inplace=True)


# In[407]:

model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005], selection='random', max_iter=20000).fit(train, y_train)


# In[408]:

coef = pd.Series(model_lasso.coef_, index = train.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")


# In[409]:

lasso_preds = np.expm1(model_lasso.predict(test))


# In[410]:

from sklearn.feature_selection import SelectFromModel
selection = SelectFromModel(model_lasso, prefit=True)
select_X_train = selection.transform(train)


# In[411]:

lasso_preds


# In[424]:

x2_train = pd.DataFrame(select_X_train)


# In[425]:

x2_train.head()


# In[327]:

predictions['lassopreds'] = lasso_preds


# In[328]:

predictions.head()


# In[358]:

from xgboost import XGBRegressor


# In[351]:

param = {
 'n_estimators':[500,800,1000,2000],
 'max_depth':[3,4,6,8,10],
 'min_child_weight':[4,6,8,10,12],
 'colsample_bytree':[0.2,0.6,0.8],
 'colsample_bylevel':[0.2,0.6,0.8]
}


# In[361]:

gsearch1 = GridSearchCV(estimator = XGBRegressor( 
        objective= 'reg:linear', 
        seed=1), 
    param_grid = param, 
    scoring='neg_mean_absolute_error',
    cv=5,
    verbose = 1)


# In[362]:

gsearch1.fit(train, y_train)


# In[ ]:




# In[347]:

{'min_child_weight': 21, 'eta': 0.01, 'colsample_bytree': 0.5, 
 'max_depth': 13, 'subsample': 0.85, 'lambda': 1.7, 'nthread': -1, 'booster': 'gbtree', 
 'eval_metric': 'rmse', 'silent': 1, 'objective': 'reg:linear', 'seed': 42}


# In[ ]:




# In[ ]:




# In[329]:

from sklearn.linear_model import ElasticNet


# In[330]:

elastic = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)


# In[331]:

elastic_model = elastic.fit(train, y_train)


# In[332]:

elastic_preds = np.expm1(elastic_model.predict(test))


# In[333]:

predictions['elasticpreds'] = elastic_preds


# In[334]:

predictions.head()


# In[ ]:




# In[335]:

from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor()


# In[336]:

knn_params = {"n_neighbors": np.arange(1, 31, 2), 'leaf_size': np.arange(1, 31, 2),
   "metric": ["euclidean", "cityblock"]}


# In[338]:

grid = GridSearchCV(neigh, knn_params)

grid.fit(train, y_train)
grid.best_params_


# In[339]:

knn_model = KNeighborsRegressor(n_neighbors = 11, leaf_size = 3, metric = 'cityblock')


# In[340]:

knn_model.fit(train, y_train)


# In[341]:

knn_preds = np.expm1(knn_model.predict(test))
knn_preds


# In[342]:

predictions['knnpreds'] = knn_preds


# In[343]:

predictions.head()


# In[344]:

finalpredictions = predictions.mean(axis=1)


# In[345]:

testpred = predictions['xgbpreds']*0.18 + predictions['lgbpreds']*0.21 + predictions['lassopreds']*0.3 + predictions['knnpreds']*0.00 + predictions['elasticpreds'] * 0.3


# In[156]:

35/2


# In[346]:

submission2 = pd.DataFrame()
submission2['Id'] = test_ID
submission2["SalePrice"] = testpred
submission2.head()
submission2.to_csv("submission.csv", index=False)


# In[ ]:



