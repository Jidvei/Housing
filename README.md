# Housing Competetion

https://www.kaggle.com/c/house-prices-advanced-regression-techniques

Minor script for housing competetion on kaggle.

 - Elementary datapreprocessesing
 - Bayesian optimization of XGBoost and LightGBM hyperparameters
 - XGBoost and LightGBM trained over 5 different seeds, whereafter their predictions on the test-set are averaged for the final submission
 
 Running BayeXGB.py will perform abovementioned steps and save the result in submission.csv that can be directly uploaded to Kaggle.
 
 Ideas for additional work:
  - Further datapreprocessing (Outlier removal, skewness corrections etc.)
  - Automatic feature engineering with Deep Feature Synthesis (https://www.featurelabs.com/wp-content/uploads/2017/12/DSAA_DSM_2015-1.pdf)
  - Additional models (Regularlized linear models, RandomForest, NN etc.)
  - Stacking of models
