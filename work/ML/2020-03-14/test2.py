#!/usr/bin/env python
# coding: utf-8

# # Machine-Learning
# 
# - 各手法の学習やまとめ
# 
# ---
# 
# ## Useful Links
# 
# ### RandomForestRegressor
# - https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
# 
# ### optuna
# - https://qiita.com/studio_haneya/items/2dc3ba9d7cafa36ddffa
# - https://qiita.com/koshian2/items/1c0f781d244a6046b83e
# - https://qiita.com/hideki/items/c09242639fd74abe73a0
# - https://tech.preferred.jp/ja/blog/hyperparameter-tuning-with-optuna-integration-lightgbm-tuner/
# - https://qiita.com/tjmnmn/items/dee7f7e61328e6dd93f7

# In[1]:


from pyspark.sql import SparkSession

spark = SparkSession.builder.enableHiveSupport().getOrCreate()
spark


# In[2]:


import numpy as np
import pandas as pd
import dask.dataframe as dd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras


# In[3]:


import xgboost as xgb
import lightgbm as lgb


# In[4]:


from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# In[5]:


import umap
import optuna


# In[6]:


import matplotlib as mpl
import matplotlib.pyplot as plt


# In[ ]:





# - TestData

# In[10]:


#boston = load_boston()
#X, y = boston["data"], boston["target"]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)


# In[9]:


df_boston = spark.table("sklearn.boston").toPandas()
df_boston


# In[17]:


df_train = df_boston.sample(frac=0.8, random_state=15)
df_test = df_boston[~df_boston.index.isin(df_train.index)]

X_train = df_train.drop(columns=["target"]).values
y_train = df_train["target"].values

X_test = df_test.drop(columns=["target"]).values
y_test = df_test["target"].values


# - `RandomForestRegressor`で`optuna`を試してみる
#     - https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
#     - https://towardsdatascience.com/optimizing-hyperparameters-in-random-forest-classification-ec7741f9d3f6

# In[63]:


def objective_rfr(trial):
    # 大体GridSearchやRandomSearchをかけるようなイメージで範囲を指定していく
    
    #n_estimators = trial.suggest_int("n_estimators", 50, 2000)
    n_estimators = trial.suggest_categorical("n_estimators", np.arange(100, 2100, step=100).tolist())
    max_features = trial.suggest_categorical("max_features", ["auto", "sqrt"])
    #max_depth = trial.suggest_categorical("max_depth", np.arange(10, 110, step=10))
    max_depth_ = np.arange(10, 110, step=10).tolist()
    max_depth_.append(None)
    max_depth = trial.suggest_categorical("max_depth", max_depth_)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 4)
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=max_features,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        bootstrap=bootstrap,
    )
    
    #X_opt, X_val, y_opt, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1123)
    #model.fit(X_opt, y_opt)
    
    #score = r2_score(y_pred=model.predict(X_val), y_true=y_val) * -1  # 最小化すべき値を返さないといけない
    score = cross_val_score(estimator=model,
                           X=X_train,
                           y=y_train,
                           cv=3,
                           scoring="r2",
                           n_jobs=-1).mean() * -1 # 最小化すべき値を返すため、-1をかける

    return score
    


# In[ ]:





# In[64]:


# verbose
optuna.logging.set_verbosity("WARN")


# In[65]:


get_ipython().run_cell_magic('time', '', 'study_rfr = optuna.create_study()\nstudy_rfr.optimize(objective_rfr, n_trials=200, n_jobs=-1,)\n\nprint(study_rfr.best_params)\nprint(study_rfr.best_value)\nprint(study_rfr.best_trial)\n\n\ndf_study_rfr = study_rfr.trials_dataframe()\ndisplay(df_study_rfr)\n\n\nmodel_opt = RandomForestRegressor(**study_rfr.best_params)\nscores = cross_val_score(estimator=model_opt, X=X_train, y=y_train, scoring="r2", cv=5, n_jobs=-1)\nprint(f"validation score: {scores.mean(): .3f} +/- {scores.std()}")\n\nmodel_opt.fit(X_train, y_train)\ntest_score = r2_score(\n    y_true=y_test,\n    y_pred=model_opt.predict(X_test)\n)\nprint(f"test score: {test_score : .3f}")')


# - 最後の試行がBestなので、より最適なところが存在すると思われる

# In[66]:


df_feature_importance = pd.DataFrame(model_opt.feature_importances_, index=df_boston.columns[:-1], columns=["feature_importance"]).sort_values("feature_importance", ascending=False)
display(df_feature_importance)
#pd.DataFrame(model_rfr_opt.feature_importances_, index=df_boston.columns[:-1]).plot.barh()
_, ax = plt.subplots(figsize=(12, 9))
df_feature_importance[::-1].plot.barh(ax=ax)
plt.show()


# In[68]:


_, ax = plt.subplots(figsize=(12, 9))
pd.DataFrame({
    "pred": model_opt.predict(X_test),
    "true": y_test
}).plot.scatter("pred", "true", ax=ax)
plt.show()


# In[71]:


_, ax = plt.subplots(figsize=(12, 9))
pd.DataFrame({
    "error": y_test - model_opt.predict(X_test)
}).plot.hist(ax=ax, bins=np.arange(-10, 20))
plt.show()


# In[ ]:





# In[50]:


def objective_rfr2(trial):
    # 大体GridSearchやRandomSearchをかけるようなイメージで範囲を指定していく
    
    #n_estimators = trial.suggest_discrete_uniform("n_estimators", 200, 2000, 200)
    n_estimators = trial.suggest_categorical("n_estimators", np.arange(200, 2200, step=200).tolist())
    max_features = trial.suggest_categorical("max_features", ["auto", "sqrt"])
    #max_depth = trial.suggest_categorical("max_depth", np.arange(10, 110, step=10))
    max_depth_ = np.arange(5, 35, step=5).tolist()
    max_depth_.append(None)
    max_depth = trial.suggest_categorical("max_depth", max_depth_)
    min_samples_split = trial.suggest_categorical("min_samples_split", [2, 5, 10])
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 4)
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=max_features,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        bootstrap=bootstrap,
        n_jobs=-1,
    )
    
    #X_opt, X_val, y_opt, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1123)
    #model.fit(X_opt, y_opt)
    
    #score = r2_score(y_pred=model.predict(X_val), y_true=y_val)
    score = cross_val_score(estimator=model,
                            X=X_train, y=y_train,
                            cv=5,
                            #scoring="r2",
                            scoring='neg_root_mean_squared_error',
                            n_jobs=-1,
                           ).mean() * -1

    return score


# In[51]:


get_ipython().run_cell_magic('time', '', 'study_rfr2 = optuna.create_study()\nstudy_rfr2.optimize(objective_rfr2, n_trials=10, n_jobs=-1,)\n\nprint(study_rfr2.best_params)\nprint(study_rfr2.best_value)\nprint(study_rfr2.best_trial)\n\ndf_study_rfr2 = study_rfr2.trials_dataframe()\ndisplay(df_study_rfr2)')


# In[54]:


get_ipython().run_cell_magic('time', '', 'model_rfr_opt = RandomForestRegressor(**study_rfr2.best_params)\nscores = cross_val_score(estimator=model_rfr_opt, X=X_train, y=y_train, scoring="r2", cv=5, n_jobs=-1)\nprint(f"validation score: {scores.mean(): .3f} +/- {scores.std()}")\n\nmodel_rfr_opt.fit(X_train, y_train)\ntest_score = r2_score(\n    y_true=y_test,\n    y_pred=model_rfr_opt.predict(X_test)\n)\nprint(f"test score: {test_score : .3f}")')


# - 試行回数が10回とかなり少ないので、増やすことでより改善が見込めるはず
# - 無調整でデフォルト設定値のRandomForestRegressorの場合は`0.85`くらいのスコアだったので、ある程度の改善が見られる
# - 一方で、おおまかな傾向を見るだけだったらデフォルト設定値でもあまり困らないのかもしれない

# In[62]:


df_feature_importance = pd.DataFrame(model_rfr_opt.feature_importances_, index=df_boston.columns[:-1], columns=["feature_importance"]).sort_values("feature_importance", ascending=False)
display(df_feature_importance)
#pd.DataFrame(model_rfr_opt.feature_importances_, index=df_boston.columns[:-1]).plot.barh()
_, ax = plt.subplots(figsize=(12, 9))
df_feature_importance[::-1].plot.barh(ax=ax)
plt.show()


# In[46]:


import sklearn
sorted(sklearn.metrics.SCORERS.keys())


# In[ ]:





# In[ ]:





# In[ ]:





# In[28]:


optuna.__version__


# In[61]:


print(boston.DESCR)


# In[ ]:




