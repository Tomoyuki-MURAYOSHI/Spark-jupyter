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
# ### optuna
# - https://qiita.com/studio_haneya/items/2dc3ba9d7cafa36ddffa
# - https://qiita.com/koshian2/items/1c0f781d244a6046b83e
# - https://qiita.com/hideki/items/c09242639fd74abe73a0
# - https://tech.preferred.jp/ja/blog/hyperparameter-tuning-with-optuna-integration-lightgbm-tuner/
# - https://qiita.com/tjmnmn/items/dee7f7e61328e6dd93f7

# In[1]:


from pyspark.sql import SparkSession

spark = SparkSession.builder.enableHiveSupport().getOrCreate()


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

# In[21]:


boston = load_boston()
X, y = boston["data"], boston["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=12)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# In[ ]:





# In[22]:


def objective(trial):
    svr_c = trial.suggest_loguniform('svr_c', 1e0, 1e2)
    epsilon = trial.suggest_loguniform('epsilon', 1e-1, 1e1)
    svr = SVR(C=svr_c, epsilon=epsilon)
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_val)
    return mean_squared_error(y_val, y_pred)


# In[33]:


# verbose
optuna.logging.set_verbosity("WARN")


# In[34]:


get_ipython().run_cell_magic('time', '', '# optuna\nstudy = optuna.create_study()\nstudy.optimize(objective, n_trials=100, n_jobs=-1, )\n\n# 最適解\nprint(study.best_params)\nprint(study.best_value)\nprint(study.best_trial)')


# In[28]:


optuna.__version__


# In[36]:


study.trials_dataframe()


# In[37]:


spark.createDataFrame(study.trials_dataframe()).createOrReplaceTempView("optimize_log")


# In[39]:


spark.table("optimize_log").cache()
df = spark.table("optimize_log")


# In[43]:


study.best_trial


# In[45]:


boston = load_boston()
X, y = boston["data"], boston["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[46]:


svr_opt = SVR(C=47.80312057103909, epsilon=0.5295153347987208)
svr_opt.fit(X_train, y_train)


# In[50]:


y_val = svr_opt.predict(X_train)
r2_score(y_pred=y_val, y_true=y_train)


# In[51]:


y_pred = svr_opt.predict(X_test)
r2_score(y_pred=y_pred, y_true=y_test)


# In[123]:


pd.DataFrame({
    "pred": svr_opt.predict(X_test),
    "true": y_test,
}).plot.scatter("pred", "true")


# In[ ]:





# In[ ]:





# In[52]:


rfr = RandomForestRegressor()


# In[53]:


rfr.fit(X_train, y_train)


# In[54]:


rfr.feature_importances_


# In[55]:


r2_score(y_pred=rfr.predict(X_train), y_true=y_train)


# In[56]:


r2_score(y_pred=rfr.predict(X_test), y_true=y_test)


# In[70]:


reg = RandomForestRegressor()


# In[71]:


cross_val_score(reg, X=X_test, y=y_test, scoring="r2", cv=5)


# In[73]:


cross_val_predict(estimator=reg, X=X_test, y=y_test, cv=5, n_jobs=-1) -y_test


# In[ ]:





# In[ ]:





# In[61]:


print(boston.DESCR)


# In[64]:


df_boston = pd.DataFrame(boston.data, columns=boston.feature_names)
df_boston["target"] = boston.target
df_boston


# In[66]:


spark.createDataFrame(df_boston).write.saveAsTable("sklearn.boston", format="orc", compression="zlib")


# In[67]:


spark.table("sklearn.boston").show()


# In[88]:


xgbr = xgb.XGBRegressor(n_estimators=500, objective="reg:squarederror",)


# In[91]:


xgbr.fit(X_train, y_train, early_stopping_rounds=None, )


# In[93]:


pd.DataFrame({
    "predict": xgbr.predict(X_train),
    "true": y_train,
}).plot.scatter("predict", "true")


# In[94]:


pd.DataFrame({
    "predict": xgbr.predict(X_test),
    "true": y_test,
}).plot.scatter("predict", "true")


# In[95]:


r2_score(y_pred=xgbr.predict(X_test), y_true=y_test)


# In[98]:


xgbr_2 = xgb.XGBRegressor(n_estimators=500, objective="reg:squarederror")
cross_val_score(estimator=xgbr, X=X, y=y, cv=5, n_jobs=-1, scoring="neg_r2")


# In[ ]:





# In[ ]:





# In[99]:


lgbr = lgb.LGBMRegressor(n_estimators=500,)


# In[100]:


lgbr


# In[101]:


cross_val_score(estimator=lgbr, X=X_train, y=y_train, cv=5)


# In[109]:


result = cross_validate(estimator=lgbr, X=X_train, y=y_train, scoring="r2", cv=5, n_jobs=-1, return_estimator=True, return_train_score=True, 
                       fit_params={
                           "early_stopping_rounds": 10,
                           "eval_set": [(X_train, y_train), (X_test, y_test)]
                       })


# In[110]:


result


# In[115]:


result['test_score'].mean()


# In[111]:


xgbr = xgb.XGBRegressor(n_estimators=500)
result2 = cross_validate(estimator=xgbr, X=X_train, y=y_train, scoring="r2", cv=5, n_jobs=-1, return_estimator=True, return_train_score=True, 
                       fit_params={
                           "early_stopping_rounds": 10,
                           "eval_set": [(X_train, y_train), (X_test, y_test)]
                       })
result2


# In[116]:


result2["test_score"].mean()


# In[117]:


xgbr.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_train, y_train), (X_test, y_test)])


# In[118]:


r2_score(y_pred=xgbr.predict(X_test), y_true=y_test)


# In[119]:


r2_score(y_pred=xgbr.predict(X_train), y_true=y_train)


# In[ ]:





# In[120]:


lgbr.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_train, y_train), (X_test, y_test)])


# In[121]:


r2_score(y_pred=lgbr.predict(X_train), y_true=y_train)


# In[122]:


r2_score(y_pred=lgbr.predict(X_test), y_true=y_test)


# In[ ]:





# In[ ]:





# In[125]:


optuna.integration.lightgbm.LGBMRegressor()


# In[145]:


lgb_optuna = optuna.integration.lightgbm_tuner


# In[ ]:





# In[150]:


best_params, tuning_history = dict(), list()

train_dataset=lgb.Dataset(X_train,y_train)
#valid_dataset=lgb.Dataset(X_test,y_test,reference=train_dataset)
valid_dataset=lgb.Dataset(X_test,y_test)

lgb_optuna.train(
    {"objective":"regression", "metric": "l2"}, 
    X_train, 
    y_train, 
    #train_set=train_dataset, 
    valid_sets=[valid_dataset], 
    #num_boost_rounds=300,
    early_stopping_rounds=20,
    verbose_eval=0, 
    best_params=best_params, 
    tuning_history=tuning_history)


# In[143]:


valid_dataset


# In[ ]:




