#!/usr/bin/env python
# coding: utf-8

# # Machine-Learning test
# 
# - `xgboost`, `lightgbm`あたりの手法を`sklearn`のpipeline, gridsearchCVと組み合わせて使えるようにする（目標）
# 
# 
# ---
# 
# ### sklearn
# - https://qiita.com/R0w0/items/3b3d8e660b8abc1f804d
# - https://qiita.com/ishizakiiii/items/0650723cc2b4eef2c1cf
# - https://qiita.com/yhyhyhjp/items/c81f7cea72a44a7bfd3a
# - https://qiita.com/saiaron/items/bb96c0f898cd0cbcd788
# - https://qiita.com/issakuss/items/d30303e200756980ae45
# 
# 
# ### xgboost
# - http://wolfin.hatenablog.com/entry/2018/02/08/092124
# - http://tekenuko.hatenablog.com/entry/2016/09/22/220814
# - https://qiita.com/katsu1110/items/a1c3185fec39e5629bcb
# - https://blog.amedama.jp/entry/2019/01/29/235642
# - https://qiita.com/aaatsushi_bb/items/0b605c0f27493f005c88
# - http://smrmkt.hatenablog.jp/entry/2015/04/28/210039
# - https://qiita.com/msrks/items/e3e958c04a5167575c41
# - https://yag.xyz/blog/2015/08/08/xgboost-python/
# - http://wolfin.hatenablog.com/entry/2018/02/08/092124

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


from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import normalize
from sklearn.impute import SimpleImputer


# In[5]:


import umap


# In[32]:


import matplotlib.pyplot as plt


# In[ ]:





# In[ ]:





# - mpgのテストデータを使ってみる

# In[6]:


df_mpg = spark.read.load("/home/jovyan/work/hive-db/external/tf_testdata/mpg", format="orc")
df_mpg.show()
pdf_mpg = df_mpg.toPandas()


# - make data

# In[7]:


pdf_mpg_oneHot = pd.get_dummies(pdf_mpg, columns=['Origin'], drop_first=False)


# In[8]:


y = pdf_mpg_oneHot.MPG.values
X = pdf_mpg_oneHot.drop(columns=["MPG"]).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)


# In[ ]:





# - imputerを使うのはPipelineのテスト用で実質的な意味は無い

# In[94]:


get_ipython().run_cell_magic('time', '', '\neval_results = {}\n\n# pipeline\npl_xgb = Pipeline([\n    ("IMP", SimpleImputer(fill_value=-99999)),\n    ("XGB", xgb.XGBRegressor(early_stopping_rounds=10,\n                             objective="reg:squarederror",\n                             #eval_metric="rmse",\n                             n_estimators=1000,\n                             callbacks=[xgb.callback.record_evaluation(eval_result=eval_results),],  # うまく働いていない様子?\n                             verbosity=0,  # silent\n                             silent=True,\n                             random_state=0))\n])\n\n# GridSearch用パラメータ（仮） 要ブラッシュアップ\nparam_grid = {  # 手法の確認が大事で、実際にサーチするのはとりあえず良いので適当に省く\n    \'XGB__learning_rate\': [0.3,], \n    \'XGB__max_depth\': [6,], \n    \'XGB__subsample\': [0.8, 0.95], \n    \'XGB__colsample_bytree\': [0.5, 1.0],\n}\nfit_params = {\n    "XGB__eval_set": [(X_train, y_train), (X_test, y_test)],\n    "XGB__eval_metric": ["rmse","mae"],\n    "XGB__verbose": 0,\n    "XGB__early_stopping_rounds": 10,\n}\n\ngrid_search = GridSearchCV(estimator=pl_xgb,\n                           param_grid=param_grid,\n                           scoring="r2",\n                           cv=3,\n                           verbose=0,\n                          )\n\ngrid_search.fit(X_train, y_train, **fit_params)')


# In[11]:


import sklearn
sklearn.metrics.SCORERS.keys()


# In[95]:


display(grid_search.best_params_)
display(grid_search.best_score_)
display(grid_search.best_estimator_)


# In[96]:


r2_score(y_pred=grid_search.predict(X_test), y_true=y_test)


# In[97]:


learning_data = grid_search.best_estimator_.steps[1][1].evals_result()


# In[98]:


df_learning = pd.DataFrame({
    "train_rmse": learning_data['validation_0']["rmse"],
    "test_rmse": learning_data["validation_1"]["rmse"],
    "train_mae": learning_data["validation_0"]["mae"],
    "test_mae": learning_data["validation_1"]["mae"]
})
display(df_learning)


# In[99]:


_, ax = plt.subplots(figsize=(12, 9))

df_learning[["train_rmse", "test_rmse"]].plot(ax=ax)
plt.show()


# In[100]:


_, ax = plt.subplots(figsize=(12, 9))

df_learning[["train_mae", "test_mae"]].plot(ax=ax)
plt.show()


# In[101]:


feature_importtance = grid_search.best_estimator_.steps[1][1].feature_importances_
display(feature_importtance)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[84]:


eval_results2={}
model_xgb = xgb.XGBRegressor(#early_stopping_rounds=10,
                             objective="reg:squarederror",
                             n_estimators=1000,
                             callbacks=[xgb.callback.record_evaluation(eval_result=eval_results2),],
                             random_state=0,
                             #silent=True,
                            )


# In[85]:


model_xgb.fit(
    X_train, y_train, eval_metric=["rmse","mae"],   
    early_stopping_rounds=20,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=0
             )


# In[86]:


eval_result2 = model_xgb.evals_result()

df_evals2 = pd.DataFrame({
    "train_rmse": eval_result2["validation_0"]["rmse"],
    "train_mae": eval_result2["validation_0"]["mae"],
    "test_rmse": eval_result2["validation_1"]["rmse"],
    "test_mae": eval_result2["validation_1"]["mae"]
})
df_evals2


# In[87]:


r2_score(y_pred=model_xgb.predict(X_test), y_true=y_test)


# In[88]:


model_xgb.feature_importances_


# In[89]:


import matplotlib.pyplot as plt
_, ax = plt.subplots(figsize=(12,9))

xgb.plot_importance(model_xgb,
                   ax=ax,
                   importance_type="gain",
                   show_values=True)
plt.show()


# In[90]:


model_xgb.feature_importances_


# - callbacksの使い方がうまく出来ていないが、これでも最低限のことは出来ていそう

# In[91]:


len(df_evals2)


# In[ ]:





# In[ ]:




