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

# In[114]:


get_ipython().run_cell_magic('time', '', '\neval_results = {}\n\n# pipeline\npl_xgb = Pipeline([\n    ("IMP", SimpleImputer(fill_value=-99999)),\n    ("XGB", xgb.XGBRegressor(#early_stopping_rounds=20,\n                             objective="reg:squarederror",\n                             #eval_metric="rmse",\n                             n_estimators=1000,\n                             callbacks=[xgb.callback.record_evaluation(eval_result=eval_results),],  # うまく働いていない様子?\n                             #verbosity=0,  # silent\n                             #silent=True,\n                             random_state=0))\n])\n\n# GridSearch用パラメータ（仮） 要ブラッシュアップ\nparam_grid = {  # 手法の確認が大事で、実際にサーチするのはとりあえず良いので適当に省く\n    \'XGB__learning_rate\': [0.05, 0.1,], \n    \'XGB__max_depth\': [3, 5, 7, 9], \n    \'XGB__subsample\': [0.8, 0.9, 1.0], \n    \'XGB__colsample_bytree\': [0.7, 0.8, 0.9],\n}\nfit_params = {\n    "XGB__eval_set": [(X_train, y_train), (X_test, y_test)],\n    "XGB__eval_metric": ["rmse","mae"],\n    "XGB__verbose": 0,\n    "XGB__early_stopping_rounds": 20,\n}\n\ngrid_search = GridSearchCV(estimator=pl_xgb,\n                           param_grid=param_grid,\n                           scoring="r2",\n                           cv=3,\n                           verbose=0,\n                          )\n\ngrid_search.fit(X_train, y_train, **fit_params)')


# In[11]:


import sklearn
sklearn.metrics.SCORERS.keys()


# In[115]:


display(grid_search.best_params_)
display(grid_search.best_score_)
display(grid_search.best_estimator_)


# In[116]:


r2_score(y_pred=grid_search.predict(X_test), y_true=y_test)


# In[117]:


learning_data = grid_search.best_estimator_.steps[1][1].evals_result()


# In[118]:


df_learning = pd.DataFrame({
    "train_rmse": learning_data['validation_0']["rmse"],
    "test_rmse": learning_data["validation_1"]["rmse"],
    "train_mae": learning_data["validation_0"]["mae"],
    "test_mae": learning_data["validation_1"]["mae"]
})
display(df_learning[-5:])


# In[119]:


_, ax = plt.subplots(figsize=(12, 9))

df_learning[["train_rmse", "test_rmse"]].plot(ax=ax)
plt.show()


# In[120]:


_, ax = plt.subplots(figsize=(12, 9))

df_learning[["train_mae", "test_mae"]].plot(ax=ax)
plt.show()


# In[121]:


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


# In[102]:


eval_result2 = model_xgb.evals_result()

df_evals2 = pd.DataFrame({
    "train_rmse": eval_result2["validation_0"]["rmse"],
    "train_mae": eval_result2["validation_0"]["mae"],
    "test_rmse": eval_result2["validation_1"]["rmse"],
    "test_mae": eval_result2["validation_1"]["mae"]
})
df_evals2[-5:]


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





# - parameter-tuning x cross-validation

# In[146]:


get_ipython().run_cell_magic('time', '', '\n\neval_results = {}\n\n# pipeline\npl_xgb = Pipeline([\n    ("XGB", xgb.XGBRegressor(objective="reg:squarederror",\n                             n_estimators=1000,\n                             n_jobs=4,\n                             #callbacks=[xgb.callback.record_evaluation(eval_result=eval_results),],  # うまく働いていない様子?\n                             random_state=0))\n])\n\n# GridSearch用パラメータ（仮） 要ブラッシュアップ\nparam_grid = {  # 手法の確認が大事で、実際にサーチするのはとりあえず良いので適当に省く\n    \'XGB__learning_rate\': [0.1,], \n    \'XGB__max_depth\': [3, 5, 7,], \n    \'XGB__subsample\': [0.8, 0.9, 1.0], \n    \'XGB__colsample_bytree\': [0.7, 0.8, 0.9],\n}\nfit_params = {\n    "XGB__eval_set": [(X_train, y_train),], #[(X_train, y_train), (X_test, y_test)],\n    #"XGB__eval_metric": ["rmse","mae"],\n    "XGB__verbose": 0,\n    "XGB__early_stopping_rounds": 10,\n}\n\ngs = GridSearchCV(estimator=pl_xgb,                           \n                  param_grid=param_grid,                           \n                  scoring="r2",                           \n                  cv=3,                        \n                  verbose=0,                          \n                 )\n\n#grid_search.fit(X_train, y_train, **fit_params)\n\nscores = cross_val_score(gs,\n                         X_train,\n                         y_train,\n                         scoring="r2",\n                         cv=5,\n                         fit_params=fit_params\n                        )\n')


# In[147]:


scores


# In[148]:


print(np.mean(scores), np.std(scores))


# - ↑モデルを何にするかの基準に使える

# In[ ]:





# In[ ]:





# - lightgbm

# In[155]:


get_ipython().run_cell_magic('time', '', '\neval_results_lgb = {} # initialize\n\n# pipeline\npl_lgbm = Pipeline([\n    ("IMP", SimpleImputer(fill_value=-99999)),\n    ("LGBM", lgb.LGBMRegressor(objective="regression",      \n                               n_jobs=-1,\n                               n_estimators=1000,                             \n                               random_state=0))\n])\n\n# GridSearch用パラメータ（仮） 要ブラッシュアップ\nparam_grid = {  # 手法の確認が大事で、実際にサーチするのはとりあえず良いので適当に省く\n    \'LGBM__learning_rate\': [0.05, 0.1,], \n    \'LGBM__max_depth\': [3, 5, 7, 9],  # -1 で制限無し\n    \'LGBM__subsample\': [0.8, 0.9, 1.0],   # alias for bagging_fraction\n    \'LGBM__colsample_bytree\': [0.7, 0.8, 0.9],  # feature_fraction\n}\nfit_params = {\n    "LGBM__eval_set": [(X_train, y_train), (X_test, y_test)],\n    "LGBM__eval_metric": ["rmse","mae"],  # <= alias for ["l2", "l1"]\n    "LGBM__verbose": False,\n    "LGBM__early_stopping_rounds": 20,\n    "LGBM__callbacks": [lgb.record_evaluation(eval_results_lgb),],\n}\n\ngs_lgbm = GridSearchCV(estimator=pl_lgbm,                           \n                       param_grid=param_grid,                           \n                       scoring="r2",                           \n                       cv=5,                           \n                       verbose=0,                          \n                      )\n\ngs_lgbm.fit(X_train, y_train, **fit_params)')


# In[156]:


display(gs_lgbm.best_params_)
display(gs_lgbm.best_score_)
display(gs_lgbm.best_estimator_)


# In[157]:


r2_score(y_pred=gs_lgbm.predict(X_test), y_true=y_test)


# In[175]:


eval_results_lgb_best = gs_lgbm.best_estimator_.steps[1][1].evals_result_

df_eval_lgbm = pd.DataFrame({
    "train_rmse": eval_results_lgb_best["valid_0"]["rmse"],
    "test_rmse": eval_results_lgb_best["valid_1"]["rmse"],
    "train_l1": eval_results_lgb_best["valid_0"]["l1"],
    "test_l1": eval_results_lgb_best["valid_1"]["l1"],
    "train_l2": eval_results_lgb_best["valid_0"]["l2"],
    "test_l2": eval_results_lgb_best["valid_1"]["l2"],
})
df_eval_lgbm[-5:]


# In[168]:


eval_results_lgb["valid_0"].keys()


# In[176]:


_, ax = plt.subplots(figsize=(12, 9))

df_eval_lgbm[["train_rmse", "test_rmse"]].plot(ax=ax)
plt.show()


# In[177]:


_, ax = plt.subplots(figsize=(12, 9))

df_eval_lgbm[["train_l1", "test_l1"]].plot(ax=ax)
plt.show()


# In[178]:


_, ax = plt.subplots(figsize=(12, 9))

df_eval_lgbm[["train_l2", "test_l2"]].plot(ax=ax)
plt.show()


# In[ ]:





# In[179]:


get_ipython().run_cell_magic('time', '', '\n\n#eval_results = {}\n\n# pipeline\npl_lgbm_cv = Pipeline([\n    ("IMP", SimpleImputer(fill_value=-99999)),\n    ("LGBM", lgb.LGBMRegressor(objective="regression",      \n                               n_jobs=-1,\n                               n_estimators=1000,                             \n                               random_state=0))\n])\n\n# GridSearch用パラメータ（仮） 要ブラッシュアップ\nparam_grid = {  # 手法の確認が大事で、実際にサーチするのはとりあえず良いので適当に省く\n    \'LGBM__learning_rate\': [0.1,], \n    \'LGBM__max_depth\': [6, 7, 8],  # -1 で制限無し\n    \'LGBM__subsample\': [0.7, 0.8, 0.9],   # alias for bagging_fraction\n    \'LGBM__colsample_bytree\': [0.6, 0.7, 0.8],  # feature_fraction\n}\nfit_params = {\n    "LGBM__eval_set": [(X_train, y_train),], #[(X_train, y_train), (X_test, y_test)],\n    "LGBM__eval_metric": ["rmse",], #["rmse","mae"],  # <= alias for ["l2", "l1"]\n    "LGBM__verbose": False,\n    "LGBM__early_stopping_rounds": 20,\n    #"LGBM__callbacks": [lgb.record_evaluation(eval_results_lgb),],\n}\n\ngs_lgbm_cv = GridSearchCV(estimator=pl_lgbm_cv,                           \n                          param_grid=param_grid,                           \n                          scoring="r2",                           \n                          cv=3,                           \n                          verbose=0,                          \n                         )\n\n#grid_search.fit(X_train, y_train, **fit_params)\n\n\nscores = cross_val_score(gs_lgbm_cv,\n                         X_train,\n                         y_train,\n                         scoring="r2",\n                         cv=5,\n                         fit_params=fit_params\n                        )')


# In[180]:


scores


# In[181]:


print(np.mean(scores), np.std(scores))


# In[ ]:





# In[ ]:





# In[183]:


get_ipython().run_cell_magic('time', '', '\n\n#eval_results = {}\n\n# pipeline\npl_lgbm_cv2 = Pipeline([\n    ("LGBM", lgb.LGBMRegressor(objective="regression",      \n                               n_jobs=-1,\n                               n_estimators=1000,                             \n                               random_state=0))\n])\n\n# GridSearch用パラメータ（仮） 要ブラッシュアップ\nparam_grid = {  # 手法の確認が大事で、実際にサーチするのはとりあえず良いので適当に省く\n    \'LGBM__learning_rate\': [0.1,], \n    \'LGBM__max_depth\': [6, 7, 8],  # -1 で制限無し\n    \'LGBM__subsample\': [0.7, 0.8, 0.9],   # alias for bagging_fraction\n    \'LGBM__colsample_bytree\': [0.6, 0.7, 0.8],  # feature_fraction\n}\nfit_params = {\n    "LGBM__eval_set": [(X_train, y_train),], #[(X_train, y_train), (X_test, y_test)],\n    "LGBM__eval_metric": ["rmse",], #["rmse","mae"],  # <= alias for ["l2", "l1"]\n    "LGBM__verbose": False,\n    "LGBM__early_stopping_rounds": 20,\n    #"LGBM__callbacks": [lgb.record_evaluation(eval_results_lgb),],\n}\n\ngs_lgbm_cv2 = GridSearchCV(estimator=pl_lgbm_cv2,                           \n                          param_grid=param_grid,                           \n                          scoring="r2",                           \n                          cv=2,                           \n                          verbose=0,                          \n                         )\n\n#grid_search.fit(X_train, y_train, **fit_params)\n\n\nscores2 = cross_val_score(gs_lgbm_cv2,\n                         X_train,\n                         y_train,\n                         scoring="r2",\n                         cv=5,\n                         fit_params=fit_params\n                        )')


# In[184]:


scores2


# In[185]:


print(np.mean(scores2), np.std(scores2))


# In[ ]:





# In[ ]:





# - 以降の課題
#     - validation_curve, learning_curveなどの確認
#         - モデルの評価や解釈についての理解深化
#     - `keras`(`tensorflow2`), `randomforest`, `SVR`などの実施と比較
#     - `optuna`を使ったベイズ最適化を使った更なるチューニング方法の確認
#     - スタッキングなどのアンサンブル

# In[ ]:




