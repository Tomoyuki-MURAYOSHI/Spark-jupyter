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

# In[4]:


from pyspark.sql import SparkSession

spark = SparkSession.builder.enableHiveSupport().getOrCreate()


# In[1]:


import numpy as np
import dask.dataframe as dd
import seaborn as sns
from tensorflow import keras


# In[11]:


import xgboost as xgb
import lightgbm as lgb


# In[123]:


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


# In[63]:


from sklearn.impute import SimpleImputer


# In[16]:


import umap


# In[ ]:





# - mpgのテストデータを使ってみる

# In[2]:


dataset_path = keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
dataset_path


# In[3]:


column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin'] 
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()


# In[9]:


df_mpg = spark.createDataFrame(dataset.rename(columns={"Model Year":"Model_Year"}))
df_mpg.show()


# In[10]:


get_ipython().run_cell_magic('time', '', 'df_mpg.write.saveAsTable("mpg", format="orc", compression="zlib", path="/home/jovyan/work/hive-db/external/tf_testdata/mpg")')


# In[29]:


pdf_mpg = df_mpg.toPandas()


# - make data

# In[52]:


pdf_mpg_oneHot = pd.get_dummies(pdf_mpg, columns=['Origin'], drop_first=True)


# In[56]:


y = pdf_mpg_oneHot.MPG.values
X = pdf_mpg_oneHot.drop(columns=["MPG"]).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)


# In[ ]:





# In[58]:


xgb_reg = xgb.XGBRegressor()


# In[60]:





# In[61]:


from sklearn.preprocessing import normalize


# In[64]:


imp = SimpleImputer(fill_value=-1)


# In[65]:


imp.fit_transform(X)


# - imputerを使うのはPipelineのテスト用で実質的な意味は無い

# In[86]:


eval_results = {}

pl_xgb = Pipeline([
    ("IMP", SimpleImputer(fill_value=-99999)),
    ("XGB", xgb.XGBRegressor(early_stopping_rounds=20,
                             objective="reg:squarederror",
                             eval_metric="rmse",
                             callbacks=[xgb.callback.record_evaluation(eval_result=eval_results),],
                             random_state=0))
])


# In[ ]:


params_grid = {'max_depth': [3, 5, 10], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5, 10, 100], 'subsample': [0.8, 0.85, 0.9, 0.95], 'colsample_bytree': [0.5, 1.0]}


# In[ ]:


grid_search = GridSearchCV


# In[74]:


xgb.XGBRegressor(num_boost_round=1000, early_stopping_rounds=10)


# In[75]:


xgb.callback.record_evaluation


# In[87]:


pl_xgb.fit(X_train, y_train)


# In[88]:


eval_results


# In[85]:


pl_xgb.steps[1][1].evals_result()


# In[101]:


eval_results={}
model_xgb = xgb.XGBRegressor(early_stopping_rounds=20,
                             objective="reg:squarederror",
                             #eval_metric="rmse",
                             #eval_set=[(X_train, y_train), (X_test, y_test)],
                             callbacks=[xgb.callback.record_evaluation(eval_result=eval_results),],
                             random_state=0)


# In[112]:


model_xgb.fit(X_train, y_train, eval_metric=["rmse","mae"],
                             eval_set=[(X_train, y_train), (X_test, y_test)],
             )


# In[106]:


eval_results


# In[113]:


a = model_xgb.evals_result()
a


# In[125]:


r2_score(y_pred=model_xgb.predict(X_test), y_true=y_test)


# In[126]:


model_xgb.feature_importances_


# In[127]:


import matplotlib.pyplot as plt
_, ax = plt.subplots()

xgb.plot_importance(model_xgb,
                   ax=ax,
                   importance_type="gain",
                   show_values=True)
plt.show()


# - 使い方はともかく、こんな感じでプロット出来る（↑は見たい項目では無い気がするが。。。）

# In[ ]:





# - callbacksの使い方がうまく出来ていないが、これでも最低限のことは出来ていそう

# In[115]:


pd.DataFrame(a)


# In[116]:


pd.DataFrame(a).to_parquet("tmp_fit_validation.snappy.parquet")


# In[120]:


spark.read.load("tmp_fit_*parquet").show()
spark.read.load("tmp_fit_*parquet").printSchema()
spark.read.load("tmp_fit_*parquet").dtypes


# In[ ]:





# In[ ]:





# In[128]:


pdf_mpg_oneHot


# In[ ]:




