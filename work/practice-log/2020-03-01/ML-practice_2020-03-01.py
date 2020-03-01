#!/usr/bin/env python
# coding: utf-8

# - Pipeline

# In[7]:


get_ipython().run_cell_magic('time', '', '\nimport pathlib\nfrom os.path import expanduser, join, abspath\n\nimport pyspark\nfrom pyspark.sql import SparkSession\nfrom pyspark.sql import functions as fn\nfrom pyspark.sql.types import *\nfrom pyspark.sql.window import Window\n\n\nwarehouse_location = abspath(\'/home/jovyan/work/hive-db/spark-warehouse\')\nconf_metastore_db = ("spark.driver.extraJavaOptions", "-Dderby.system.home=/home/jovyan/work/hive-db")\n# https://www.ibm.com/support/knowledgecenter/en/SS3H8V_1.1.0/com.ibm.izoda.v1r1.azka100/topics/azkic_t_updconfigfiles.htm\n\nspark = SparkSession \\\n        .builder \\\n        .config("spark.sql.warehouse.dir", warehouse_location) \\\n        .config(*conf_metastore_db) \\\n        .enableHiveSupport() \\\n        .appName("local-test") \\\n        .getOrCreate()\n\nspark')


# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import dask.array as dd


# In[23]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold


# In[5]:


get_ipython().run_cell_magic('time', '', 'df = pd.read_csv(\n    "https://archive.ics.uci.edu/ml/machine-learning-databases/"\n    "breast-cancer-wisconsin/wdbc.data", header=None)\ndf')


# In[9]:


get_ipython().run_cell_magic('time', '', 'spark.sql("create database if not exists test_data")\nspark.createDataFrame(df).write.saveAsTable("test_data.breast_cancer_0", format=\'orc\', compression="zlib")')


# In[ ]:





# In[11]:


X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)
le.classes_


# In[14]:


le.transform(['M', 'B'])


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=1)


# In[17]:


pipe_lr = make_pipeline(StandardScaler(),
                       PCA(n_components=2),
                       LogisticRegression(random_state=1))


# In[18]:


pipe_lr.fit(X_train, y_train)


# In[19]:


y_pred = pipe_lr.predict(X_test)


# In[20]:


y_pred


# In[21]:


print("Test Accracy: %.3f" % pipe_lr.score(X_test, y_test))


# In[ ]:





# In[25]:


kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True).split(X_train, y_train)


# In[26]:


scores = []


# In[28]:


for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %2d, Class dist.: %s, Acc: %.3f' % 
         (k+1, np.bincount(y_train[train]), score))

print('\nCV accracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


# In[ ]:





# In[ ]:





# In[34]:


dict(pipe_lr.steps)['logisticregression']


# In[ ]:




