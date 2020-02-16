#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('time', '', '\nimport pathlib\nfrom os.path import expanduser, join, abspath\n\nimport pyspark\nfrom pyspark.sql import SparkSession\nfrom pyspark.sql import functions as fn\nfrom pyspark.sql.types import *\nfrom pyspark.sql.window import Window\n\n\nwarehouse_location = abspath(\'/home/jovyan/work/hive-db/spark-warehouse\')\nconf_metastore_db = ("spark.driver.extraJavaOptions", "-Dderby.system.home=/home/jovyan/work/hive-db")\n# https://www.ibm.com/support/knowledgecenter/en/SS3H8V_1.1.0/com.ibm.izoda.v1r1.azka100/topics/azkic_t_updconfigfiles.htm\n\nspark = SparkSession \\\n        .builder \\\n        .config("spark.sql.warehouse.dir", warehouse_location) \\\n        .config(*conf_metastore_db) \\\n        .enableHiveSupport() \\\n        .appName("local-test") \\\n        .getOrCreate()\n\nspark')


# In[2]:


import seaborn as sns
import pandas as pd
import numpy as np
import dask.dataframe as dd


# In[ ]:





# - テストデータ用意

# In[3]:


iris = sns.load_dataset('iris')
df_iris = spark.createDataFrame(iris)
df_iris.show()


# In[4]:


# 接続テスト
spark.sql("""
create database if not exists tmp
""")

spark.sql("""
show databases
""").show()


# In[5]:


get_ipython().run_cell_magic('time', '', '# DataFrame-APIから永続テーブルに保存\ndf_iris.write.saveAsTable("tmp.iris")\n\nspark.sql("""\nselect * from tmp.iris\n""").show()')


# - ちゃんと指定した場所にHiveのDBとmetastore_db(derby)両方が設定されていることを確認
#     - https://www.ibm.com/support/knowledgecenter/en/SS3H8V_1.1.0/com.ibm.izoda.v1r1.azka100/topics/azkic_t_updconfigfiles.htm

# - データベース生成

# In[6]:


spark.sql("""
create database if not exists sns
""")


# - `/path/to/spark-datawarehouse/sns.db/`が生成される

# In[7]:


spark.sql("""
create database if not exists ext
location '/home/jovyan/work/hive-db/spark-warehouse/ext.db'
""")


# In[8]:


spark.sql("""
describe database ext
""").toPandas()


# In[9]:


spark.sql("""
show databases
""").show()


# In[17]:


#spark.sql("drop database ext") 削除


# In[13]:


spark.sql("""
create table if not exists sns.iris_3
using orc
options ("compression"="zlib")
as select * from tmp.iris
""")

spark.sql("""
select * from sns.iris_3
""").show()


# In[15]:


spark.sql("""
describe table sns.iris_3
""").show()


# In[ ]:





# In[ ]:





# In[ ]:




