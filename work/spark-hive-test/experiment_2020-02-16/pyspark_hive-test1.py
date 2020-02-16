#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('time', '', '\nimport pathlib\nfrom os.path import expanduser, join, abspath\n\nimport pyspark\nfrom pyspark.sql import SparkSession\nfrom pyspark.sql import functions as fn\nfrom pyspark.sql.types import *\nfrom pyspark.sql.window import Window\n\nwarehouse_location = abspath(\'/home/jovyan/work/hive-db/spark-warehouse\')\n#warehouse_location = abspath(\'spark-warehouse\')\n\nspark = SparkSession \\\n        .builder \\\n        .config("spark.sql.warehouse.dir", warehouse_location) \\\n        .enableHiveSupport() \\\n        .appName("local-test") \\\n        .getOrCreate()\n\nspark')


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


get_ipython().run_cell_magic('time', '', '# DataFrame-APIから永続テーブルに保存\ndf_iris.write.saveAsTable("iris")')


# - 下記リンクなどを参考にしているが、hive-site.xmlの設定はうまくいっていない
#     - Notebookを実行しているディレクトリ上に`metastore_db/`と`derby.log`が設定されてしまう（←生成場所を制御したい）
#     - https://software.fujitsu.com/jp/manual/manualfiles/m160007/j2ul2025/02z200/j2025-c-03-00.html

# In[ ]:





# - データベース生成

# In[10]:


spark.sql("""
create database if not exists sns
""")


# - `/path/to/spark-datawarehouse/sns.db/`が生成される

# In[15]:


spark.sql("""
create database if not exists ext
location '/home/jovyan/work/hive-db/spark-warehouse/ext.db'
""")


# In[16]:


spark.sql("""
describe database ext
""").toPandas()


# In[17]:


#spark.sql("drop database ext") 削除


# In[19]:


spark.sql("""
create table if not exists sns.iris
as select * from iris
""")

spark.sql("""
select * from sns.iris
""").show()


# In[21]:


spark.sql("""
describe table sns.iris
""").show()


# In[30]:


get_ipython().run_cell_magic('time', '', '# DataFrame-APIから永続テーブルを保存\nflights = sns.load_dataset(\'flights\')\ndf_flights = spark.createDataFrame(flights)\ndf_flights.write.saveAsTable(\'sns.flights\')\n\nspark.sql("""\nselect * from sns.flights\n""").show()')


# In[28]:


spark.sql("""
show tables
""").show()


# In[27]:


spark.sql("""
show tables from sns
""").show()


# In[39]:


spark.sql("""
create table sns.flights2
-- location '/home/jovyan/work/hive-db/spark-warehouse/sns.db/flights2'
using parquet
partitioned by ( pt1,  pt2 )
as select *, year as pt1, month as pt2 from flights
""")

spark.sql("""
select * from sns.flights2
""")

spark.sql("""
describe table sns.flights2
""")


# - `using parquet`などを使わないとうまく動かない

# In[ ]:




