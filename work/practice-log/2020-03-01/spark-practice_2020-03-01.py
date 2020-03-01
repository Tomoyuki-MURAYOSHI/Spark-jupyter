#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('time', '', '\nimport pathlib\nfrom os.path import expanduser, join, abspath\n\nimport pyspark\nfrom pyspark.sql import SparkSession\nfrom pyspark.sql import functions as fn\nfrom pyspark.sql.types import *\nfrom pyspark.sql.window import Window\n\n\nwarehouse_location = abspath(\'/home/jovyan/work/hive-db/spark-warehouse\')\nconf_metastore_db = ("spark.driver.extraJavaOptions", "-Dderby.system.home=/home/jovyan/work/hive-db")\n# https://www.ibm.com/support/knowledgecenter/en/SS3H8V_1.1.0/com.ibm.izoda.v1r1.azka100/topics/azkic_t_updconfigfiles.htm\n\nspark = SparkSession \\\n        .builder \\\n        .config("spark.sql.warehouse.dir", warehouse_location) \\\n        .config(*conf_metastore_db) \\\n        .enableHiveSupport() \\\n        .appName("local-test") \\\n        .getOrCreate()\n\nspark\n')


# In[3]:


get_ipython().run_cell_magic('time', '', 'spark.sql("show tables from tmp").toPandas()')


# In[4]:


get_ipython().run_cell_magic('time', '', 'spark.sql("describe table tmp.timeseries_test1").show()')


# In[5]:


get_ipython().run_cell_magic('time', '', 'spark.sql("describe table tmp.timeseries_test1").toPandas()')


# In[10]:


df = spark.table("tmp.timeseries_test1")
df.show()


# In[12]:


get_ipython().run_cell_magic('time', '', 'df2 = df.withColumn("date", fn.to_date(fn.col("timestamp")))\ndf2.show()')


# In[17]:


pdf2 = pd.DataFrame(df2.rdd.collect(), columns=df2.columns)
pdf2


# In[20]:


df2.write.saveAsTable("tmp.timeseries_test2", bucketBy="id", sortBy='timestamp')


# In[22]:


spark.table("tmp.timeseries_test2").show()


# In[23]:


spark.table("tmp.timeseries_test2").createTempView("tmp")


# In[24]:


df2.show()


# In[29]:


df2.write.bucketBy(10, col="id").sortBy("timestamp").saveAsTable("tmp.timeseries_test3", compression="zlib", format="orc")


# In[31]:


df3 = spark.table("tmp.timeseries_test3")
df3.show()


# In[32]:


spark.sql("describe table tmp.timeseries_test3").toPandas()


# In[33]:


df3.write.saveAsTable("tmp.timeseries_test4",
                     format="orc", compression="zlib",
                     partitionBy="id", sortBy="timestamp",
                     mode="overwrite")


# In[34]:


df4 = spark.table("tmp.timeseries_test4")
df4.show()


# In[35]:


df3.write.saveAsTable("tmp.timeseries_test5",
                     format="orc", compression="zlib",
                     partitionBy="id", orderBy="timestamp",
                     mode="overwrite")


# In[36]:


df5 = spark.table("tmp.timeseries_test5")
df5.show()


# In[47]:


df2.repartition(10, "id").sortWithinPartitions('timestamp').show()


# In[48]:


df4.show()


# In[54]:


df4.where(fn.rand() <= 4).count()


# In[55]:


df.show()


# In[63]:


df4.withColumn("tmp", fn.count('timestamp').over(Window.partitionBy('id'))).show()


# In[65]:


df4.groupBy('id').count().show()


# In[71]:


df4.groupBy('id').agg(fn.count('timestamp').alias('count')).orderBy(fn.col('id').desc()).show()


# In[72]:


df4.withColumn("count", fn.count('timestamp').over(Window.partitionBy('id')))    .withColumn("flg", (10 / fn.col("count")) =< fn.rand()).show()


# In[80]:


df_tmp = df4.withColumn("count", fn.count('timestamp').over(Window.partitionBy('id')))

df_result = df_tmp.where( (10 / fn.col("count")) >= fn.rand()).drop("count")
df_result.show()
df_result.groupBy('id').count().show()


# In[81]:


df_tmp = df4.withColumn("count", fn.count('timestamp').over(Window.partitionBy('id')))

df_result = df_tmp.where( (5 / fn.col("count")) >= fn.rand()).drop("count")
df_result.show()
df_result.groupBy('id').count().show()


# In[84]:


df_tmp = df4.withColumn("count", fn.count('timestamp').over(Window.partitionBy('id')))

df_result = df_tmp.where( (50 / fn.col("count")) >= fn.rand()).drop("count")
df_result.show()
df_result.groupBy('id').count().sort('id').show()  # sort = orderBy


# In[86]:


df_result[['date']].limit(15).rdd.flatMap(lambda x:x).collect()


# In[87]:


df_result[['value1']].limit(10).rdd.flatMap(lambda x:x).collect()


# In[90]:


df.select(fn.col('timestamp').cast(StringType())).limit(15).rdd.flatMap(lambda x:x).collect()


# In[91]:


pd.DataFrame(df_result.rdd.collect(), columns=df_result.columns)


# - ↑toPandas()ではdatetime.date型に変換出来ずエラーになるが、この方法ならエラー無く変換可能

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Note: 角度付きの矩形領域・楕円領域のIn-Out判定
# 
# ### Parameter
# - 基準中心点の緯度経度(lat_base, lon_base)と回転角:$\theta$、
#     - 矩形の場合、縦横の長さ(a, b)
#     - 楕円の場合、長軸・短軸の半径(a, b)
# 
# の計5つ
# 
# 
# ### Outline
# 1. 判定対象点の緯度経度について、基準中心点の緯度経度との比較から$X$, $Y$(`[m]` or `[km]`)の座標に変換
#     - $X = c1 \times lon$、$Y = c2 \times lat$
#     - $c1, c2$は準定数（はじめに基準点の緯度経度を元にして求める必要があるが、極端に広い領域でなければその後は定数のような扱いで良い）
#         - $c1 := c1(lat\_base, lon\_base), c2 := c2(lat\_base, lon\_base)$
#         - $lat, lon$は判定対象の各点の緯度経度
# 2. 回転角$\theta$にもとづいて座標変換（回転）し、$X^{'}$、$Y^{'}$に変換する
#     - $X^{'} = X\cos(\theta) - Y\sin(\theta)$
#     - $Y^{'} = X\sin(\theta) + Y\cos(\theta)$
# 3. 変換後の座標では単純な条件で判定可能
#     - 矩形
#         - ($|X^{'}| <= a$) && ($|Y^{'}| <= b$)
#     - 楕円
#         - $(\frac{X^{'}}{a})^2 + (\frac{Y^{'}}{b})^2 <= 1$
# 

# In[ ]:




