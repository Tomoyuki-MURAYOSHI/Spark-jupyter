#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('time', '', '\nimport pathlib\nfrom os.path import expanduser, join, abspath\n\nimport pyspark\nfrom pyspark.sql import SparkSession\nfrom pyspark.sql import functions as fn\nfrom pyspark.sql.types import *\nfrom pyspark.sql.window import Window\n\nwarehouse_location = abspath(\'/home/jovyan/work/hive-db/spark-warehouse\')\nconf_jar = ("spark.jars", "scala//target/scala-2.11/sparkudfs_2.11-0.1.jar")\n\nspark = SparkSession \\\n        .builder \\\n        .config("spark.sql.warehouse.dir", warehouse_location) \\\n        .config(*conf_jar) \\\n        .enableHiveSupport() \\\n        .appName("local-test") \\\n        .getOrCreate()\n\nspark')


# - configの`spark.jars`でjarファイルを渡せる
# - https://spark.apache.org/docs/latest/configuration.html

# In[3]:


spark.udf.registerJavaFunction("add_one", "com.example.spark.udfs.addOne","integer")


# In[5]:


get_ipython().run_cell_magic('time', '', "# prepare test-data\nimport seaborn as sns\ndf_flights = spark.createDataFrame(sns.load_dataset('flights'))\ndf_flights.createOrReplaceTempView('flights')\nspark.sql('select * from flights').show()")


# In[6]:


get_ipython().run_cell_magic('time', '', '\n# test scala-udf\n\nspark.sql("""\nselect *, add_one(passengers) as added from flights\n""").show()')


# In[7]:


get_ipython().run_cell_magic('time', '', '\n# test scala-udf\n\nspark.sql("""\nselect *, add_one(cast(passengers as int)) as added from flights\n""").show()')


# - cast(value as int)すると通る

# - 参考文献：
#     - https://qiita.com/neppysan/items/f90156635571a0c327c6

# In[ ]:





# - DataFrame-APIで使える関数の作成
#     - 上記のscalaクラスのjarファイルを使って試みたが失敗している
#     - https://stackoverflow.com/questions/36171208/implement-a-java-udf-and-call-it-from-pyspark
#     - http://ja.voidcc.com/question/p-euycvicn-bq.html

# In[9]:


def my_udf(col): 
    from pyspark.sql.column import Column, _to_java_column, _to_seq 
    pcls = "com.example.spark.udfs.addOne" 
    jc = sc._jvm.java.lang.Thread.currentThread()      .getContextClassLoader().loadClass(pcls).newInstance().getUdf().apply 
    return Column(jc(_to_seq(sc, [col], _to_java_column))) 


# In[10]:


my_udf


# In[11]:


df_flights.show()


# In[12]:


df_flights.dtypes


# In[13]:


df_flights.withColumn("added", my_udf("passengers")).show()


# In[ ]:





# In[17]:


sc = spark.sparkContext
def my_udf(col): 
    from pyspark.sql.column import Column, _to_java_column, _to_seq 
    pcls = "com.example.spark.udfs.addOne" 
    jc = sc._jvm.java.lang.Thread.currentThread()      .getContextClassLoader().loadClass(pcls).newInstance().getUdf().apply 
    return Column(jc(_to_seq(sc, [col], _to_java_column))) 


# In[18]:


my_udf


# In[19]:


df_flights.withColumn("added", my_udf("passengers")).show()


# - DataFrame-APIの中で使えるような関数・集約関数の形にするのは若干手間がかかりそう

# In[ ]:




