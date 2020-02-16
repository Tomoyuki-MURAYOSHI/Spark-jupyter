# spark-hive test 2020-02-16

- `spark-warehouse`の場所指定
- `metastore_db/`および、`derby.log`の場所指定
- https://www.ibm.com/support/knowledgecenter/en/SS3H8V_1.1.0/com.ibm.izoda.v1r1.azka100/topics/azkic_t_updconfigfiles.htm



Template:
```Python
%%time

import pathlib
from os.path import expanduser, join, abspath

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as fn
from pyspark.sql.types import *
from pyspark.sql.window import Window


warehouse_location = abspath('/home/jovyan/work/hive-db/spark-warehouse')
conf_metastore_db = ("spark.driver.extraJavaOptions", "-Dderby.system.home=/home/jovyan/work/hive-db")
# https://www.ibm.com/support/knowledgecenter/en/SS3H8V_1.1.0/com.ibm.izoda.v1r1.azka100/topics/azkic_t_updconfigfiles.htm

spark = SparkSession \
        .builder \
        .config("spark.sql.warehouse.dir", warehouse_location) \
        .config(*conf_metastore_db) \
        .enableHiveSupport() \
        .appName("local-test") \
        .getOrCreate()

spark
```