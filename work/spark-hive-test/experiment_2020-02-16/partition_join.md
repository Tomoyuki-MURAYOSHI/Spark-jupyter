# partition & join note

- （調査中）
- partition付きのテーブルを作っておいて、partitionカラムをキーにしてjoinすれば良さそう?
- データ、対象テーブルの内容の性質を鑑みつつ`broadcastjoin`など、joinの方法・アルゴリズムを考える必要があるかも
- `select`時に`cluster by <column-name>`を指定して作ったテーブルやview同士を結合するのも有効かもしれない

---

## useful links:

- https://github.com/vaquarkhan/Apache-Kafka-poc-and-notes/wiki/high-performance-spark---join-and-partition
- https://github.com/vaquarkhan/Apache-Kafka-poc-and-notes/wiki/Apache-Spark-Join-guidelines-and-Performance-tuning
- https://community.cloudera.com/t5/Support-Questions/How-to-reduce-Spark-shuffling-caused-by-join-with-data/td-p/205787
- https://spark.apache.org/docs/latest/sql-performance-tuning.html
- https://docs.oracle.com/cd/F17736_01/12.2.1.3/odibd/working-spark.html#GUID-12197C8B-8528-40DD-87A8-979A03C70102
- https://stackoverflow.com/questions/48160627/partition-data-for-efficient-joining-for-spark-dataframe-dataset
- https://qiita.com/neppysan/items/aa4be9b9a07c3c54f2fb

