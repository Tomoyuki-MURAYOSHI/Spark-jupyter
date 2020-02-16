package com.example.spark.udfs

import org.apache.spark.sql.api.java.UDF1

class addOne extends UDF1[Integer, Integer] {
  def call(x: Integer) = x + 1
} 
