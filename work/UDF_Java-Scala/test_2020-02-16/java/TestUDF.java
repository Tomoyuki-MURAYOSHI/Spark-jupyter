package com.test1.test2;

import org.apache.spark.sql.api.java.UDF1;

public class TestClassUdf implements UDF1<Integer, Integer> {

    Integer retval;

    public TestClassUdf(Integer val) {
        retval = val;
    }

    @Override
    public Integer call(Integer arg0) throws Exception {
        return retval;
    }   
}
