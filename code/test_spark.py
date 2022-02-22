from pyspark import SparkContext, SparkConf


def get_sc(app_name="test_spark"):
    conf = (
        SparkConf().setAppName(app_name).setMaster("spark://zhdeMacBook-Pro.local:7077")
    )
    sc = SparkContext(conf=conf)

    sc.addPyFile("./algebra.py")
    sc.addPyFile("./rdd_ntt.py")
    sc.addPyFile("./univariate.py")
    sc.addPyFile("./rdd_merkle.py")
    sc.addPyFile("./rdd_poly.py")
    sc.addPyFile("./rdd_fri.py")

    return sc
