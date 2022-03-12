from pyspark import SparkContext, SparkConf


def get_sc(app_name="test_spark"):
    master_url = "spark://zhdeMacBook-Pro.local:7077"
    master_url = "spark://spark:7077"
    conf = SparkConf().setAppName(app_name).setMaster(master_url)
    sc = SparkContext()

    base_path = "/Users/zh/Desktop/stark-anatomy/code"
    base_path = "/opt/bitnami/spark/code"

    sc.addPyFile(base_path + "/algebra.py")
    sc.addPyFile(base_path + "/univariate.py")
    sc.addPyFile(base_path + "/rdd_ntt.py")
    sc.addPyFile(base_path + "/rdd_merkle.py")
    sc.addPyFile(base_path + "/rdd_poly.py")
    sc.addPyFile(base_path + "/rdd_fri.py")

    return sc
