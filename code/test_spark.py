from pyspark import SparkContext, SparkConf


def get_sc(app_name="test_spark"):
    master_url = "spark://spark:7077"
    conf = (
        SparkConf()
        .setAppName(app_name)
        .setMaster(master_url)
        .set("spark.driver.memory", "4g")
        .set("spark.executor.memory", "4g")
    )
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    base_path = "/opt/bitnami/spark/code"

    sc.addPyFile(base_path + "/base.zip")
    sc.addPyFile(base_path + "/rdd.zip")
    sc.addPyFile(base_path + "/rescue.zip")

    return sc
