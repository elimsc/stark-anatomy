from os import urandom

from pyspark import SparkConf, SparkContext, StorageLevel
from time import time

from algebra import *
from rdd_poly import ntt1, rdd_ntt

conf = SparkConf().set("spark.driver.memory", "8g").set("spark.executor.memory", "4g")

field = Field.main()
n = 1 << 16
primitive_root = field.primitive_nth_root(n)

arr = [(i, field.sample(urandom(17))) for i in range(n)]


def test_ntt():
    coefficients = [v for (_, v) in arr]
    print("test ntt")
    start = time()
    values1 = ntt1(primitive_root, coefficients)
    print("finished. ", time() - start)


def test_rdd_ntt():
    sc = SparkContext(conf=conf)
    rdd_arr = sc.parallelize(arr)
    print("test rdd ntt")
    start = time()
    values1 = rdd_ntt(primitive_root, rdd_arr)
    print("finished. ", time() - start)


test_rdd_ntt()
test_ntt()
