from os import urandom

from pyspark import SparkConf, SparkContext, StorageLevel
from time import time
from rdd.rdd_merkle import (
    Merkle,
    merkle_build,
    _merkle_build0,
    merkle_build_temp,
)
from base.algebra import *
from base.univariate import Polynomial
from rdd.rdd_poly import ntt1, rdd_ntt

conf = SparkConf().set("spark.driver.memory", "8g").set("spark.executor.memory", "4g")


n = 1 << 22
data_arr = [(i, urandom(int(urandom(1)[0]))) for i in range(n)]


def test_rdd_merkle():
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")
    rdd_arr = sc.parallelize(data_arr)

    print("rdd build tree")
    start = time()
    tree2 = merkle_build(rdd_arr, n)
    print(time() - start)
    sc.stop()


def test_merkle():
    print("build tree")
    start = time()
    tree1 = _merkle_build0(data_arr)
    print(time() - start)


test_rdd_merkle()
test_merkle()
