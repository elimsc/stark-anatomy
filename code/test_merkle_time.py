import json
from os import urandom
from ast import literal_eval

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
import sys

conf = (
    SparkConf()
    .set("spark.driver.memory", "4g")
    .set("spark.executor.memory", "4g")
    .set("spark.rpc.message.maxSize", "1024")
    .set("spark.default.parallelism", "8")
)


def test_rdd_merkle(n):
    global data_arr

    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")
    rdd_arr = sc.parallelize(data_arr)
    print(rdd_arr.getNumPartitions())
    print("rdd build tree")
    start = time()
    tree2 = merkle_build(rdd_arr, n)
    print(time() - start)


def test_merkle(n):
    global data_arr
    # data_arr1 = list(data_arr)
    print("build tree")
    start = time()
    tree1 = _merkle_build0(data_arr)
    print(time() - start)


if __name__ == "__main__":
    mode = int(sys.argv[1])
    logn = int(sys.argv[2])
    n = 1 << logn
    print(n)
    # with open("tree.txt", "w+") as f:
    #     for i in range(n):
    #         f.write("(" + str(i) + "," + str(urandom(int(urandom(1)[0]))) + ")\n")
    # data_arr = (
    #     sc.textFile("tree.txt").map(literal_eval).persist(StorageLevel.MEMORY_AND_DISK)
    # )
    data_arr = [(i, urandom(int(urandom(1)[0]))) for i in range(n)]

    if mode == 0:
        test_merkle(n)
    elif mode == 1:
        test_rdd_merkle(n)
    else:
        test_merkle(n)
        test_rdd_merkle(n)
