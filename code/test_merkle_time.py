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

from test_spark import get_sc


def test_rdd_merkle(n):
    global data_arr

    sc = get_sc()
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

    data_arr = [(i, urandom(int(urandom(1)[0]))) for i in range(n)]

    if mode == 0:
        test_merkle(n)
    elif mode == 1:
        test_rdd_merkle(n)
    else:
        test_merkle(n)
        test_rdd_merkle(n)
