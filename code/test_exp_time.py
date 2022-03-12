from os import urandom

from pyspark import SparkConf, SparkContext, StorageLevel
from time import time

from base.algebra import *
from base.ntt import (
    fast_coset_divide,
    fast_coset_evaluate,
    fast_exp,
    fast_multiply,
    intt,
)
from base.univariate import Polynomial
from rdd.rdd_poly import (
    ntt1,
    poly_exp,
    rdd_fast_coset_divide,
    rdd_fast_coset_evaluate,
    rdd_fast_multiply,
    rdd_intt,
    rdd_ntt,
)
import sys

conf = SparkConf().set("spark.driver.memory", "8g").set("spark.executor.memory", "4g")


field = Field.main()
g = field.generator()


def test_fast_exp():
    global arr
    global primitive_root
    global n

    coefficients = [v for (_, v) in arr]
    poly = Polynomial(coefficients)
    print("test_fast_exp")
    start = time()
    values1 = fast_exp(poly, 8, primitive_root, n)
    print("finished. ", time() - start)


def test_rdd_fast_exp():
    global arr
    global primitive_root
    global n

    sc = SparkContext()
    sc.setLogLevel("WARN")

    rdd_arr = sc.parallelize(arr)
    print("test_rdd_fast_exp")
    start = time()
    values1 = poly_exp(rdd_arr, 8, primitive_root, n)
    print("finished. ", time() - start)
    sc.stop()


if __name__ == "__main__":
    mode = int(sys.argv[1])
    logn = int(sys.argv[2])  # 15
    n = 1 << logn
    print(n)
    primitive_root = field.primitive_nth_root(n)
    arr = [(i, field.sample(urandom(17))) for i in range(n)]
    if mode == 0:
        test_fast_exp()
    elif mode == 1:
        test_rdd_fast_exp()
    else:
        test_fast_exp()
        test_rdd_fast_exp()
