from os import urandom

from pyspark import SparkConf, SparkContext, StorageLevel
from time import time

from base.algebra import *
from base.ntt import fast_coset_divide, fast_coset_evaluate, fast_multiply
from base.univariate import Polynomial
from rdd.rdd_poly import (
    ntt1,
    poly_combine,
    poly_scale,
    rdd_fast_coset_divide,
    rdd_fast_coset_evaluate,
    rdd_fast_multiply,
    rdd_ntt,
)

conf = SparkConf().set("spark.driver.memory", "8g").set("spark.executor.memory", "4g")


field = Field.main()
n = 1 << 18
primitive_root = field.primitive_nth_root(n)
g = field.generator()

arr = [(i, field.sample(urandom(17))) for i in range(n)]


def test_ntt():
    coefficients = [v for (_, v) in arr]
    print("test ntt")
    start = time()
    values1 = ntt1(primitive_root, coefficients)
    print("finished. ", time() - start)


def test_rdd_ntt():
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")
    rdd_arr = sc.parallelize(arr)
    print("test rdd ntt")
    start = time()
    values1 = rdd_ntt(primitive_root, n, rdd_arr)
    print("finished. ", time() - start)
    sc.stop()


def test_fast_coset_evaluate():
    coefficients = [v for (_, v) in arr]
    poly = Polynomial(coefficients)
    print("test_fast_coset_evaluate")
    start = time()
    values = fast_coset_evaluate(poly, g, primitive_root, n)
    print("finished. ", time() - start)


def test_rdd_fast_coset_evaluate():
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    rdd_arr = sc.parallelize(arr)
    print("test_rdd_fast_coset_evaluate")
    start = time()
    values1 = rdd_fast_coset_evaluate(rdd_arr, g, primitive_root, n)
    print("finished. ", time() - start)
    sc.stop()


def test_fast_multiply():
    coefficients = [v for (_, v) in arr]
    poly = Polynomial(coefficients)
    print("test_fast_multiply")
    start = time()
    values1 = fast_multiply(poly, poly, primitive_root, n)
    print("finished. ", time() - start)


def test_rdd_fast_multiply():
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    rdd_arr = sc.parallelize(arr)
    print("test_rdd_fast_multiply")
    start = time()
    values1 = rdd_fast_multiply(rdd_arr, rdd_arr, primitive_root, n)
    print("finished. ", time() - start)
    sc.stop()


def test_fast_coset_divide():
    coefficients = [v for (_, v) in arr]
    poly = Polynomial(coefficients)
    print("test_fast_coset_divide")
    start = time()
    values1 = fast_coset_divide(poly, poly, g, primitive_root, n)
    print("finished. ", time() - start)


def test_rdd_fast_coset_divide():
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    rdd_arr = sc.parallelize(arr)
    print("test_rdd_fast_coset_divide")
    start = time()
    values1 = rdd_fast_coset_divide(rdd_arr, rdd_arr, g, primitive_root, n)
    print("finished. ", time() - start)
    sc.stop()


def test_poly_scale():
    coefficients = [v for (_, v) in arr]
    poly = Polynomial(coefficients)
    print("test_poly_scale")
    start = time()
    values1 = poly.scale(g)
    print("finished. ", time() - start)


def test_rdd_poly_scale():
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    rdd_arr = sc.parallelize(arr)
    print("test_rdd_poly_scale")
    start = time()
    values1 = poly_scale(rdd_arr, g)
    print("finished. ", time() - start)
    sc.stop()


def test_poly_combine():
    coefficients = [v for (_, v) in arr]
    poly = Polynomial(coefficients)
    print("test_poly_combine")
    start = time()
    values1 = poly * Polynomial([g]) + poly * Polynomial([primitive_root])
    print("finished. ", time() - start)


def test_rdd_poly_combine():
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    rdd_arr = sc.parallelize(arr)
    print("test_rdd_poly_combine")
    start = time()
    values1 = poly_combine(rdd_arr, rdd_arr, primitive_root, g)
    print("finished. ", time() - start)
    sc.stop()


# test_rdd_ntt()
# test_ntt()

# test_rdd_fast_coset_evaluate()
# test_fast_coset_evaluate()

# test_rdd_fast_multiply()
# test_fast_multiply()

# test_rdd_fast_coset_divide()  # 这两个都慢
# test_fast_coset_divide()

test_rdd_poly_scale()
test_poly_scale()

# test_rdd_poly_combine()
# test_poly_combine()
