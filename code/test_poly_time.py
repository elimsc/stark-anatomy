from os import urandom

from pyspark import SparkConf, SparkContext, StorageLevel
from time import time

from base.algebra import *
from base.ntt import fast_coset_divide, fast_coset_evaluate, fast_multiply
from base.univariate import Polynomial
from rdd.rdd_poly import (
    ntt1,
    poly_add,
    poly_combine,
    poly_mul_constant,
    poly_scale,
    poly_sub,
    rdd_fast_coset_divide,
    rdd_fast_coset_evaluate,
    rdd_fast_multiply,
    rdd_ntt,
)
import sys

from test_spark import get_sc


field = Field.main()

g = field.generator()


def test_poly_scale():  # 这个比较耗时
    coefficients = [v for (_, v) in arr]
    poly = Polynomial(coefficients)
    print("test_poly_scale")
    start = time()
    values1 = poly.scale(g)
    print("finished. ", time() - start)


def test_rdd_poly_scale():
    sc = get_sc()

    rdd_arr = sc.parallelize(arr)
    print("test_rdd_poly_scale")
    start = time()
    values1 = poly_scale(rdd_arr, g).collect()
    print("finished. ", time() - start)
    sc.stop()


def test_poly_add():
    coefficients = [v for (_, v) in arr]
    poly = Polynomial(coefficients)
    coefficients1 = [v for (_, v) in arr1]
    poly1 = Polynomial(coefficients1)
    print("test_poly_add")
    start = time()
    values1 = poly + poly1
    print("finished. ", time() - start)


def test_rdd_poly_add():
    sc = get_sc()

    rdd_arr = sc.parallelize(arr)
    rdd_arr1 = sc.parallelize(arr)
    print("test_rdd_poly_add")
    start = time()
    values1 = poly_add(rdd_arr, rdd_arr1).collect()
    print("finished. ", time() - start)
    sc.stop()


def test_poly_sub():
    coefficients = [v for (_, v) in arr]
    poly = Polynomial(coefficients)
    coefficients1 = [v for (_, v) in arr1]
    poly1 = Polynomial(coefficients1)
    print("test_poly_sub")
    start = time()
    values1 = poly - poly1
    print("finished. ", time() - start)


def test_rdd_poly_sub():
    sc = get_sc()

    rdd_arr = sc.parallelize(arr)
    rdd_arr1 = sc.parallelize(arr1)
    print("test_rdd_poly_sub")
    start = time()
    values1 = poly_sub(rdd_arr, rdd_arr1).collect()
    print("finished. ", time() - start)
    sc.stop()


def test_poly_mul_constant():
    coefficients = [v for (_, v) in arr]
    poly = Polynomial(coefficients)
    print("test_poly_mul_constant")
    start = time()
    values1 = Polynomial([v * coefficients[0] for v in poly.coefficients])
    print("finished. ", time() - start)


def test_rdd_poly_mul_constant():
    sc = get_sc()

    rdd_arr = sc.parallelize(arr)
    print("test_rdd_poly_mul_constant")
    start = time()
    values1 = poly_mul_constant(rdd_arr, arr[0][1]).collect()
    print("finished. ", time() - start)
    sc.stop()


if __name__ == "__main__":
    mode = int(sys.argv[1])
    logn = int(sys.argv[2])  # 15
    n = 1 << logn
    print(n)
    primitive_root = field.primitive_nth_root(n)
    arr = [(i, field.sample(urandom(17))) for i in range(n)]
    arr1 = [(i, field.sample(urandom(17))) for i in range(n)]
    if mode == 0:
        test_poly_scale()
        # test_poly_add()
        # test_poly_sub()
        # test_poly_mul_constant()
    elif mode == 1:
        test_rdd_poly_scale()
        # test_rdd_poly_add()
        # test_rdd_poly_sub()
        # test_rdd_poly_mul_constant()
    else:
        test_poly_scale()
        test_rdd_poly_scale()

        # test_poly_add()
        # test_rdd_poly_add()

        # test_poly_sub()
        # test_rdd_poly_sub()

        # test_poly_mul_constant()
        # test_rdd_poly_mul_constant()
