from algebra import FieldElement
from rdd_poly import *
from rdd_fast_stark import FastStark as RddFastStark

from pyspark import SparkContext, SparkConf

conf = (
    SparkConf().setAppName("test_fast_stark").setMaster("spark://zhdeMacBook-Pro:7077")
)
sc = SparkContext(conf=conf)

sc.addPyFile("./algebra.py")
sc.addPyFile("./rdd_ntt.py")
sc.addPyFile("./univariate.py")
sc.addPyFile("./rdd_merkle.py")

f = Field.main()


def rdd_field_list(l):
    return sc.parallelize([(i, FieldElement(v, f)) for (i, v) in enumerate(l)])


def field_list(l):
    return [FieldElement(v, f) for (_, v) in enumerate(l)]


def test_poly_sub_list():
    lhs = rdd_field_list([3, 2, 3, 4])
    rhs = field_list([2, 2, 2])
    r = poly_sub_list(lhs, rhs).values().collect()
    assert r == field_list([1, 0, 1, 4])


def test_poly_append_zero():
    poly = rdd_field_list([1, 2, 3, 4])
    poly = poly_append_zero(poly, 4, 3).values().collect()
    assert poly == field_list([1, 2, 3, 4, 0, 0, 0])


def test_poly_degree():
    poly = rdd_field_list([0, 0, 1, 5, 3, 4, 0, 0])
    assert poly_degree(poly) == 5


def test_poly_scale():
    poly = rdd_field_list([1, 2, 3, 4])
    poly = poly_scale(poly, FieldElement(2, f))
    assert poly.values().collect() == field_list([1, 4, 12, 32])


test_poly_sub_list()
test_poly_append_zero()
test_poly_degree()
test_poly_scale()
