from pyspark import SparkConf
from base.algebra import FieldElement
from base.ntt import fast_multiply
from rdd.rdd_poly import *
from rdd.rdd_fast_stark import FastStark as RddFastStark


conf = SparkConf().set("spark.driver.memory", "8g").set("spark.executor.memory", "4g")
sc = SparkContext(conf=conf)

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


def test_poly_mul_x():
    poly = rdd_field_list([1, 2, 3, 4])
    poly = poly_mul_x(poly, 3)
    assert poly.values().collect() == field_list([0, 0, 0, 1, 2, 3, 4])


def test_poly_combine():
    poly1 = rdd_field_list([1, 2, 3, 4])
    a = FieldElement(2, f)
    poly2 = rdd_field_list([1, 1, 1, 1])
    b = FieldElement(3, f)
    poly = poly_combine(poly1, poly2, a, b)
    assert poly.values().collect() == field_list([5, 7, 9, 11])

    poly3 = rdd_field_list([6, 4, 2, 0])
    c = FieldElement(1, f)
    poly = poly_combine_list([poly1, poly2, poly3], [a, b, c])
    assert poly.values().collect() == field_list([11, 11, 11, 11])


def test_rdd_take_by_indexs():
    rdd = rdd_field_list(list(range(2, 102)))
    assert rdd_take_by_indexs(rdd, [50, 70]) == dict(
        {50: FieldElement(52, f), 70: FieldElement(72, f)}
    )


def test_poly_exp():
    field = Field.main()

    logn = 10
    n = 1 << logn
    primitive_root = field.primitive_nth_root(n)

    poly = rdd_field_list([1, 2, 3, 4])
    poly1 = poly
    exp = 5
    for i in range(1, exp):
        poly1 = rdd_fast_multiply(poly1, poly, primitive_root, n)

    poly2 = poly_exp(poly, exp, primitive_root, n)
    assert poly1.collect() == poly2.collect()


# test_poly_sub_list()
# test_poly_append_zero()
# test_poly_degree()
# test_poly_scale()
# test_poly_mul_x()
# test_poly_combine()
# test_rdd_take_by_indexs()
test_poly_exp()

sc.stop()
