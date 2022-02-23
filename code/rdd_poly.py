from pyspark import RDD, SparkContext

from algebra import Field


def poly_scale(poly: RDD, factor) -> RDD:  # poly: RDD[(index,v)]
    return poly.map(lambda x: (x[0], (factor ^ x[0]) * x[1]))


def poly_degree(poly: RDD) -> int:
    return poly.filter(lambda x: x[1] != x[1].field.zero()).keys().max()


def poly_append_zero(poly: RDD, start, zero_count: int) -> RDD:
    zero = Field.main().zero()
    sc = poly.context
    poly = poly.union(sc.parallelize([(start + i, zero) for i in range(zero_count)]))
    return poly


def poly_mul_x(poly: RDD, n) -> RDD:
    zero = Field.main().zero()
    sc = poly.context
    poly = poly.map(lambda x: (x[0] + n, x[1]))
    poly = sc.parallelize([i, zero] for i in range(n)).union(poly)
    return poly


# a * poly1 + b * poly2
def poly_combine(poly1: RDD, poly2: RDD, a, b) -> RDD:
    poly1 = poly1.map(lambda x: (x[0], x[1] * a))
    poly2 = poly2.map(lambda x: (x[0], x[1] * b))

    def sum_list(l):
        sum = l[0]
        for v in l[1:]:
            sum += v
        return sum

    return (
        poly1.union(poly2)
        .groupByKey()
        .mapValues(list)
        .map(lambda x: (x[0], sum_list(x[1])))
    )


def poly_combine_list(polys: list[RDD], vs: list) -> RDD:
    assert len(polys) == len(vs)
    assert len(vs) > 2
    one = vs[0].field.one()
    poly = poly_combine(polys[0], polys[1], vs[0], vs[1])
    for i in range(2, len(vs)):
        poly = poly_combine(poly, polys[i], one, vs[i])
    return poly


def poly_sub_list(lhs: RDD, rhs: list) -> RDD:
    assert lhs.count() >= len(rhs)

    def sub(x, rhs):
        if x[0] < len(rhs):
            return (x[0], x[1] - rhs[x[0]])
        return x

    return lhs.map(lambda x: sub(x, rhs))


def rdd_take_by_indexs(rdd: RDD, indexs: list):
    return rdd.filter(lambda x: x[0] in indexs).collectAsMap()


# [g ^ i for i in range(size)]
def generate_domain(sc: SparkContext, g, size) -> RDD:  # domain: RDD[(i, X)]
    result = sc.parallelize([])
    for i in range(size):
        result = result.union(sc.parallelize([(i, g ^ i)]))
    return result
