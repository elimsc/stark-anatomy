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


# def poly_add(lhs: RDD, rhs: RDD) -> RDD:
#     return lhs.zip(rhs).map(lambda v: v[0] + v[1])


def poly_sub_list(lhs: RDD, rhs: list) -> RDD:
    assert lhs.count() >= len(rhs)

    def sub(x, rhs):
        if x[0] < len(rhs):
            return (x[0], x[1] - rhs[x[0]])
        return x

    return lhs.map(lambda x: sub(x, rhs))


# [g ^ i for i in range(size)]
def generate_domain(sc: SparkContext, g, size) -> RDD:  # domain: RDD[(i, X)]
    result = sc.parallelize([])
    for i in range(size):
        result = result.union(sc.parallelize([(i, g ^ i)]))
    return result
