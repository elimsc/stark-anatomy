from numpy import poly
from pyspark import RDD, SparkContext, StorageLevel

from base.algebra import Field
import math
from base.univariate import *

# '001' => '100', return number
def reverse_bits(num, n):
    num_bits = bin(num)[2:]
    if len(num_bits) < n:
        num_bits = (n - len(num_bits)) * "0" + num_bits
    return int(num_bits[::-1], 2)


# 自下而上, cooley-Tukey FFT
# http://blog.jierenchen.com/2010/08/fft-with-mapreduce.html
def ntt1(primitive_root, values):
    n = len(values)
    assert n & (n - 1) == 0, "cannot compute ntt of non-power-of-two sequence"
    field = values[0].field
    assert (
        primitive_root ^ n == field.one()
    ), "primitive root must be nth root of unity, where n is len(values)"
    assert (
        primitive_root ^ (n // 2) != field.one()
    ), "primitive root is not primitive nth root of unity, where n is len(values)//2"

    if n <= 1:
        return values

    n_bits = int(math.log2(n))
    depth = n_bits

    cur_layer = []
    for i in range(n):
        cur_layer += [[values[reverse_bits(i, n_bits)]]]

    for i in range(depth):
        j = 0
        layer_length = 2 ** (depth - i)
        layer_primitive_root = primitive_root ^ (layer_length // 2)
        item_length = 2**i

        prev_layer = []  # 上一层
        while j < layer_length:
            evens = cur_layer[j]
            odds = cur_layer[j + 1]
            prev_layer += [
                [
                    evens[k % item_length]
                    + (layer_primitive_root ^ k) * odds[k % item_length]
                    for k in range(item_length * 2)
                ]
            ]
            j += 2
        cur_layer = prev_layer

    return cur_layer[0]


def rdd_ntt(
    primitive_root, root_order, values: RDD
) -> RDD:  # values: RDD[(index,value)]
    # n = values.count()
    n = root_order
    assert n & (n - 1) == 0, "cannot compute intt of non-power-of-two sequence"

    if n == 1:
        return values

    log_n = int(math.log2(n))
    r = 2 ** (log_n // 2)
    sc = values.context

    return (
        values.map(lambda x: (x[0] % r, (x[0] // r, x[1])))
        .Key(sc.defaultParallelism * 2)  # r
        .persist(StorageLevel.MEMORY_AND_DISK)
        .mapValues(list)
        .flatMap(
            lambda x: [
                (x[0], v)
                for v in enumerate(
                    ntt1(primitive_root ^ r, [v for (_, v) in sorted(x[1])])
                )
            ]
        )
        .map(
            # k: x[0], i: x[1][0], v: x[1][1]
            lambda x: (x[1][0], (x[0], x[1][1] * (primitive_root ^ (x[0] * x[1][0]))))
        )
        .groupByKey(sc.defaultParallelism * 2)  # n // r
        .persist(StorageLevel.MEMORY_AND_DISK)
        .mapValues(list)
        .flatMap(
            lambda x: [
                (x[0] * r + v[0], v[1])
                for v in enumerate(
                    ntt1(primitive_root ^ (n // r), [v for (_, v) in sorted(x[1])])
                )
            ]
        )
        .map(lambda x: ((x[0] % r) * (n // r) + x[0] // r, x[1]))
        .sortByKey()
        .persist(StorageLevel.MEMORY_AND_DISK)
    )


def rdd_intt(
    primitive_root,
    root_order,
    ninv,
    values: RDD,
) -> RDD:
    transformed_values = rdd_ntt(primitive_root.inverse(), root_order, values)
    return transformed_values.map(lambda x: (x[0], x[1] * ninv)).persist(
        StorageLevel.MEMORY_AND_DISK
    )


def rdd_fast_coset_evaluate(polynomial: RDD, offset, generator, order):
    scaled_polynomial = poly_scale(polynomial, offset)
    cur_len = scaled_polynomial.count()
    values = rdd_ntt(
        generator,
        order,
        poly_append_zero(scaled_polynomial, cur_len, order - cur_len),
    )
    return values.persist(StorageLevel.MEMORY_AND_DISK)


# lhs, rhs为RDD，输出为RDD, [] * []
def rdd_fast_multiply(lhs: RDD, rhs: RDD, primitive_root, root_order) -> RDD:
    assert (
        primitive_root ^ root_order == primitive_root.field.one()
    ), "supplied root does not have supplied order"
    assert (
        primitive_root ^ (root_order // 2) != primitive_root.field.one()
    ), "supplied root is not primitive root of supplied order"

    # if lhs.is_zero() or rhs.is_zero():
    #     return Polynomial([])

    root = primitive_root
    order = root_order
    lhs_degree = poly_degree(lhs)
    rhs_degree = poly_degree(rhs)
    degree = lhs_degree + rhs_degree

    if degree < 8:
        sc = lhs.context
        lhs = Polynomial(lhs.values().collect())
        rhs = Polynomial(rhs.values().collect())
        poly = lhs * rhs
        return sc.parallelize(list(enumerate(poly.coefficients)))

    while degree < order // 2:
        root = root ^ 2
        order = order // 2

    assert root ^ order == root.field.one()

    ninv = FieldElement(order, root.field).inverse()

    if lhs_degree + 1 < order:
        lhs = poly_append_zero(lhs, lhs_degree + 1, order - (lhs_degree + 1))
    if rhs_degree + 1 < order:
        rhs = poly_append_zero(rhs, rhs_degree + 1, order - (rhs_degree + 1))

    # print("lhs, rhs, order", lhs.count(), rhs.count(), order)

    lhs_codeword = rdd_ntt(root, order, lhs)
    rhs_codeword = rdd_ntt(root, order, rhs)

    hadamard_product = lhs_codeword.zip(rhs_codeword).map(
        lambda x: (x[0][0], x[0][1] * x[1][1])
    )
    product_coefficients = rdd_intt(root, order, ninv, hadamard_product)

    return product_coefficients.filter(lambda x: x[0] <= degree).persist(
        StorageLevel.MEMORY_AND_DISK
    )


# lhs, rhs 为RDD，输出为RDD
def rdd_fast_coset_divide(
    lhs: RDD, rhs: RDD, offset, primitive_root, root_order
) -> RDD:  # clean division only!
    assert (
        primitive_root ^ root_order == primitive_root.field.one()
    ), "supplied root does not have supplied order"
    assert (
        primitive_root ^ (root_order // 2) != primitive_root.field.one()
    ), "supplied root is not primitive root of supplied order"
    # assert not rhs.is_zero(), "cannot divide by zero polynomial"

    # if lhs.is_zero():
    #     return Polynomial([])

    root = primitive_root
    order = root_order
    lhs_degree = poly_degree(lhs)
    rhs_degree = poly_degree(rhs)
    assert rhs_degree <= lhs_degree, "cannot divide by polynomial of larger degree"
    degree = max(lhs_degree, rhs_degree)

    if degree < 8:
        sc = lhs.context
        lhs = Polynomial(lhs.values().collect())
        rhs = Polynomial(rhs.values().collect())
        poly = lhs / rhs
        return sc.parallelize(poly.coefficients)

    while degree < order // 2:
        root = root ^ 2
        order = order // 2

    scaled_lhs = poly_scale(lhs, offset)
    scaled_rhs = poly_scale(rhs, offset)

    scaled_lhs = poly_append_zero(scaled_lhs, lhs_degree + 1, order - (lhs_degree + 1))
    scaled_rhs = poly_append_zero(scaled_rhs, rhs_degree + 1, order - (rhs_degree + 1))

    lhs_codeword = rdd_ntt(root, order, scaled_lhs)
    rhs_codeword = rdd_ntt(root, order, scaled_rhs)

    quotient_codeword = lhs_codeword.zip(rhs_codeword).map(
        lambda x: (x[0][0], x[0][1] / x[1][1])
    )

    ninv = FieldElement(order, root.field).inverse()
    scaled_quotient = rdd_intt(root, order, ninv, quotient_codeword)
    scaled_quotient = scaled_quotient.filter(
        lambda x: x[0] <= (lhs_degree - rhs_degree)
    )

    return poly_scale(scaled_quotient, offset.inverse()).persist(
        StorageLevel.MEMORY_AND_DISK
    )


# --------------- poly ops -----------------------------


def poly_scale(poly: RDD, factor) -> RDD:  # poly: RDD[(index,v)]
    return poly.map(lambda x: (x[0], (factor ^ x[0]) * x[1])).persist(
        StorageLevel.MEMORY_AND_DISK
    )


def poly_degree(poly: RDD) -> int:
    return poly.filter(lambda x: x[1] != x[1].field.zero()).keys().max()


def poly_append_zero(poly: RDD, start, zero_count: int) -> RDD:
    zero = Field.main().zero()
    sc = poly.context
    poly = poly.union(sc.parallelize([(start + i, zero) for i in range(zero_count)]))
    return poly.persist(StorageLevel.MEMORY_AND_DISK)


def poly_mul_x(poly: RDD, n) -> RDD:
    zero = Field.main().zero()
    sc = poly.context
    poly = poly.map(lambda x: (x[0] + n, x[1]))
    poly = sc.parallelize([i, zero] for i in range(n)).union(poly)
    return poly.persist(StorageLevel.MEMORY_AND_DISK)


# a * poly1 + b * poly2
def poly_combine(poly1: RDD, poly2: RDD, a, b) -> RDD:
    poly1 = poly_mul_constant(poly1, a)
    poly2 = poly_mul_constant(poly2, b)

    return poly_add(poly1, poly2)


def poly_combine_list(polys: list, vs: list) -> RDD:
    assert len(polys) == len(vs)
    assert len(vs) > 2
    one = vs[0].field.one()
    poly = poly_combine(polys[0], polys[1], vs[0], vs[1])
    for i in range(2, len(vs)):
        poly = poly_combine(poly, polys[i], one, vs[i])
    return poly.persist(StorageLevel.MEMORY_AND_DISK)


def poly_sub_list(lhs: RDD, rhs: list) -> RDD:
    # assert lhs.count() >= len(rhs)

    def sub(x, rhs):
        if x[0] < len(rhs):
            return (x[0], x[1] - rhs[x[0]])
        return x

    return lhs.map(lambda x: sub(x, rhs)).persist(StorageLevel.MEMORY_AND_DISK)


def poly_mul_constant(lhs: RDD, constant) -> RDD:
    return lhs.map(lambda x: (x[0], x[1] * constant)).persist(
        StorageLevel.MEMORY_AND_DISK
    )


def poly_exp(rdd: RDD, exponent, primitive_root, root_order) -> RDD:
    assert exponent >= 0
    one = rdd.context.parallelize([(0, primitive_root.field.one())])
    if exponent == 0:
        return one
    acc = one
    for i in reversed(range(len(bin(exponent)[2:]))):
        acc = rdd_fast_multiply(acc, acc, primitive_root, root_order)
        if (1 << i) & exponent != 0:
            acc = rdd_fast_multiply(acc, rdd, primitive_root, root_order)
    return acc.persist(StorageLevel.MEMORY_AND_DISK)


def poly_add(lhs: RDD, rhs: RDD) -> RDD:
    def sum_arr(l: list):
        assert len(l) > 0
        sum = l[0]
        for i in range(1, len(l)):
            sum += l[i]
        return sum

    sc = lhs.context
    return (
        lhs.union(rhs)
        .groupByKey(sc.defaultParallelism * 2)
        .mapValues(list)
        .map(lambda x: (x[0], sum_arr(x[1])))
        .persist(StorageLevel.MEMORY_AND_DISK)
    )


def poly_sub(lhs: RDD, rhs: RDD) -> RDD:
    def sub_arr(l: list):
        assert len(l) > 0
        r = l[0]
        for i in range(1, len(l)):
            r -= l[i]
        return r

    sc = lhs.context
    return (
        lhs.union(rhs)
        .groupByKey(sc.defaultParallelism * 2)
        .mapValues(list)
        .map(lambda x: (x[0], sub_arr(x[1])))
        .persist(StorageLevel.MEMORY_AND_DISK)
    )


def rdd_take_by_indexs(rdd: RDD, indexs: list):
    return rdd.filter(lambda x: x[0] in indexs).collectAsMap()


# [g ^ i for i in range(size)]
def generate_domain(sc: SparkContext, g, size) -> RDD:  # domain: RDD[(i, X)]
    result = sc.parallelize([])
    for i in range(size):
        result = result.union(sc.parallelize([(i, g ^ i)]))
    return result


# class RddPolynomial:
#     """
#     定义操作: +, -, *, /, ^
#     primitive_root, root_order用于计算 *, /
#     """

#     def __init__(self, rdd: RDD, primitive_root, root_order):
#         assert primitive_root ^ root_order == primitive_root.field.one()
#         self.rdd = rdd
#         self.primitive_root = primitive_root
#         self.root_order = root_order

#     def __neg__(self):
#         return RddPolynomial(
#             self.rdd.map(lambda x: (x[0], -x[1])), self.primitive_root, self.root_order
#         )

#     def __add__(self, other):
#         def sum_arr(l: list):
#             assert len(l) > 0
#             sum = l[0]
#             for i in range(1, len(l)):
#                 sum += l[i]
#             return sum

#         return RddPolynomial(
#             self.rdd.union(other)
#             .groupByKey()
#             .mapValues(list)
#             .map(lambda x: (x[0], sum_arr(x[1]))),
#             self.primitive_root,
#             self.root_order,
#         )

#     def __sub__(self, other):
#         return self.__add__(-other)

#     def __mul__(self, other):  # other是一个值, 不是一个多项式
#         val = other.rdd.first()[1]
#         rdd = self.rdd.map(lambda x: (x[0], val * x[1]))
#         return RddPolynomial(
#             rdd,
#             self.primitive_root,
#             self.root_order,
#         )

#     def __truediv__(self, other):
#         return RddPolynomial(
#             rdd_fast_coset_divide(
#                 self.rdd, other, self.primitive_root, self.root_order
#             ),
#             self.primitive_root,
#             self.root_order,
#         )

#     def __xor__(self, exponent):
#         one = self.rdd.context.parallelize([(0, self.primitive_root.field.one())])
#         if exponent == 0:
#             return one
#         acc = one
#         for i in reversed(range(len(bin(exponent)[2:]))):
#             acc = rdd_fast_multiply(acc, acc, self.primitive_root, self.root_order)
#             if (1 << i) & exponent != 0:
#                 acc = rdd_fast_multiply(
#                     acc, self.rdd, self.primitive_root, self.root_order
#                 )
#         return RddPolynomial(acc, self.primitive_root, self.root_order)
