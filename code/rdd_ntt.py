from pyspark import RDD
from rdd_poly import poly_append_zero, poly_degree, poly_scale
from univariate import *
import math

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


def rdd_ntt(primitive_root, values: RDD) -> RDD:  # values: RDD[(index,value)]
    n = values.count()
    assert n & (n - 1) == 0, "cannot compute intt of non-power-of-two sequence"

    if n == 1:
        return values

    log_n = int(math.log2(n))
    r = 2 ** (log_n // 2)

    return (
        values.map(lambda x: (x[0] % r, (x[0] // r, x[1])))
        .groupByKey()
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
        .groupByKey()
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
    )


def rdd_intt(
    primitive_root,
    ninv,
    values: RDD,
) -> RDD:
    transformed_values = rdd_ntt(primitive_root.inverse(), values)
    return transformed_values.map(lambda x: (x[0], x[1] * ninv))


def rdd_fast_coset_evaluate(polynomial: RDD, offset, generator, order):
    scaled_polynomial = poly_scale(polynomial, offset)
    cur_len = scaled_polynomial.count()
    values = rdd_ntt(
        generator,
        poly_append_zero(scaled_polynomial, cur_len, order - cur_len),
    )
    return values


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
        return sc.parallelize(poly.coefficients)

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

    lhs_codeword = rdd_ntt(root, lhs)
    rhs_codeword = rdd_ntt(root, rhs)

    hadamard_product = lhs_codeword.zip(rhs_codeword).map(
        lambda x: (x[0][0], x[0][1] * x[1][1])
    )
    product_coefficients = rdd_intt(root, ninv, hadamard_product)

    return product_coefficients.filter(lambda x: x[0] <= degree)


def rdd_fast_zerofier(domain: RDD, primitive_root, root_order) -> RDD:
    assert (
        primitive_root ^ root_order == primitive_root.field.one()
    ), "supplied root does not have supplied order"
    assert (
        primitive_root ^ (root_order // 2) != primitive_root.field.one()
    ), "supplied root is not primitive root of supplied order"

    if len(domain) == 0:
        return Polynomial([])

    if len(domain) == 1:
        return Polynomial([-domain[0], primitive_root.field.one()])

    half = len(domain) // 2

    left = fast_zerofier(domain[:half], primitive_root, root_order)
    right = fast_zerofier(domain[half:], primitive_root, root_order)
    return fast_multiply(left, right, primitive_root, root_order)


# lhs, rhs 为RDD，输出为RDD
def rdd_fast_coset_divide(
    lhs: RDD, rhs: RDD, offset, primitive_root, root_order
):  # clean division only!
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

    lhs_codeword = rdd_ntt(root, scaled_lhs)
    rhs_codeword = rdd_ntt(root, scaled_rhs)

    quotient_codeword = lhs_codeword.zip(rhs_codeword).map(
        lambda x: (x[0][0], x[0][1] / x[1][1])
    )

    ninv = FieldElement(order, root.field).inverse()
    scaled_quotient = rdd_intt(root, ninv, quotient_codeword)
    scaled_quotient = scaled_quotient.filter(
        lambda x: x[0] <= (lhs_degree - rhs_degree)
    )

    return poly_scale(scaled_quotient, offset.inverse())
