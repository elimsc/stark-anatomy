from pyspark import RDD
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
                (v[0] * r + x[0], v[1])
                for v in enumerate(
                    ntt1(primitive_root ^ (n // r), [v for (_, v) in sorted(x[1])])
                )
            ]
        )
        .sortByKey()
    )


def rdd_intt(
    primitive_root,
    ninv,
    values: RDD,
) -> RDD:
    # print(values.collect())
    transformed_values = rdd_ntt(primitive_root.inverse(), values)
    # print(transformed_values.collect())
    return transformed_values.map(lambda x: (x[0], x[1] * ninv))


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

    # assert rhs.degree() <= lhs.degree(), "cannot divide by polynomial of larger degree"

    field = lhs.coefficients[0].field
    root = primitive_root
    order = root_order
    degree = max(lhs.degree(), rhs.degree())

    if degree < 8:
        return lhs / rhs

    while degree < order // 2:
        root = root ^ 2
        order = order // 2

    scaled_lhs = lhs.scale(offset)
    scaled_rhs = rhs.scale(offset)

    lhs_coefficients = scaled_lhs.coefficients[: (lhs.degree() + 1)]
    while len(lhs_coefficients) < order:
        lhs_coefficients += [field.zero()]
    rhs_coefficients = scaled_rhs.coefficients[: (rhs.degree() + 1)]
    while len(rhs_coefficients) < order:
        rhs_coefficients += [field.zero()]
    lhs_codeword = ntt(root, lhs_coefficients)
    rhs_codeword = ntt(root, rhs_coefficients)

    quotient_codeword = [l / r for (l, r) in zip(lhs_codeword, rhs_codeword)]
    scaled_quotient_coefficients = intt(root, quotient_codeword)
    scaled_quotient = Polynomial(
        scaled_quotient_coefficients[: (lhs.degree() - rhs.degree() + 1)]
    )

    return scaled_quotient.scale(offset.inverse())
