from algebra import *
from univariate import *
from ntt import *
import os
from rdd_ntt import ntt1, rdd_intt, rdd_ntt

from pyspark import SparkContext, SparkConf


conf = (
    SparkConf().setAppName("test_fast_stark").setMaster("spark://zhdeMacBook-Pro:7077")
)
sc = SparkContext(conf=conf)

sc.addPyFile("./algebra.py")
sc.addPyFile("./rdd_ntt.py")
sc.addPyFile("./univariate.py")


def test_ntt():
    field = Field.main()
    logn = 8
    n = 1 << logn
    primitive_root = field.primitive_nth_root(n)

    coefficients = [field.sample(os.urandom(17)) for i in range(n)]
    poly = Polynomial(coefficients)

    rdd_cofs = sc.parallelize(list(enumerate(coefficients)))

    values = ntt(primitive_root, coefficients)
    values1 = ntt1(primitive_root, coefficients)
    rdd_value = rdd_ntt(primitive_root, rdd_cofs)
    values2 = [v for (_, v) in rdd_value.collect()]

    values_again = poly.evaluate_domain(
        [primitive_root ^ i for i in range(len(values))]
    )

    assert values == values_again, "ntt does not compute correct batch-evaluation"
    assert values == values1, "rdd_ntt does not compute correct batch-evaluation"
    assert values == values2, "rdd_ntt does not compute correct batch-evaluation"


def test_intt():
    field = Field.main()

    logn = 8  # 仅当为偶数是生效 TODO
    n = 1 << logn
    primitive_root = field.primitive_nth_root(n)

    ninv = FieldElement(n, field).inverse()

    values = [field.sample(os.urandom(1)) for i in range(n)]
    coeffs = ntt(primitive_root, values)

    rdd_coeffs = sc.parallelize(list(enumerate(coeffs)))

    values_again = intt(primitive_root, coeffs)
    rdd_value = rdd_intt(primitive_root, ninv, rdd_coeffs)
    values2 = [v for (_, v) in rdd_value.collect()]

    assert values == values_again, "inverse ntt is different from forward ntt"
    assert values == values2, "rdd_intt dont work"


def test_multiply():
    field = Field.main()

    logn = 6
    n = 1 << logn
    primitive_root = field.primitive_nth_root(n)

    for trial in range(20):
        lhs_degree = int(os.urandom(1)[0]) % (n // 2)
        rhs_degree = int(os.urandom(1)[0]) % (n // 2)

        lhs = Polynomial([field.sample(os.urandom(17)) for i in range(lhs_degree + 1)])
        rhs = Polynomial([field.sample(os.urandom(17)) for i in range(rhs_degree + 1)])

        fast_product = fast_multiply(lhs, rhs, primitive_root, n)
        slow_product = lhs * rhs

        assert fast_product == slow_product, "fast product does not equal slow product"


def test_divide():
    field = Field.main()

    logn = 6
    n = 1 << logn
    primitive_root = field.primitive_nth_root(n)

    for trial in range(20):
        lhs_degree = int(os.urandom(1)[0]) % (n // 2)
        rhs_degree = int(os.urandom(1)[0]) % (n // 2)

        lhs = Polynomial([field.sample(os.urandom(17)) for i in range(lhs_degree + 1)])
        rhs = Polynomial([field.sample(os.urandom(17)) for i in range(rhs_degree + 1)])

        fast_product = fast_multiply(lhs, rhs, primitive_root, n)
        quotient = fast_coset_divide(
            fast_product, lhs, field.generator(), primitive_root, n
        )

        assert quotient == rhs, "fast divide does not equal original factor"


def test_interpolate():
    field = Field.main()

    logn = 9
    n = 1 << logn
    primitive_root = field.primitive_nth_root(n)

    for trial in range(10):
        N = sum((1 << (8 * i)) * int(os.urandom(1)[0]) for i in range(8)) % n
        if N == 0:
            continue
        print("N:", N)
        values = [field.sample(os.urandom(17)) for i in range(N)]
        domain = [field.sample(os.urandom(17)) for i in range(N)]
        poly = fast_interpolate(domain, values, primitive_root, n)
        print("poly degree:", poly.degree())
        values_again = fast_evaluate(poly, domain, primitive_root, n)[0:N]
        # values_again = poly.evaluate_domain(domain)

        if values != values_again:
            print("fast interpolation and evaluation are not inverses")
            print("expected:", ",".join(str(c.value) for c in values))
            print("observed:", ",".join(str(c.value) for c in values_again))
            assert False
        print("")


def test_coset_evaluate():
    field = Field.main()

    logn = 9
    n = 1 << logn
    primitive_root = field.primitive_nth_root(n)

    two = FieldElement(2, field)

    domain = [two * (primitive_root ^ i) for i in range(n)]

    degree = ((int(os.urandom(1)[0]) * 256 + int(os.urandom(1)[0])) % n) - 1
    coefficients = [field.sample(os.urandom(17)) for i in range(degree + 1)]
    poly = Polynomial(coefficients)

    values_fast = fast_coset_evaluate(poly, two, primitive_root, n)
    values_traditional = [poly.evaluate(d) for d in domain]

    assert all(
        vf == vt for (vf, vt) in zip(values_fast, values_traditional)
    ), "values do not match with traditional evaluations"


test_ntt()
test_intt()
test_coset_evaluate()

sc.stop()
