from pyspark import SparkConf, SparkContext
from base.algebra import *
from rdd.rdd_poly import poly_degree
from base.univariate import *
from base.ntt import *
import os
from rdd.rdd_poly import (
    ntt1,
    rdd_fast_coset_divide,
    rdd_fast_coset_evaluate,
    rdd_fast_multiply,
    rdd_fast_zerofier,
    rdd_intt,
    rdd_ntt,
)

conf = SparkConf().set("spark.driver.memory", "8g").set("spark.executor.memory", "4g")
sc = SparkContext(conf=conf)


def test_ntt():
    field = Field.main()
    logn = 7
    n = 1 << logn
    primitive_root = field.primitive_nth_root(n)

    coefficients = [field.sample(os.urandom(17)) for i in range(n)]
    poly = Polynomial(coefficients)

    rdd_cofs = sc.parallelize(list(enumerate(coefficients)))

    values = ntt(primitive_root, coefficients)
    values1 = ntt1(primitive_root, coefficients)
    rdd_value = rdd_ntt(primitive_root, rdd_cofs)
    values2 = rdd_value.values().collect()

    values_again = poly.evaluate_domain(
        [primitive_root ^ i for i in range(len(values))]
    )

    assert values == values_again, "ntt does not compute correct batch-evaluation"
    assert values == values1, "rdd_ntt does not compute correct batch-evaluation"
    assert values == values2, "rdd_ntt does not compute correct batch-evaluation"


def test_intt():
    field = Field.main()

    logn = 7
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

    for trial in range(5):
        lhs_degree = int(os.urandom(1)[0]) % (n // 2)
        rhs_degree = int(os.urandom(1)[0]) % (n // 2)

        lhs = Polynomial([field.sample(os.urandom(17)) for i in range(lhs_degree + 1)])
        rhs = Polynomial([field.sample(os.urandom(17)) for i in range(rhs_degree + 1)])

        fast_product = fast_multiply(lhs, rhs, primitive_root, n)
        slow_product = lhs * rhs
        lhs1 = sc.parallelize(list(enumerate(lhs.coefficients)))
        rhs1 = sc.parallelize(list(enumerate(rhs.coefficients)))
        product1 = rdd_fast_multiply(lhs1, rhs1, primitive_root, n)

        assert len(slow_product.coefficients) == product1.count()

        assert fast_product == slow_product, "fast product does not equal slow product"
        assert slow_product.coefficients == product1.values().collect()


def test_divide():
    field = Field.main()

    logn = 6
    n = 1 << logn
    primitive_root = field.primitive_nth_root(n)

    for trial in range(10):
        lhs_degree = int(os.urandom(1)[0]) % (n // 2)
        rhs_degree = int(os.urandom(1)[0]) % (n // 2)

        lhs_coffs = [field.sample(os.urandom(17)) for i in range(lhs_degree + 1)]
        rhs_coffs = [field.sample(os.urandom(17)) for i in range(rhs_degree + 1)]

        lhs = Polynomial(lhs_coffs)
        rhs = Polynomial(rhs_coffs)

        fast_product = fast_multiply(lhs, rhs, primitive_root, n)
        quotient = fast_coset_divide(
            fast_product, lhs, field.generator(), primitive_root, n
        )

        lhs1 = sc.parallelize(list(enumerate(fast_product.coefficients)))
        rhs1 = sc.parallelize(list(enumerate(lhs.coefficients)))
        quotient1 = rdd_fast_coset_divide(
            lhs1, rhs1, field.generator(), primitive_root, n
        )

        assert quotient1.count() == len(quotient.coefficients)
        assert quotient1.values().collect() == quotient.coefficients

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

    rdd_poly = sc.parallelize(list(enumerate(coefficients)))

    values_fast = fast_coset_evaluate(poly, two, primitive_root, n)
    values_traditional = [poly.evaluate(d) for d in domain]
    values_rdd = rdd_fast_coset_evaluate(rdd_poly, two, primitive_root, n)

    assert all(
        vf == vt for (vf, vt) in zip(values_fast, values_traditional)
    ), "values do not match with traditional evaluations"
    assert values_fast == values_rdd.values().collect()


def test_fast_zerofier():
    field = Field.main()

    logn = 9
    n = 1 << logn
    n1 = 1 << (logn - 2)
    primitive_root = field.primitive_nth_root(n)
    domain = [primitive_root ^ i for i in range(n1)]
    zerofier1 = fast_zerofier(domain, primitive_root, n)

    rdd_domain = sc.parallelize(list(enumerate(domain)))
    # zerofier2 = rdd_fast_zerofier(rdd_domain, primitive_root, n)
    assert zerofier1.coefficients == zerofier2.values().collect()


# test_ntt()
# test_intt()
# test_coset_evaluate()
# test_divide()
test_multiply()
# test_fast_zerofier()

sc.stop()
