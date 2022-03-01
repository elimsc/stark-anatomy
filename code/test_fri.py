from pyspark import SparkConf, SparkContext
from base.algebra import *
from base.fri import *
from time import time

from rdd.rdd_fri import RddFri


def test_fri():
    field = Field.main()
    degree = 2**10 - 1  # 18
    expansion_factor = 4
    num_colinearity_tests = 17

    initial_codeword_length = (degree + 1) * expansion_factor
    log_codeword_length = 0
    codeword_length = initial_codeword_length
    while codeword_length > 1:
        codeword_length //= 2
        log_codeword_length += 1

    assert (
        1 << log_codeword_length == initial_codeword_length
    ), "log not computed correctly"

    omega = field.primitive_nth_root(initial_codeword_length)
    generator = field.generator()

    assert (
        omega ^ (1 << log_codeword_length) == field.one()
    ), "omega not nth root of unity"
    assert (
        omega ^ (1 << (log_codeword_length - 1)) != field.one()
    ), "omega not primitive"

    fri = Fri(
        generator,
        omega,
        initial_codeword_length,
        expansion_factor,
        num_colinearity_tests,
    )

    polynomial = Polynomial([FieldElement(i, field) for i in range(degree + 1)])
    domain = [omega ^ i for i in range(initial_codeword_length)]

    start = time()
    codeword = fast_coset_evaluate(
        polynomial, FieldElement(1, field), omega, initial_codeword_length
    )
    print("ntt time:", time() - start)

    # test valid codeword
    print("testing valid codeword ...")
    proof_stream = ProofStream()

    start = time()
    fri.prove(codeword, proof_stream)
    print("fri prove time:", time() - start)
    points = []

    start = time()
    verdict = fri.verify(proof_stream, points)
    print("fri verify time:", time() - start)
    if verdict == False:
        print("rejecting proof, but proof should be valid!")
        return

    for (x, y) in points:
        if polynomial.evaluate(omega ^ x) != y:
            print("polynomial evaluates to wrong value")
            assert False
    print("success! \\o/")

    # disturb then test for failure
    print("testing invalid codeword ...")
    proof_stream = ProofStream()
    for i in range(0, degree // 3):
        codeword[i] = field.zero()

    fri.prove(codeword, proof_stream)
    points = []
    assert False == fri.verify(
        proof_stream, points
    ), "proof should fail, but is accepted ..."
    print("success! \\o/")


conf = SparkConf().set("spark.driver.memory", "8g").set("spark.executor.memory", "4g")
sc = SparkContext(conf=conf)


def test_rdd_fri():
    field = Field.main()
    degree = 2**8 - 1
    expansion_factor = 4
    num_colinearity_tests = 17

    initial_codeword_length = (degree + 1) * expansion_factor
    log_codeword_length = 0
    codeword_length = initial_codeword_length
    while codeword_length > 1:
        codeword_length //= 2
        log_codeword_length += 1

    assert (
        1 << log_codeword_length == initial_codeword_length
    ), "log not computed correctly"

    omega = field.primitive_nth_root(initial_codeword_length)
    generator = field.generator()

    assert (
        omega ^ (1 << log_codeword_length) == field.one()
    ), "omega not nth root of unity"
    assert (
        omega ^ (1 << (log_codeword_length - 1)) != field.one()
    ), "omega not primitive"

    fri = RddFri(
        generator,
        omega,
        initial_codeword_length,
        expansion_factor,
        num_colinearity_tests,
    )

    polynomial = Polynomial([FieldElement(i, field) for i in range(degree + 1)])
    domain = [omega ^ i for i in range(initial_codeword_length)]

    start = time()
    codeword = fast_coset_evaluate(
        polynomial, FieldElement(1, field), omega, initial_codeword_length
    )
    print("ntt time:", time() - start)

    rdd_codeword = sc.parallelize(list(enumerate(codeword)))

    # test valid codeword
    print("testing valid codeword ...")
    proof_stream = ProofStream()

    start = time()
    fri.prove(rdd_codeword, proof_stream)
    print("fri prove time:", time() - start)
    points = []

    start = time()
    verdict = fri.verify(proof_stream, points)
    print("fri verify time:", time() - start)
    if verdict == False:
        print("rejecting proof, but proof should be valid!")
        return

    for (x, y) in points:
        if polynomial.evaluate(omega ^ x) != y:
            print("polynomial evaluates to wrong value")
            assert False
    print("success! \\o/")


# test_fri()
test_rdd_fri()
sc.stop()
