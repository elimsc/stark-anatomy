from time import time
from algebra import *
from univariate import *
from multivariate import *
from rescue_prime import *
from fri import *
from ip import *
from fast_stark import *
import sys

from rdd_fast_stark import FastStark as RddFastStark

from test_spark import get_sc

# sc = get_sc("test_fast_stark")


def test_fast_stark():
    field = Field.main()
    expansion_factor = 4
    num_colinearity_checks = 2
    security_level = 2

    rp = RescuePrime()
    output_element = field.sample(bytes(b"0xdeadbeef"))

    for trial in range(0, 1):  # 20
        input_element = output_element
        print("running trial with input:", input_element.value)
        output_element = rp.hash(input_element)
        num_cycles = rp.N + 1
        state_width = rp.m

        stark = FastStark(
            field,
            expansion_factor,
            num_colinearity_checks,
            security_level,
            state_width,
            num_cycles,
        )
        (
            transition_zerofier,
            transition_zerofier_codeword,
            transition_zerofier_root,
        ) = stark.preprocess()

        # prove honestly
        print("honest proof generation ...")

        # prove
        trace = rp.trace(input_element)
        air = rp.transition_constraints(stark.omicron, stark.omicron_domain_length)
        boundary = rp.boundary_constraints(output_element)
        start = time()
        proof = stark.prove(
            trace, air, boundary, transition_zerofier, transition_zerofier_codeword
        )
        print("prove time:", time() - start)

        # verify
        verdict = stark.verify(proof, air, boundary, transition_zerofier_root)

        assert verdict == True, "valid stark proof fails to verify"
        print("success \\o/")

        print("verifying false claim ...")
        # verify false claim
        output_element_ = output_element + field.one()
        boundary_ = rp.boundary_constraints(output_element_)
        verdict = stark.verify(proof, air, boundary_, transition_zerofier_root)

        assert verdict == False, "invalid stark proof verifies"
        print("proof rejected! \\o/")

        # prove with false witness
        print(
            "attempting to prove with witness violating transition constraints (should not fail because using fast division) ..."
        )
        cycle = 1 + (int(os.urandom(1)[0]) % len(trace) - 1)
        register = int(os.urandom(1)[0]) % state_width
        error = field.sample(os.urandom(17))

        trace[cycle][register] = trace[cycle][register] + error

        proof = stark.prove(
            trace, air, boundary, transition_zerofier, transition_zerofier_codeword
        )

        print(" ... but verification should fail :D")
        verdict = stark.verify(proof, air, boundary, transition_zerofier_root)
        assert verdict == False, "STARK produced from false witness verifies :("
        print("proof rejected! \\o/")


def test_fast_stark_time():
    field = Field.main()
    expansion_factor = 4
    num_colinearity_checks = 2
    security_level = 2

    rp = RescuePrime(40)
    output_element = field.sample(bytes(b"0xdeadbeef"))

    for trial in range(0, 1):  # 20
        input_element = output_element
        print("running trial with input:", input_element.value)
        output_element = rp.hash(input_element)
        num_cycles = rp.N + 1
        state_width = rp.m

        stark = FastStark(
            field,
            expansion_factor,
            num_colinearity_checks,
            security_level,
            state_width,
            num_cycles,
        )
        (
            transition_zerofier,
            transition_zerofier_codeword,
            transition_zerofier_root,
        ) = stark.preprocess()

        # prove honestly
        print("honest proof generation ...")

        # prove
        trace = rp.trace(input_element)
        boundary = rp.boundary_constraints(output_element)

        round_constants1 = rp.round_constants_polynomials(
            stark.omicron, stark.omicron_domain_length
        )

        print("stark prove--------------------")
        start = time()
        proof = stark.prove(
            trace,
            round_constants1,
            rp.transition_constaints_f,
            boundary,
            transition_zerofier,
            transition_zerofier_codeword,
        )
        print("prove time:", time() - start)

        # verify
        print("\ncompute air")
        start = time()

        temp = []
        for poly_list in round_constants1:
            temp += [[MPolynomial.lift(poly, 0) for poly in poly_list]]
        air = rp.transition_constraints(temp)
        print("finished", time() - start)
        print("stark verify--------------------")
        start = time()
        verdict = stark.verify(proof, air, boundary, transition_zerofier_root)
        print("finished", time() - start)
        assert verdict == True, "valid stark proof fails to verify"
        print("success \\o/")


def test_rdd_fast_stark():
    field = Field.main()
    expansion_factor = 4
    num_colinearity_checks = 2
    security_level = 2

    rp = RescuePrime(1000)
    output_element = field.sample(bytes(b"0xdeadbeef"))

    for trial in range(0, 1):  # 20
        input_element = output_element
        print("running trial with input:", input_element.value)
        output_element = rp.hash(input_element)
        num_cycles = rp.N + 1
        state_width = rp.m

        stark = RddFastStark(
            field,
            expansion_factor,
            num_colinearity_checks,
            security_level,
            state_width,
            num_cycles,
            sc=sc,
        )
        (
            transition_zerofier,
            transition_zerofier_codeword,
            transition_zerofier_root,
        ) = stark.preprocess()

        # prove honestly
        print("honest proof generation ...")

        # prove
        trace = rp.trace(input_element)
        trace1 = []
        for i in range(stark.num_registers):
            trace1 += [sc.parallelize([(j, trace[j][i]) for j in range(len(trace))])]
        assert len(trace1) == stark.num_registers, "len(trace1) == stark.num_registers"
        assert trace1[0].take(1)[0][0] == 0
        trace = trace1
        air = rp.transition_constraints(stark.omicron, stark.omicron_domain_length)
        boundary = rp.boundary_constraints(output_element)
        proof = stark.prove(
            trace, air, boundary, transition_zerofier, transition_zerofier_codeword
        )

        # verify
        verdict = stark.verify(proof, air, boundary, transition_zerofier_root)

        assert verdict == True, "valid stark proof fails to verify"
        print("success \\o/")


# test_fast_stark()
test_fast_stark_time()
# test_rdd_fast_stark()

# sc.stop()
