from hashlib import sha256
from base.fri import *
from base.univariate import *
from base.multivariate import *
from base.ntt import *
from functools import reduce
from rdd.rdd_merkle import Merkle as Merkle1, merkle_root
import os
from time import time

from rescue.rescue_prime import RescuePrime


def next_power_two(n):
    if n & (n - 1) == 0:
        return n
    return 1 << len(bin(n)[2:])


class FastStark:
    def __init__(
        self,
        field,
        expansion_factor,
        num_colinearity_checks,
        security_level,
        num_registers,
        num_cycles,
        transition_constraints_degree=3,
    ):
        assert (
            len(bin(field.p)) - 2 >= security_level
        ), "p must have at least as many bits as security level"
        assert (
            expansion_factor & (expansion_factor - 1) == 0
        ), "expansion factor must be a power of 2"
        assert expansion_factor >= 4, "expansion factor must be 4 or greater"
        assert (
            num_colinearity_checks * 2 >= security_level
        ), "number of colinearity checks must be at least half of security level"

        self.field = field
        self.lde_expansion_factor = expansion_factor
        self.num_colinearity_checks = num_colinearity_checks
        self.security_level = security_level

        self.num_registers = num_registers
        self.original_trace_length = num_cycles

        # self.num_randomizers >= 4 * num_colinearity_checks
        self.num_randomizers = (
            next_power_two(self.original_trace_length + 4 * num_colinearity_checks)
            - self.original_trace_length
        )

        self.randomized_trace_length = self.original_trace_length + self.num_randomizers

        self.omicron_domain_length = self.randomized_trace_length
        self.ce_domain_length = next_power_two(
            self.randomized_trace_length * transition_constraints_degree
        )
        self.ce_root = self.field.primitive_nth_root(self.ce_domain_length)

        self.ce_expansion_factor = self.ce_domain_length // self.omicron_domain_length
        self.fri_domain_length = self.ce_domain_length * expansion_factor

        self.generator = self.field.generator()
        self.omega = self.field.primitive_nth_root(self.fri_domain_length)
        self.omicron = self.field.primitive_nth_root(self.omicron_domain_length)
        self.omicron_domain = [
            self.omicron ^ i for i in range(self.omicron_domain_length)
        ]

        self.expansion_factor = self.ce_expansion_factor * self.lde_expansion_factor

        self.fri = Fri(
            self.generator,
            self.omega,
            self.fri_domain_length,
            self.lde_expansion_factor,
            self.num_colinearity_checks,
        )

    def preprocess(self):
        print("compute transition_zerofier")
        start = time()
        # transition_zerofier = fast_zerofier(
        #     self.omicron_domain[: (self.original_trace_length - 1)],
        #     self.omicron,
        #     self.omicron_domain_length,
        # )
        # (x-1)(x-g)...(x-g^i) = x^n / (x-g^n-1)(x-g^n-2)...(x-g^i+1)
        transition_zerofier1 = fast_zerofier(
            self.omicron_domain[self.original_trace_length - 1 :],
            self.omicron,
            self.omicron_domain_length,
        )
        zero = self.field.zero()
        one = self.field.one()
        transition_zerofier = fast_coset_divide(
            Polynomial([-one] + [zero] * (self.omicron_domain_length - 1) + [one]),
            transition_zerofier1,
            self.omega,
            self.ce_root,
            self.ce_domain_length,
        )

        print(time() - start)

        print("compute transition_zerofier_codeword")
        transition_zerofier_codeword = fast_coset_evaluate(
            transition_zerofier, self.generator, self.omega, self.fri.domain_length
        )
        print(time() - start)

        print("compute transition_zerofier_tree")
        transition_zerofier_tree = Merkle1(transition_zerofier_codeword)
        print(time() - start)

        transition_zerofier_root = transition_zerofier_tree.root()
        return (
            transition_zerofier,
            transition_zerofier_codeword,
            transition_zerofier_tree,
            transition_zerofier_root,
        )

    def transition_degree_bounds(self, transition_constraints):
        point_degrees = [1] + [
            self.original_trace_length + self.num_randomizers - 1
        ] * 2 * self.num_registers
        return [
            max(
                sum(
                    r * l for r, l in zip(point_degrees, k)
                )  # randomlized_poly_degree * transition_degree
                for k, v in a.dictionary.items()
            )
            for a in transition_constraints
        ]

    def transition_quotient_degree_bounds(self, transition_constraints):
        return [
            d - (self.original_trace_length - 1)
            for d in self.transition_degree_bounds(transition_constraints)
        ]

    # 组合多项式的阶
    def max_degree(self, transition_constraints):
        md = max(self.transition_quotient_degree_bounds(transition_constraints))
        return (1 << (len(bin(md)[2:]))) - 1

    def boundary_zerofiers(self, boundary):
        zerofiers = []
        for s in range(self.num_registers):
            points = [self.omicron ^ c for c, r, v in boundary if r == s]
            zerofiers = zerofiers + [Polynomial.zerofier_domain(points)]
        return zerofiers

    def boundary_interpolants(self, boundary):
        interpolants = []
        for s in range(self.num_registers):
            points = [(c, v) for c, r, v in boundary if r == s]
            domain = [self.omicron ^ c for c, v in points]
            values = [v for c, v in points]
            interpolants = interpolants + [
                Polynomial.interpolate_domain(domain, values)
            ]
        return interpolants

    def boundary_quotient_degree_bounds(self, randomized_trace_length, boundary):
        randomized_trace_degree = randomized_trace_length - 1
        return [
            randomized_trace_degree - bz.degree()
            for bz in self.boundary_zerofiers(boundary)
        ]

    def sample_weights(self, number, randomness):
        return [
            self.field.sample(sha256(randomness + bytes(i)).digest())
            for i in range(0, number)
        ]

    def prove(
        self,
        trace,
        round_constants_polys,
        transition_constraints,
        boundary,
        # transition_zerofier,
        # transition_zerofier_codeword,
        # transition_zerofier_tree,
        proof_stream=None,
    ):
        # def get_transition_polynomials(cur_state, next_state):
        #     return transition_constraints(
        #         cur_state, next_state, round_constants_polys, lambda x: Polynomial([x])
        #     )
        def get_transition_polynomials(cur_state, next_state):
            return transition_constraints(
                cur_state,
                next_state,
                round_constants_polys,
                self.ce_root,
                self.ce_domain_length,
            )

        # create proof stream object if necessary
        if proof_stream == None:
            proof_stream = ProofStream()

        # assert self.max_degree(transition_constraints) + 1 == self.omicron_domain_length

        # concatenate randomizers
        for k in range(self.num_randomizers):
            trace = trace + [
                [self.field.sample(os.urandom(17)) for s in range(self.num_registers)]
            ]

        trace_length = self.randomized_trace_length

        # trace 旋转
        trace1 = []
        for s in range(self.num_registers):
            trace1 += [[trace[c][s] for c in range(trace_length)]]
        trace = []  # trace不再被需要

        print(
            "trace_length",
            trace_length,
            "ce_domain_length",
            self.ce_domain_length,
            "fri_domain_length",
            self.fri_domain_length,
        )

        (
            transition_zerofier,
            transition_zerofier_codeword,
            transition_zerofier_tree,
            transition_zerofier_root,
        ) = self.preprocess()

        proof_stream.push(transition_zerofier_root)

        # interpolate
        print("interpolate trace_polynomials")
        start = time()
        trace_polynomials = []
        for s in range(self.num_registers):
            single_trace = trace1[s]
            trace_polynomials = trace_polynomials + [
                Polynomial(intt(self.omicron, single_trace))
            ]
        print("interpolate trace_polynomials finished", time() - start)

        # subtract boundary interpolants and divide out boundary zerofiers
        print("subtract boundary interpolants and divide out boundary zerofiers")
        start = time()
        boundary_quotients = []
        boundary_interpolants = self.boundary_interpolants(boundary)
        boundary_zerofiers = self.boundary_zerofiers(boundary)
        for s in range(self.num_registers):
            interpolant = boundary_interpolants[s]
            zerofier = boundary_zerofiers[s]
            quotient = fast_coset_divide(
                trace_polynomials[s] - interpolant,
                zerofier,
                self.generator,
                self.omicron,
                self.omicron_domain_length,
            )
            # quotient1 = (trace_polynomials[s] - interpolant) / zerofier
            # assert quotient == quotient1
            boundary_quotients += [quotient]
        print(
            "subtract boundary interpolants and divide out boundary zerofiers finished",
            time() - start,
        )

        # commit to boundary quotients
        print("commit to boundary quotients")
        start = time()
        boundary_quotient_codewords = []
        boundary_quotient_trees = []
        for s in range(self.num_registers):
            boundary_quotient_codewords = boundary_quotient_codewords + [
                fast_coset_evaluate(
                    boundary_quotients[s],
                    self.generator,
                    self.omega,
                    self.fri_domain_length,
                )
            ]
            # merkle_root = Merkle.commit(boundary_quotient_codewords[s])
            boundary_quotient_trees += [Merkle1(boundary_quotient_codewords[s])]
            boundary_quotient_root = boundary_quotient_trees[s].root()
            proof_stream.push(boundary_quotient_root)
        print("commit to boundary quotients finished", time() - start)

        # symbolically evaluate transition constraints
        print("get transition_polynomials from transition constraints")
        start = time()
        # point = (
        #     [Polynomial([self.field.zero(), self.field.one()])]
        #     + trace_polynomials
        #     + [tp.scale(self.omicron) for tp in trace_polynomials]
        # )
        # transition_polynomials = [
        #     a.evaluate_symbolic(point) for a in transition_constraints
        # ]
        transition_polynomials = get_transition_polynomials(
            trace_polynomials, [tp.scale(self.omicron) for tp in trace_polynomials]
        )

        print(
            "get transition_polynomials from transition constraints finished",
            time() - start,
        )

        # divide out zerofier
        print("transition_polynomials divide out zerofier")
        start = time()
        transition_quotients = [
            fast_coset_divide(
                tp,
                transition_zerofier,
                self.generator,
                self.ce_root,
                self.ce_domain_length,
            )
            for tp in transition_polynomials
        ]
        # transition_quotients1 = [
        #     tp / transition_zerofier for tp in transition_polynomials
        # ]
        # assert transition_quotients == transition_quotients1
        print("transition_polynomials divide out zerofier finished", time() - start)

        # commit to randomizer polynomial
        print("commit to randomizer polynomial")
        start = time()
        randomizer_polynomial = Polynomial(
            [self.field.sample(os.urandom(17)) for i in range(self.ce_domain_length)]
        )
        randomizer_codeword = fast_coset_evaluate(
            randomizer_polynomial, self.generator, self.omega, self.fri_domain_length
        )
        randomizer_tree = Merkle1(randomizer_codeword)
        # randomizer_root = Merkle.commit(randomizer_codeword)
        randomizer_root = randomizer_tree.root()
        proof_stream.push(randomizer_root)
        print("commit to randomizer polynomial finished", time() - start)

        # print("transition_quotients committed")

        # get weights for nonlinear combination
        #  - 1 randomizer
        #  - 2 for every transition quotient
        #  - 2 for every boundary quotient
        weights = self.sample_weights(
            1 + 2 * len(transition_quotients) + 2 * len(boundary_quotients),
            proof_stream.prover_fiat_shamir(),
        )

        # assert [
        #     tq.degree() for tq in transition_quotients
        # ] == self.transition_quotient_degree_bounds(
        #     transition_constraints
        # ), "transition quotient degrees do not match with expectation"

        # compute terms of nonlinear combination polynomial
        x = Polynomial([self.field.zero(), self.field.one()])
        max_degree = self.ce_domain_length - 1
        terms = []
        terms += [randomizer_polynomial]
        self.transition_quotients_degree = []
        for i in range(len(transition_quotients)):
            terms += [transition_quotients[i]]
            self.transition_quotients_degree += [transition_quotients[i].degree()]
            shift = max_degree - self.transition_quotients_degree[i]
            # 多项式的最大阶都变为 max_degree
            terms += [(x ^ shift) * transition_quotients[i]]
        self.boundary_quotients_degree = []
        for i in range(self.num_registers):
            terms += [boundary_quotients[i]]
            self.boundary_quotients_degree += [boundary_quotients[i].degree()]
            shift = max_degree - self.boundary_quotients_degree[i]
            # 多项式的最大阶都变为 max_degree
            terms += [(x ^ shift) * boundary_quotients[i]]

        # take weighted sum
        # combination = sum(weights[i] * terms[i] for all i)
        print("compute combination polynomial")
        start = time()
        combination = reduce(
            lambda a, b: a + b,
            [Polynomial([weights[i]]) * terms[i] for i in range(len(terms))],
            Polynomial([]),
        )
        print("compute combination polynomial finished", time() - start)

        # compute matching codeword
        print("compute combined_codeword")
        start = time()
        combined_codeword = fast_coset_evaluate(
            combination, self.generator, self.omega, self.fri_domain_length
        )
        print("compute combined_codeword finished", time() - start)

        # prove low degree of combination polynomial, and collect indices
        print("prove low degree of combination polynomial, and collect indices")
        start = time()
        indices = self.fri.prove(combined_codeword, proof_stream)
        print(
            "prove low degree of combination polynomial, and collect indices finished",
            time() - start,
        )

        # process indices
        duplicated_indices = [i for i in indices] + [
            (i + self.expansion_factor) % self.fri.domain_length for i in indices
        ]
        quadrupled_indices = [i for i in duplicated_indices] + [
            (i + (self.fri.domain_length // 2)) % self.fri.domain_length
            for i in duplicated_indices
        ]
        assert 4 * self.num_colinearity_checks == len(quadrupled_indices)
        quadrupled_indices.sort()

        # open indicated positions in the boundary quotient codewords
        print("open indicated positions in the boundary quotient codewords")
        start = time()
        for s in range(len(boundary_quotient_codewords)):
            for i in quadrupled_indices:
                bqc = boundary_quotient_codewords[s]
                proof_stream.push(bqc[i])
                # path = Merkle.open(i, bqc)
                path = boundary_quotient_trees[s].open(i)
                proof_stream.push(path)
        print(
            "open indicated positions in the boundary quotient codewords finished",
            time() - start,
        )

        # ... as well as in the randomizer
        print("open indicated positions in the randomizer_codeword")
        start = time()
        for i in quadrupled_indices:
            proof_stream.push(randomizer_codeword[i])
            path = randomizer_tree.open(i)
            proof_stream.push(path)
        print(
            "open indicated positions in the randomizer_codeword finished",
            time() - start,
        )

        # ... and also in the zerofier!
        print("open indicated positions in the transition_zerofier_codeword")
        start = time()
        for i in quadrupled_indices:
            proof_stream.push(transition_zerofier_codeword[i])
            path = transition_zerofier_tree.open(i)
            proof_stream.push(path)
        print(
            "open indicated positions in the transition_zerofier_codeword finished",
            time() - start,
        )

        # indices = sorted(indices)
        # cur_index = indices[0]
        # x = self.generator * (self.omega ^ cur_index)
        # print(cur_index, x)
        # print([poly.evaluate(x) for poly in transition_polynomials])
        # print("cur_state", [poly.evaluate(x) for poly in trace_polynomials])
        # x1 = x * (self.omega ^ self.expansion_factor)
        # print("next_state", [poly.evaluate(x1) for poly in trace_polynomials])
        # print("terms", [poly.evaluate(x) for poly in terms])
        # print("cur transition zerofier", transition_zerofier.evaluate(x))
        # print("combination", combination.evaluate(x))

        # the final proof is just the serialized stream
        return proof_stream.serialize()

    def verify(
        self,
        proof,
        round_constants_polys,
        transition_constaints,
        boundary,
        proof_stream=None,
    ):
        def eval_transition_constraints(cur_index, point, cur_state, next_state):
            round_constants_vals = []
            for poly_list in round_constants_polys:
                round_constants_vals += [[poly.evaluate(point) for poly in poly_list]]
            return transition_constaints(
                cur_state, next_state, round_constants_vals, lambda x: x
            )

        # infer trace length from boundary conditions
        # original_trace_length = 1 + max(c for c, r, v in boundary)
        # randomized_trace_length = original_trace_length + self.num_randomizers

        # deserialize with right proof stream
        if proof_stream == None:
            proof_stream = ProofStream()
        proof_stream = proof_stream.deserialize(proof)
        transition_zerofier_root = proof_stream.pull()

        # get Merkle roots of boundary quotient codewords
        boundary_quotient_roots = []
        for s in range(self.num_registers):
            boundary_quotient_roots = boundary_quotient_roots + [proof_stream.pull()]

        # get Merkle root of randomizer polynomial
        randomizer_root = proof_stream.pull()

        # get weights for nonlinear combination
        weights = self.sample_weights(
            1 + 2 * self.num_registers + 2 * len(self.boundary_interpolants(boundary)),
            proof_stream.verifier_fiat_shamir(),
        )

        # verify low degree of combination polynomial
        polynomial_values = []
        verifier_accepts = self.fri.verify(proof_stream, polynomial_values)
        polynomial_values.sort(key=lambda iv: iv[0])
        if not verifier_accepts:
            return False

        indices = [i for i, v in polynomial_values]
        values = [v for i, v in polynomial_values]

        # read and verify leafs, which are elements of boundary quotient codewords
        duplicated_indices = [i for i in indices] + [
            (i + self.expansion_factor) % self.fri.domain_length for i in indices
        ]
        duplicated_indices.sort()
        leafs = []
        for r in range(len(boundary_quotient_roots)):
            leafs = leafs + [dict()]
            for i in duplicated_indices:
                leafs[r][i] = proof_stream.pull()
                path = proof_stream.pull()
                verifier_accepts = verifier_accepts and Merkle.verify(
                    boundary_quotient_roots[r], i, path, leafs[r][i]
                )
                if not verifier_accepts:
                    return False

        # read and verify randomizer leafs
        randomizer = dict()
        for i in duplicated_indices:
            randomizer[i] = proof_stream.pull()
            path = proof_stream.pull()
            verifier_accepts = verifier_accepts and Merkle.verify(
                randomizer_root, i, path, randomizer[i]
            )
            if not verifier_accepts:
                return False

        # read and verify transition zerofier leafs
        transition_zerofier = dict()
        for i in duplicated_indices:
            transition_zerofier[i] = proof_stream.pull()
            path = proof_stream.pull()
            verifier_accepts = verifier_accepts and Merkle.verify(
                transition_zerofier_root, i, path, transition_zerofier[i]
            )
            if not verifier_accepts:
                return False

        # verify leafs of combination polynomial
        for i in range(len(indices)):
            current_index = indices[i]  # do need i

            # get trace values by applying a correction to the boundary quotient values (which are the leafs)
            domain_current_index = self.generator * (self.omega ^ current_index)
            next_index = (
                current_index + self.expansion_factor
            ) % self.fri.domain_length
            domain_next_index = self.generator * (self.omega ^ next_index)
            current_trace = [self.field.zero() for s in range(self.num_registers)]
            next_trace = [self.field.zero() for s in range(self.num_registers)]

            boundary_zerofiers = self.boundary_zerofiers(boundary)
            boundary_interpolants = self.boundary_interpolants(boundary)
            for s in range(self.num_registers):
                zerofier = boundary_zerofiers[s]
                interpolant = boundary_interpolants[s]

                current_trace[s] = leafs[s][current_index] * zerofier.evaluate(
                    domain_current_index
                ) + interpolant.evaluate(domain_current_index)
                next_trace[s] = leafs[s][next_index] * zerofier.evaluate(
                    domain_next_index
                ) + interpolant.evaluate(domain_next_index)

            point = [domain_current_index] + current_trace + next_trace
            # transition_constraints 在对应 (X, current_trace, next_trace) 处的值
            # transition_constraints_values1 = [
            #     transition_constraints[s].evaluate(point)
            #     for s in range(len(transition_constraints))
            # ]
            transition_constraints_values = eval_transition_constraints(
                current_index, domain_current_index, current_trace, next_trace
            )

            # compute nonlinear combination
            counter = 0
            terms = []
            max_degree = self.ce_domain_length - 1
            terms += [randomizer[current_index]]
            for s in range(
                len(transition_constraints_values)
            ):  # 求阶对齐后的 transition_quotient
                tcv = transition_constraints_values[s]
                quotient = tcv / transition_zerofier[current_index]
                terms += [quotient]
                shift = max_degree - self.transition_quotients_degree[s]
                terms += [quotient * (domain_current_index ^ shift)]
            for s in range(self.num_registers):  # 求阶对齐后的 boundary quotient
                bqv = leafs[s][current_index]  # boundary quotient value
                terms += [bqv]
                shift = max_degree - self.boundary_quotients_degree[s]
                terms += [bqv * (domain_current_index ^ shift)]
            # 根据上面的randomizer值, transition_quotient值, boundary_quotient值, 求出组合多项式的值
            combination = reduce(
                lambda a, b: a + b,
                [terms[j] * weights[j] for j in range(len(terms))],
                self.field.zero(),
            )
            # print(current_index, next_index, domain_current_index)
            # print("transition_constraints_values", transition_constraints_values)
            # print("cur_state", point[1 : 1 + self.num_registers])
            # print("next_state", point[1 + self.num_registers :])
            # print("terms", terms)
            # print(values)
            # print(combination)

            # verify against combination polynomial value
            # 验证 自己求出来的组合多项式的值 和 fri第一层的merkle树上的值 是否相等
            verifier_accepts = verifier_accepts and (combination == values[i])
            if not verifier_accepts:
                return False

        return verifier_accepts
