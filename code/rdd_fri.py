from pyspark import RDD
from algebra import *
from merkle import *
from ip import *
from ntt import *
from binascii import hexlify, unhexlify
import math
from hashlib import blake2b

from univariate import *
from util import *
from rdd_merkle import merkle_build, merkle_open, merkle_root


class RddFri:
    def __init__(
        self,
        offset,
        omega,
        initial_domain_length,
        expansion_factor,
        num_colinearity_tests,
    ):
        self.offset = offset
        self.omega = omega
        self.domain_length = initial_domain_length
        self.field = omega.field
        self.expansion_factor = expansion_factor
        self.num_colinearity_tests = num_colinearity_tests

        assert self.num_rounds() >= 1, "cannot do FRI with less than one round"

    def last_layer_size(self):
        return max(
            self.expansion_factor, next_power_two(4 * self.num_colinearity_tests)
        )

    def num_rounds(self):  # 一般为第一个大于 4 * num_colinearity_tests 的2的幂
        codeword_length = self.domain_length
        num_rounds = 0
        # while (
        #     codeword_length > self.expansion_factor
        #     and 4 * self.num_colinearity_tests < codeword_length
        # ):
        while codeword_length >= self.last_layer_size():
            codeword_length /= 2
            num_rounds += 1
        return num_rounds

    def sample_index(byte_array, size):
        acc = 0
        for b in byte_array:
            acc = (acc << 8) ^ int(b)
        return acc % size

    def sample_indices(self, seed, size, reduced_size, number):
        # reduced_size 为最后一层的点的个数
        assert (
            number <= reduced_size
        ), f"cannot sample more indices than available in last codeword; requested: {number}, available: {reduced_size}"
        assert (
            number <= 2 * reduced_size
        ), "not enough entropy in indices wrt last codeword"

        indices = []
        reduced_indices = []
        counter = 0
        while len(indices) < number:
            index = RddFri.sample_index(blake2b(seed + bytes(counter)).digest(), size)
            reduced_index = index % reduced_size
            counter += 1
            if reduced_index not in reduced_indices:
                indices += [index]
                reduced_indices += [reduced_index]

        return indices

    def eval_domain(self):
        return [self.offset * (self.omega ^ i) for i in range(self.domain_length)]

    def commit(self, codeword: RDD, proof_stream: ProofStream, round_index=0):
        one = self.field.one()
        two = FieldElement(2, self.field)
        omega = self.omega
        offset = self.offset
        codewords = []
        merkle_trees = []

        codeword_length = self.domain_length

        # print("in fri.commit, first layer codeword length:", codeword_length)

        # for each round
        for r in range(self.num_rounds()):
            N = codeword_length

            # make sure omega has the right order
            assert (
                omega ^ (N - 1) == omega.inverse()
            ), "error in commit: omega does not have the right order!"

            # compute and send Merkle root
            tree = merkle_build(codeword)
            merkle_trees += [tree]
            root = merkle_root(tree)
            proof_stream.push(root)

            # prepare next round, but only if necessary
            if r == self.num_rounds() - 1:
                break

            # get challenge
            alpha = self.field.sample(proof_stream.prover_fiat_shamir())

            # collect codeword
            codewords += [codeword]

            # split and fold
            halfN = N // 2
            codeword = (
                codeword.map(lambda x: (x[0] % halfN, x[1]))
                .groupByKey()
                .mapValues(list)
                .map(
                    lambda x: (
                        x[0],
                        two.inverse()
                        * (
                            (one + alpha / (offset * (omega ^ x[0]))) * x[1][0]
                            + (one - alpha / (offset * (omega ^ x[0]))) * x[1][1]
                        ),
                    )
                )
            ).sortByKey()
            # codeword = [
            #     two.inverse()
            #     * (
            #         (one + alpha / (offset * (omega ^ i))) * codeword[i]
            #         + (one - alpha / (offset * (omega ^ i))) * codeword[N // 2 + i]
            #     )
            #     for i in range(N // 2)
            # ]

            omega = omega ^ 2
            offset = offset ^ 2
            codeword_length = codeword_length // 2

        # send last codeword
        # print("in fri.commit, last layer codeword length:", len(codeword))
        proof_stream.push(codeword.values().collect())

        # collect last codeword too
        codewords = codewords + [codeword]

        return codewords, merkle_trees

    def query(
        self,
        current_codeword_length,
        current_codeword: RDD,
        next_codeword: RDD,
        cur_merkle_tree: RDD,
        next_merkle_tree: RDD,
        c_indices: list,
        proof_stream,
    ):
        # infer a and b indices
        a_indices = [index for index in c_indices]
        b_indices = [index + current_codeword_length // 2 for index in c_indices]

        def filter1(x):
            return (x in a_indices) or (x in b_indices)

        def filter2(x):
            return x in c_indices

        map1 = current_codeword.filter(lambda x: filter1(x[0])).collectAsMap()
        map2 = next_codeword.filter(lambda x: filter2(x[0])).collectAsMap()

        for s in range(self.num_colinearity_tests):
            proof_stream.push(
                (
                    map1[a_indices[s]],
                    map1[b_indices[s]],
                    map2[c_indices[s]],
                )
            )

        # reveal authentication paths
        for s in range(self.num_colinearity_tests):
            path_a = merkle_open(a_indices[s], cur_merkle_tree)
            path_b = merkle_open(b_indices[s], cur_merkle_tree)
            path_c = merkle_open(c_indices[s], next_merkle_tree)
            proof_stream.push(path_a)
            proof_stream.push(path_b)
            proof_stream.push(path_c)

        return a_indices + b_indices

    def prove(self, codeword: RDD, proof_stream: ProofStream):
        # commit phase
        # print("fri commit phase")
        codewords, merkle_trees = self.commit(codeword, proof_stream)

        # get indices
        top_level_indices = self.sample_indices(
            proof_stream.prover_fiat_shamir(),
            self.domain_length // 2,
            self.last_layer_size(),
            self.num_colinearity_tests,
        )
        indices = [index for index in top_level_indices]

        # query phase
        # print("fri query phase")
        for i in range(len(codewords) - 1):
            cur_codeword_length = self.domain_length // (2**i)
            indices = [index % (cur_codeword_length // 2) for index in indices]  # fold
            self.query(
                cur_codeword_length,
                codewords[i],
                codewords[i + 1],
                merkle_trees[i],
                merkle_trees[i + 1],
                indices,
                proof_stream,
            )

        return top_level_indices

    def verify(self, proof_stream, polynomial_values):
        omega = self.omega
        offset = self.offset

        # extract all roots and alphas
        roots = []
        alphas = []
        for r in range(self.num_rounds()):
            roots += [proof_stream.pull()]
            alphas += [self.field.sample(proof_stream.verifier_fiat_shamir())]

        # extract last codeword
        last_codeword = proof_stream.pull()

        # check if it matches the given root
        if roots[-1] != Merkle.commit(last_codeword):
            print("last codeword is not well formed")
            return False

        # check if it is low degree
        degree = (len(last_codeword) // self.expansion_factor) - 1
        last_omega = omega
        last_offset = offset
        for r in range(self.num_rounds() - 1):
            last_omega = last_omega ^ 2
            last_offset = last_offset ^ 2

        # assert that last_omega has the right order
        assert last_omega.inverse() == last_omega ^ (
            len(last_codeword) - 1
        ), "omega does not have right order"

        # compute interpolant
        last_domain = [
            last_offset * (last_omega ^ i) for i in range(len(last_codeword))
        ]
        # poly = Polynomial.interpolate_domain(last_domain, last_codeword)
        coefficients = intt(last_omega, last_codeword)
        poly = Polynomial(coefficients).scale(last_offset.inverse())

        # verify by  evaluating
        assert (
            poly.evaluate_domain(last_domain) == last_codeword
        ), "re-evaluated codeword does not match original!"
        if poly.degree() > degree:
            print(
                "last codeword does not correspond to polynomial of low enough degree"
            )
            print("observed degree:", poly.degree())
            print("but should be:", degree)
            return False

        # get indices
        top_level_indices = self.sample_indices(
            proof_stream.verifier_fiat_shamir(),
            self.domain_length >> 1,
            self.domain_length >> (self.num_rounds() - 1),
            self.num_colinearity_tests,
        )

        # for every round, check consistency of subsequent layers
        for r in range(0, self.num_rounds() - 1):

            # fold c indices
            c_indices = [
                index % (self.domain_length >> (r + 1)) for index in top_level_indices
            ]

            # infer a and b indices
            a_indices = [index for index in c_indices]
            b_indices = [index + (self.domain_length >> (r + 1)) for index in a_indices]

            # read values and check colinearity
            aa = []
            bb = []
            cc = []
            for s in range(self.num_colinearity_tests):
                (ay, by, cy) = proof_stream.pull()
                aa += [ay]
                bb += [by]
                cc += [cy]

                # record top-layer values for later verification
                if r == 0:
                    polynomial_values += [(a_indices[s], ay), (b_indices[s], by)]

                # colinearity check
                ax = offset * (omega ^ a_indices[s])
                bx = offset * (omega ^ b_indices[s])
                cx = alphas[r]
                if test_colinearity([(ax, ay), (bx, by), (cx, cy)]) == False:
                    print("colinearity check failure")
                    return False

            # verify authentication paths
            for i in range(self.num_colinearity_tests):
                path = proof_stream.pull()
                if Merkle.verify(roots[r], a_indices[i], path, aa[i]) == False:
                    print("merkle authentication path verification fails for aa")
                    return False
                path = proof_stream.pull()
                if Merkle.verify(roots[r], b_indices[i], path, bb[i]) == False:
                    print("merkle authentication path verification fails for bb")
                    return False
                path = proof_stream.pull()
                if Merkle.verify(roots[r + 1], c_indices[i], path, cc[i]) == False:
                    print("merkle authentication path verification fails for cc")
                    return False

            # square omega and offset to prepare for next round
            omega = omega ^ 2
            offset = offset ^ 2

        # all checks passed
        return True
