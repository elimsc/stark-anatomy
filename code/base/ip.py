from hashlib import sha256
import pickle as pickle  # serialization


class ProofStream:
    def __init__(self):
        self.objects = []
        self.read_index = 0

    def push(self, obj):
        self.objects += [obj]

    def pull(self):
        assert self.read_index < len(
            self.objects
        ), "ProofStream: cannot pull object; queue empty."
        obj = self.objects[self.read_index]
        self.read_index += 1
        return obj

    def serialize(self):
        return pickle.dumps(self.objects)

    def prover_fiat_shamir(self, num_bytes=32):
        # shake_256
        # return sha256(self.serialize()).digest(num_bytes)
        return sha256(self.serialize()).digest()

    def verifier_fiat_shamir(self, num_bytes=32):
        return sha256(pickle.dumps(self.objects[: self.read_index])).digest()

    def deserialize(self, bb):
        ps = ProofStream()
        ps.objects = pickle.loads(bb)
        return ps
