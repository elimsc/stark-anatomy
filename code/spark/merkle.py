from hashlib import blake2b
from typing import List


class Merkle:
    H = blake2b

    def __init__(self, data_array: List) -> None:
        assert (
            len(data_array) & (len(data_array) - 1) == 0
        ), "length must be power of two"
        leaves = [Merkle.H(bytes(da)).digest() for da in data_array]  # 哈希后的值列表

        tree_nodes = [] + leaves

        cur_layer = leaves
        prev_layer = []
        while len(cur_layer) > 1:
            i = 0
            while i < len(cur_layer):
                prev_layer.append(Merkle.H(cur_layer[i] + cur_layer[i + 1]).digest())
                i += 2
            tree_nodes = prev_layer + tree_nodes
            cur_layer = prev_layer
            prev_layer = []

        # 整个merkle树的节点为 self.nodes[1:]
        self.nodes = [0] + tree_nodes

    def root(self):
        return self.nodes[1]

    def leaves(self):
        n = len(self.nodes)
        return self.nodes[n // 2 :]

    def commit_(leafs):
        assert len(leafs) & (len(leafs) - 1) == 0, "length must be power of two"
        if len(leafs) == 1:
            return leafs[0]
        else:
            return Merkle.H(
                Merkle.commit_(leafs[: len(leafs) // 2])
                + Merkle.commit_(leafs[len(leafs) // 2 :])
            ).digest()

    def open_(index: int, leafs: List):
        assert len(leafs) & (len(leafs) - 1) == 0, "length must be power of two"
        assert 0 <= index and index < len(leafs), "cannot open invalid index"
        if len(leafs) == 2:
            return [leafs[1 - index]]
        elif index < (len(leafs) / 2):
            return Merkle.open_(index, leafs[: len(leafs) // 2]) + [
                Merkle.commit_(leafs[len(leafs) // 2 :])
            ]
        else:
            return Merkle.open_(index - len(leafs) // 2, leafs[len(leafs) // 2 :]) + [
                Merkle.commit_(leafs[: len(leafs) // 2])
            ]

    def open(self, index):
        return Merkle.open_(index, self.leaves())

    def verify_(root, index, path, leaf):
        assert 0 <= index and index < (1 << len(path)), "cannot verify invalid index"
        if len(path) == 1:
            if index == 0:
                return root == Merkle.H(leaf + path[0]).digest()
            else:
                return root == Merkle.H(path[0] + leaf).digest()
        else:
            if index % 2 == 0:
                return Merkle.verify_(
                    root, index >> 1, path[1:], Merkle.H(leaf + path[0]).digest()
                )
            else:
                return Merkle.verify_(
                    root, index >> 1, path[1:], Merkle.H(path[0] + leaf).digest()
                )

    def verify(root, index, path, data_element):
        return Merkle.verify_(root, index, path, Merkle.H(bytes(data_element)).digest())
