from hashlib import sha256
import math

from pyspark import RDD, StorageLevel


def next_power_two(n):
    if n & (n - 1) == 0:
        return n
    return 1 << len(bin(n)[2:])


def merkle_build_temp(data_array: RDD) -> RDD:
    rdd_zero = data_array.context.parallelize([(0, 0)])
    n = data_array.count()
    assert n & (n - 1) == 0, "must power of two"
    depth = int(math.log2(n))
    cur_layer = data_array.map(
        lambda v: (v[0] + n, Merkle.H(bytes(v[1])).digest())
    ).persist()
    tree_nodes = cur_layer

    for i in range(depth):
        cur_layer_length = 2 ** (depth - i)

        prev_layer = (
            cur_layer.map(lambda v: ((v[0] - cur_layer_length) // 2, (v[0], v[1])))
            .reduceByKey(lambda x, y: (x[0] // 2, Merkle.H(x[1] + y[1]).digest()))
            .map(lambda v: v[1])
            .persist()
        )

        tree_nodes = prev_layer.union(tree_nodes)
        cur_layer = prev_layer
    tree_nodes = rdd_zero.union(tree_nodes)
    return tree_nodes.sortByKey()


def merkle_open(index, tree: RDD, num_nodes) -> list:  # tree: RDD[(index, value)]
    # num_nodes = tree.count()
    assert num_nodes & (num_nodes - 1) == 0, "must power of two"
    real_index = num_nodes // 2 + index
    path_indexes = []
    while real_index > 1:
        if real_index % 2 == 0:
            path_indexes += [real_index + 1]
        else:
            path_indexes += [real_index - 1]
        real_index = real_index // 2
    return (
        tree.filter(lambda x: x[0] in path_indexes).sortByKey(False).values().collect()
    )


def merkle_root(tree: RDD):
    return tree.take(2)[1][1]


def merkle_build(data_array: RDD, n) -> RDD:
    sc = data_array.context
    if n <= (1 << 15):
        return sc.parallelize(
            [(0, 0)]
            + _merkle_build0(data_array.map(lambda x: (x[0] + n, x[1])).collect())
        )

    # n = data_array.count()
    assert n & (n - 1) == 0, "must power of two"
    logn = int(math.log2(n))
    # r = data_array.getNumPartitions()  # r个分区
    r = sc.defaultParallelism
    r = next_power_two(r)
    if 4 * r < n:
        r = 2 * r
    partition_size = n // r

    subtree = (
        data_array.map(lambda x: (x[0] // partition_size, (x[0] + n, x[1])))
        .groupByKey(r)
        .persist(StorageLevel.MEMORY_AND_DISK)
        .mapValues(list)
        .map(lambda x: (x[0], _merkle_build0(x[1])))
        .flatMap(lambda x: x[1])
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    # subtree = (
    #     data_array.map(lambda x: (x[0] + n, x[1]))
    #     .glom()
    #     .map(lambda x: _merkle_build0(x))
    #     .flatMap(lambda x: x)
    #     .persist(StorageLevel.MEMORY_AND_DISK)
    # )
    # print(r)
    # r = subtree.getNumPartitions()
    print(
        "rdd merkle_build: num_partition-----------------------------",
        r,
        subtree.getNumPartitions(),
    )

    subtree_root_index_range = list(range(r, 2 * r))

    top_tree_leaves = (
        subtree.filter(lambda x: x[0] in subtree_root_index_range).sortByKey().collect()
    )
    top_tree = _merkle_build1(top_tree_leaves)
    return (
        sc.parallelize([(0, 0)] + top_tree)
        .union(subtree)
        .sortByKey()
        .persist(StorageLevel.MEMORY_AND_DISK)
    )


def _merkle_build0(data_array: list):  # list[(index, value)]
    leaves = [(da[0], Merkle.H(bytes(da[1])).digest()) for da in data_array]
    return _merkle_build1(leaves) + leaves


def _merkle_build1(leaves: list):  # list[(index, value)], 返回结果不包含叶子结点
    tree_nodes = []
    cur_layer = leaves
    prev_layer = []
    while len(cur_layer) > 1:
        i = 0
        while i < len(cur_layer):
            prev_layer.append(
                (
                    cur_layer[i][0] // 2,
                    Merkle.H(cur_layer[i][1] + cur_layer[i + 1][1]).digest(),
                )
            )
            i += 2
        tree_nodes = prev_layer + tree_nodes
        cur_layer = prev_layer
        prev_layer = []

    return tree_nodes


class Merkle:
    H = sha256

    def __init__(self, data_array: list) -> None:
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

    def open(self, index):
        num_nodes = len(self.nodes)
        real_index = index + num_nodes // 2
        path_indexes = []

        while real_index > 1:
            if real_index % 2 == 0:
                path_indexes += [real_index + 1]
            else:
                path_indexes += [real_index - 1]
            real_index = real_index // 2

        return [self.nodes[i] for i in path_indexes]

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
