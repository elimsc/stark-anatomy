from pyspark import SparkContext
from base.merkle import Merkle
from os import urandom
from rdd.rdd_merkle import Merkle as Merkle1, merkle_root
from rdd.rdd_merkle import merkle_build, merkle_open, _merkle_build0


def test_merkle():
    n = 64

    leafs = [urandom(int(urandom(1)[0])) for i in range(n)]
    root = Merkle.commit_(leafs)

    # opening any leaf should work
    for i in range(n):
        path = Merkle.open_(i, leafs)
        assert Merkle.verify_(root, i, path, leafs[i])

    # opening non-leafs should not work
    for i in range(n):
        path = Merkle.open_(i, leafs)
        assert False == Merkle.verify_(root, i, path, urandom(51))

    # opening wrong leafs should not work
    for i in range(n):
        path = Merkle.open_(i, leafs)
        j = (i + 1 + (int(urandom(1)[0] % (n - 1)))) % n
        assert False == Merkle.verify_(root, i, path, leafs[j])

    # opening leafs with the wrong index should not work
    for i in range(n):
        path = Merkle.open_(i, leafs)
        j = (i + 1 + (int(urandom(1)[0] % (n - 1)))) % n
        assert False == Merkle.verify_(root, j, path, leafs[i])

    # opening leafs to a false root should not work
    for i in range(n):
        path = Merkle.open_(i, leafs)
        assert False == Merkle.verify_(urandom(32), i, path, leafs[i])

    # opening leafs with even one falsehood in the path should not work
    for i in range(n):
        path = Merkle.open_(i, leafs)
        for j in range(len(path)):
            fake_path = path[0:j] + [urandom(32)] + path[j + 1 :]
            assert False == Merkle.verify_(root, i, fake_path, leafs[i])

    # opening leafs to a different root should not work
    fake_root = Merkle.commit_([urandom(32) for i in range(n)])
    for i in range(n):
        path = Merkle.open_(i, leafs)
        assert False == Merkle.verify_(fake_root, i, path, leafs[i])


sc = SparkContext()


def test_rdd_merkle():
    n = 64

    data_array = [urandom(int(urandom(1)[0])) for i in range(n)]
    merkle1 = Merkle1(data_array)
    merkle2 = _merkle_build0([(i + n, data_array[i]) for i in range(n)])

    assert list(enumerate(merkle1.nodes))[1:] == merkle2
    rdd_arr = sc.parallelize(list(enumerate(data_array)))
    rdd_tree = merkle_build(rdd_arr, n)

    root = Merkle.commit(data_array)
    assert root == merkle1.root()
    assert root == merkle_root(rdd_tree)

    # opening any leaf should work
    for i in range(n):
        path = Merkle.open(i, data_array)
        path1 = merkle1.open(i)
        path2 = merkle_open(i, rdd_tree)
        assert path == path1
        assert path == path2
        assert Merkle.verify(root, i, path, data_array[i])
        assert Merkle1.verify(root, i, path, data_array[i])


# test_merkle()
test_rdd_merkle()
sc.stop()
