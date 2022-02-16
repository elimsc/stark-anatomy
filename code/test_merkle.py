from merkle import Merkle
from os import urandom
from rdd_merkle import Merkle as Merkle1
from rdd_merkle import merkle_build, merkle_open


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


from pyspark import SparkContext, SparkConf


conf = (
    SparkConf().setAppName("test_fast_stark").setMaster("spark://zhdeMacBook-Pro:7077")
)
sc = SparkContext(conf=conf)

sc.addPyFile("./rdd_merkle.py")


def test_spark_merkle():
    n = 64

    data_array = [urandom(int(urandom(1)[0])) for i in range(n)]
    merkle1 = Merkle1(data_array)
    rdd_arr = sc.parallelize(list(enumerate(data_array)))
    rdd_tree = merkle_build(rdd_arr)

    root = Merkle.commit(data_array)
    assert root == merkle1.root()
    assert root == rdd_tree.take(2)[1][1]

    # opening any leaf should work
    for i in range(n):
        path = Merkle.open(i, data_array)
        path1 = merkle1.open(i)
        path2 = merkle_open(i, rdd_tree)
        assert path == path1
        assert path == path2
        assert Merkle.verify(root, i, path, data_array[i])
        assert Merkle1.verify(root, i, path, data_array[i])


test_merkle()
test_spark_merkle()
sc.stop()
