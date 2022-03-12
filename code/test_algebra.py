from base.algebra import Field


def test_field():
    f = Field.main()
    nth_root = f.primitive_nth_root(128)
    assert nth_root ^ 128 == f.one()

    n_of_whole_field = 1 << 119  # p = 1 + 11 * 37 * 2^119
    assert f.generator() ^ n_of_whole_field == f.one()
    assert f.generator() == f.primitive_nth_root(n_of_whole_field)


test_field()
