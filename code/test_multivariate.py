from base.multivariate import *


def test_evaluate():
    field = Field.main()
    variables = MPolynomial.variables(4, field)
    zero = field.zero()
    one = field.one()
    two = FieldElement(2, field)
    five = FieldElement(5, field)

    mpoly1 = (
        MPolynomial.constant(one) * variables[0]
        + MPolynomial.constant(two) * variables[1]
        + MPolynomial.constant(five) * (variables[2] ^ 3)
    )
    mpoly2 = (
        MPolynomial.constant(one) * variables[0] * variables[3]
        + MPolynomial.constant(five) * (variables[3] ^ 3)
        + MPolynomial.constant(five)
    )

    mpoly3 = mpoly1 * mpoly2

    point = [zero, five, five, two]

    eval1 = mpoly1.evaluate(point)
    eval2 = mpoly2.evaluate(point)
    eval3 = mpoly3.evaluate(point)

    assert (
        eval1 * eval2 == eval3
    ), "multivariate polynomial multiplication does not commute with evaluation"
    assert eval1 + eval2 == (mpoly1 + mpoly2).evaluate(
        point
    ), "multivariate polynomial addition does not commute with evaluation"

    print("eval3:", eval3.value)
    print("multivariate evaluate test success \\o/")


def test_lift():
    field = Field.main()
    variables = MPolynomial.variables(4, field)
    zero = field.zero()
    one = field.one()
    two = FieldElement(2, field)
    five = FieldElement(5, field)

    upoly = Polynomial.interpolate_domain([zero, one, two], [two, five, five])
    mpoly = MPolynomial.lift(upoly, 3)

    assert upoly.evaluate(five) == mpoly.evaluate(
        [zero, zero, zero, five]
    ), "lifting univariate to multivariate failed"

    print("lifting univariate to multivariate polynomial success \\o/")


def test_evaluate_symbolic():
    field = Field.main()
    variables = MPolynomial.variables(3, field)
    zero = field.zero()
    one = field.one()
    three = FieldElement(3, field)
    # f(X,Y,Z) = 1 + X*Y*Z + 3*X^2*Y*Z
    mpoly1 = (
        MPolynomial.constant(one)
        + MPolynomial.constant(one) * variables[0] * variables[1] * variables[2]
        + MPolynomial.constant(three) * (variables[0] ^ 2) * variables[1] * variables[2]
    )
    assert mpoly1.dictionary == dict({(0, 0, 0): one, (1, 1, 1): one, (2, 1, 1): three})
    # X, X^2, X^3
    point = [
        Polynomial([zero, one]),
        Polynomial([zero, zero, one]),
        Polynomial([zero, zero, zero, one]),
    ]
    poly1 = mpoly1.evaluate_symbolic(point)  # 1 + X^6 + 3X^7
    assert poly1 == Polynomial([one, zero, zero, zero, zero, zero, one, three])


test_evaluate()
test_lift()
test_evaluate_symbolic()
