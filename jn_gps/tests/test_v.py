import jax.numpy as np
import numpy.testing as npt
import pytest
from gpflow.base import AnyNDArray
from gpflow.experimental.check_shapes.exceptions import ShapeMismatchError
from jax import grad, jit, random

from jn_gps.v import Input, VModule, derived, multi_get, multi_set, to_loss_function


def test_v() -> None:
    ab_count = 0
    abc_count = 0

    class Foo(VModule):
        a = Input(int)
        b = Input(int)
        c = Input(int)

        @derived()
        def ab(self) -> int:
            nonlocal ab_count
            ab_count += 1
            return self.a + self.b

        @derived()
        def abc(self) -> int:
            nonlocal abc_count
            abc_count += 1
            return self.ab + self.c

    foo = Foo()
    foo = foo(a=3, b=7)
    assert 3 == foo.a
    assert 7 == foo.b
    assert 3 + 7 == foo.ab
    assert 1 == ab_count

    old_foo = foo
    foo = foo(c=4)
    assert 3 + 7 == foo.ab
    assert 1 == ab_count
    assert 3 + 7 + 4 == foo.abc
    assert 1 == abc_count

    foo = foo(a=5)
    assert 5 + 7 == foo.ab
    assert 2 == ab_count
    assert 5 + 7 + 4 == foo.abc
    assert 2 == abc_count

    assert 3 + 7 == old_foo.ab
    assert 2 == ab_count


def test_v__nesting() -> None:
    class Bar(VModule):
        a = Input(int)
        b = Input(int)

        @derived()
        def s(self) -> int:
            return self.a + self.b

    class Foo(VModule):
        ab = Input(Bar)
        c = Input(int)

        @derived()
        def abc(self) -> Bar:
            return Bar(a=self.ab.s, b=self.c)

    foo = Foo()
    foo = foo(ab=Bar(a=3, b=7))
    assert 3 == foo.ab.a
    assert 7 == foo.ab.b
    assert 3 + 7 == foo.ab.s

    old_foo = foo
    foo = foo(c=4)
    assert 3 + 7 == foo.ab.s
    assert 3 + 7 + 4 == foo.abc.s

    foo = foo(ab=foo.ab(a=5))
    assert 5 + 7 == foo.ab.s
    assert 5 + 7 + 4 == foo.abc.s

    assert 3 + 7 == old_foo.ab.s


def test_v__shapes() -> None:
    class Foo(VModule):
        a: Input[AnyNDArray] = Input(shape="[1, m]")
        b: Input[AnyNDArray] = Input(shape="[n, 1]")
        c: Input[AnyNDArray] = Input(shape="[n, m]")

        @derived(shape="[n, m]")
        def ab(self) -> AnyNDArray:
            return self.a + self.b

    foo = Foo(a=np.zeros((1, 3)), b=np.ones((4, 1)))
    foo(c=np.ones((4, 3)))
    foo.ab  # pylint: disable=pointless-statement

    with pytest.raises(ShapeMismatchError):
        foo(c=np.ones((3, 4)))

    with pytest.raises(ShapeMismatchError):
        Foo(a=np.zeros((1, 3)), b=np.ones((4, 2)))


def test_v__multi() -> None:
    class Bar(VModule):
        a = Input(int)
        b = Input(int)

        @derived()
        def ab(self) -> int:
            return self.a + self.b

    class Foo(VModule):
        ab = Input(Bar)
        c = Input(int)

        @derived()
        def abc(self) -> Bar:
            return Bar(a=self.ab.ab, b=self.c)

    foo = Foo(ab=Bar(a=2, b=3), c=4)

    assert {
        "ab.a": 2,
    } == multi_get(foo, ["ab.a"])
    assert {
        "ab.b": 3,
    } == multi_get(foo, ["ab.b"])
    assert {
        "c": 4,
    } == multi_get(foo, ["c"])
    assert {
        "ab.a": 2,
        "ab.b": 3,
        "c": 4,
    } == multi_get(foo, ["ab.a", "ab.b", "c"])

    foo = multi_set(
        foo,
        {
            "ab.a": 7,
            "c": 8,
        },
    )

    assert {
        "ab.a": 7,
        "ab.b": 3,
        "c": 8,
    } == multi_get(foo, ["ab.a", "ab.b", "c"])


def test_jax_integration() -> None:
    class LinearModel(VModule):

        weights: Input[AnyNDArray] = Input(shape="[n_features]")
        features: Input[AnyNDArray] = Input(shape="[batch..., n_features]")

        @derived(shape="[batch...]")
        def prediction(self) -> AnyNDArray:
            return np.einsum("...i,i -> ...", self.features, self.weights)

    class Loss(VModule):

        model = Input(LinearModel)
        x: Input[AnyNDArray] = Input(shape="[batch..., n_features]")
        y: Input[AnyNDArray] = Input(shape="[batch...]")

        @derived()
        def prediction(self) -> AnyNDArray:
            return self.model(features=self.x)

        @derived(shape="[]")
        def loss(self) -> AnyNDArray:
            return np.mean((self.y - self.prediction.prediction) ** 2)

    target = np.array([0.3, 0.6])
    key = random.PRNGKey(20220506)
    x = random.normal(key, (10, 2))
    y = LinearModel(weights=target, features=x).prediction
    loss = Loss(model=LinearModel(weights=np.zeros((2,))), x=x, y=y)

    loss_fn, params = to_loss_function(loss, ["model.weights"], "loss")
    loss_grad = jit(grad(loss_fn))

    for _ in range(100):
        param_grads = loss_grad(params)
        params = {k: v - 0.1 * param_grads[k] for k, v in params.items()}

    loss = multi_set(loss, params)
    npt.assert_allclose(target, loss.model.weights, atol=1e-7)
