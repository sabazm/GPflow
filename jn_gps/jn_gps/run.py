from pathlib import Path
from typing import Tuple

import jax.numpy as np
import jax.scipy.linalg as sl
import jax.scipy.stats as ss
import matplotlib.pyplot as plt
from gpflow.base import AnyNDArray
from gpflow.experimental.check_shapes import check_shape as cs
from gpflow.experimental.check_shapes import check_shapes
from jax import grad, jit, random
from matplotlib.axes import Axes

from .v import Input, VModule, derived, multi_set, to_loss_function

OUT_DIR = Path(__file__).parent.parent


# TODO:
# * Parameter constraints
# * What other frameworks are there for learning in JAX?
# * Multiple inputs
# * Sparse GPs
# * Variational GPs
# * Multiple outputs
# * V error messages
# * V inheritance / interfaces
# * Allow check_shape in @derived


class MeanFunction(VModule):

    x: Input[AnyNDArray] = Input(shape="[n, 1]")

    @derived(shape="[n, 1]")
    def f(self) -> AnyNDArray:
        raise NotImplementedError


class ZeroMeanFunction(MeanFunction):
    @derived(shape="[n, 1]")
    def f(self) -> AnyNDArray:
        return np.zeros_like(self.x)


class PolynomialMeanFunction(MeanFunction):

    coeffs: Input[AnyNDArray] = Input(shape="[degree]")

    @derived(shape="[n, 1]")
    def f(self) -> AnyNDArray:
        return np.polyval(self.coeffs, self.x)


class CovarianceFunction(VModule):

    x1: Input[AnyNDArray] = Input(shape="[n1, 1]")
    x2: Input[AnyNDArray] = Input(shape="[n2, 1]")

    @derived(shape="[n1, n2]")
    def full(self) -> AnyNDArray:
        raise NotImplementedError


class RBF(CovarianceFunction):

    variance = Input(float)

    @derived(shape="[n1, n2]")
    def full(self) -> AnyNDArray:
        errs = self.x1[:, None, :] - self.x2[None, :, :]
        sum_sq_errs = np.sum(errs ** 2, axis=-1)
        return np.exp(sum_sq_errs / (-2 * self.variance))


class GP(VModule):

    jitter = Input(default=1e-6)

    mean_func = Input(MeanFunction)
    covariance_func = Input(CovarianceFunction)
    noise_var: Input[AnyNDArray] = Input(shape="[]")

    x_data: Input[AnyNDArray] = Input(shape="[n_data, 1]")
    y_data: Input[AnyNDArray] = Input(shape="[n_data, 1]")
    x_predict: Input[AnyNDArray] = Input(shape="[n_predict, 1]")

    @check_shapes(
        "x: [n_rows, ...]",
    )
    def n_rows(self, x: AnyNDArray) -> int:
        n_rows = x.shape[0]
        assert isinstance(n_rows, int)
        return n_rows

    @derived()
    def n_data(self) -> int:
        return self.n_rows(self.x_data)

    @derived()
    def n_predict(self) -> int:
        return self.n_rows(self.x_predict)

    @derived(shape="[n_data, 1]")
    def mu_f(self) -> AnyNDArray:
        return self.mean_func(x=self.x_data).f

    @derived(shape="[n_predict, 1]")
    def mu_x(self) -> AnyNDArray:
        return self.mean_func(x=self.x_predict).f

    @derived(shape="[n_data, n_data]")
    def K_ff(self) -> AnyNDArray:
        return self.covariance_func(x1=self.x_data, x2=self.x_data).full + (
            self.jitter + self.noise_var
        ) * np.eye(self.n_data)

    @derived(shape="[n_data, n_predict]")
    def K_fx(self) -> AnyNDArray:
        return self.covariance_func(x1=self.x_data, x2=self.x_predict).full

    @derived(shape="[n_predict, n_predict]")
    def K_xx(self) -> AnyNDArray:
        return self.covariance_func(
            x1=self.x_predict, x2=self.x_predict
        ).full + self.jitter * np.eye(self.n_predict)

    @derived()
    def K_ff_cho_factor(self) -> Tuple[AnyNDArray, bool]:
        return sl.cho_factor(self.K_ff)  # type: ignore

    @derived(shape="[n_predict, n_data]")
    def K_ff_inv_fx_T(self) -> AnyNDArray:
        K_ff_inv_fx = sl.cho_solve(self.K_ff_cho_factor, self.K_fx)
        return K_ff_inv_fx.T

    @derived(shape="[n_predict, 1]")
    def f_mean(self) -> AnyNDArray:
        return self.mu_x + self.K_ff_inv_fx_T @ (self.y_data - self.mu_f)

    @derived(shape="[n_predict, 1]")
    def y_mean(self) -> AnyNDArray:
        return self.f_mean

    @derived(shape="[n_predict, n_predict]")
    def f_covariance(self) -> AnyNDArray:
        return self.K_xx - self.K_ff_inv_fx_T @ self.K_fx

    @derived(shape="[n_predict, n_predict]")
    def y_covariance(self) -> AnyNDArray:
        return self.f_covariance + self.noise_var * np.eye(self.n_predict)

    @derived(shape="[]")
    def log_likelihood(self) -> AnyNDArray:
        return ss.multivariate_normal.logpdf(self.y_data[:, 0], self.mu_f[:, 0], self.K_ff)


@check_shapes()
def main() -> None:

    dtype = np.float64

    @check_shapes()
    def plot_model(model: GP, name: str) -> None:
        n_rows = 3
        n_columns = 1
        plot_width = n_columns * 6.0
        plot_height = n_rows * 4.0
        _fig, (sample_ax, f_ax, y_ax) = plt.subplots(
            nrows=n_rows, ncols=n_columns, figsize=(plot_width, plot_height)
        )

        plot_x = cs(np.linspace(0.0, 10.0, num=100, dtype=dtype)[:, None], "[n_plot, 1]")
        model = model(x_predict=plot_x)

        key = random.PRNGKey(20220506)
        key, *keys = random.split(key, num=5)
        for i, k in enumerate(keys):
            plot_y = cs(
                random.multivariate_normal(k, model.f_mean[:, 0], model.f_covariance)[:, None],
                "[n_plot, 1]",
            )
            sample_ax.plot(plot_x, plot_y, label=str(i))
        sample_ax.set_title("Samples")

        @check_shapes(
            "plot_mean: [n_plot, 1]",
            "plot_full_cov: [n_plot, n_plot]",
        )
        def plot_dist(
            ax: Axes, title: str, plot_mean: AnyNDArray, plot_full_cov: AnyNDArray
        ) -> None:
            plot_cov = cs(np.diag(plot_full_cov), "[n_plot]")
            plot_std = cs(np.sqrt(plot_cov), "[n_plot]")
            plot_lower = cs(plot_mean[:, 0] - plot_std, "[n_plot]")
            plot_upper = cs(plot_mean[:, 0] + plot_std, "[n_plot]")
            (mean_line,) = ax.plot(plot_x, plot_mean)
            color = mean_line.get_color()
            ax.fill_between(plot_x[:, 0], plot_lower, plot_upper, color=color, alpha=0.3)
            ax.scatter(model.x_data, model.y_data, color=color)
            ax.set_title(title)

        plot_dist(f_ax, "f", model.f_mean, model.f_covariance)
        plot_dist(y_ax, "y", model.y_mean, model.y_covariance)

        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{name}.png")
        plt.close()

    model = GP(
        mean_func=PolynomialMeanFunction(coeffs=np.array([1.0, 0.0, 0.0])),
        covariance_func=RBF(variance=1.0),
        noise_var=np.array(0.1),
        x_data=np.zeros((0, 1)),
        y_data=np.zeros((0, 1)),
    )
    plot_model(model, "prior")

    x1 = cs(np.array([[1.0], [2.0], [3.0]], dtype=dtype), "[n_data, 1]")
    y1 = cs(np.array([[0.0], [2.0], [1.0]], dtype=dtype), "[n_data, 1]")
    model_2 = model(x_data=x1, y_data=y1)
    plot_model(model_2, "posterior")

    loss, params = to_loss_function(
        model_2,
        [
            "mean_func.coeffs",
            "noise_var",
        ],
        "log_likelihood",
    )
    loss_grad = jit(grad(loss))

    for i in range(100):
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print(i, loss(params), params)
        param_grads = loss_grad(params)
        params = {k: np.maximum(v + 0.1 * param_grads[k], 1e-6) for k, v in params.items()}

    model_3 = multi_set(model_2, params)
    plot_model(model_3, "trained")


if __name__ == "__main__":
    main()
