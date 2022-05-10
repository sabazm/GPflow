# Copyright 2018-2020 The GPflow Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from typing import Optional, Sequence, Tuple

import tensorflow as tf

from ...base import Parameter, TensorType
from ...experimental.check_shapes import check_shape as cs
from ...experimental.check_shapes import check_shapes, inherit_check_shapes
from ...utilities.ops import leading_tile, leading_transpose
from ..base import Combination, Kernel


class MultioutputKernel(Kernel):
    """
    Multi Output Kernel class.
    This kernel can represent correlation between outputs of different datapoints.
    Therefore, subclasses of Mok should implement `K` which returns:

    - [N..., P, N..., P] if full_output_cov = True
    - [P, N..., N...] if full_output_cov = False

    and `K_diag` returns:

    - [N..., P, P] if full_output_cov = True
    - [N..., P] if full_output_cov = False

    The `full_output_cov` argument holds whether the kernel should calculate
    the covariance between the outputs. In case there is no correlation but
    `full_output_cov` is set to True the covariance matrix will be filled with zeros
    until the appropriate size is reached.
    """

    @property
    @abc.abstractmethod
    def num_latent_gps(self) -> int:
        """The number of latent GPs in the multioutput kernel"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def latent_kernels(self) -> Tuple[Kernel, ...]:
        """The underlying kernels in the multioutput kernel"""
        raise NotImplementedError

    @abc.abstractmethod
    @check_shapes(
        "X: [N..., D]",
        "X2: [N2..., D]",
        "return: [N..., P, N2..., P] if full_output_cov",
        "return: [P, N1..., N2...] if not full_output_cov",
    )
    def K(
        self, X: TensorType, X2: Optional[TensorType] = None, full_output_cov: bool = True
    ) -> tf.Tensor:
        """
        Returns the correlation of f(X) and f(X2), where f(.) can be multi-dimensional.

        :param X: data matrix
        :param X2: data matrix
        :param full_output_cov: calculate correlation between outputs.
        :return: cov[f(X), f(X2)]
        """
        raise NotImplementedError

    @abc.abstractmethod
    @check_shapes(
        "X: [N..., D]",
        "return: [N..., P, P] if full_output_cov",
        "return: [N..., P] if not full_output_cov",
    )
    def K_diag(self, X: TensorType, full_output_cov: bool = True) -> tf.Tensor:
        """
        Returns the correlation of f(X) and f(X), where f(.) can be multi-dimensional.

        :param X: data matrix
        :param full_output_cov: calculate correlation between outputs.
        :return: var[f(X)]
        """
        raise NotImplementedError

    @check_shapes(
        "X: [N..., D]",
        "X2: [N2..., D]",
        "return: [N..., P, N2..., P] if full_cov and full_output_cov",
        "return: [P, N..., N2...] if full_cov and (not full_output_cov)",
        "return: [N..., P, P] if (not full_cov) and full_output_cov",
        "return: [N..., P] if (not full_cov) and (not full_output_cov)",
    )
    def __call__(
        self,
        X: TensorType,
        X2: Optional[TensorType] = None,
        *,
        full_cov: bool = False,
        full_output_cov: bool = True,
        presliced: bool = False,
    ) -> tf.Tensor:
        if not presliced:
            X, X2 = self.slice(X, X2)
        if not full_cov and X2 is not None:
            raise ValueError(
                "Ambiguous inputs: passing in `X2` is not compatible with `full_cov=False`."
            )
        if not full_cov:
            return self.K_diag(X, full_output_cov=full_output_cov)
        return self.K(X, X2, full_output_cov=full_output_cov)


class SharedIndependent(MultioutputKernel):
    """
    - Shared: we use the same kernel for each latent GP
    - Independent: Latents are uncorrelated a priori.

    .. warning::
       This class is created only for testing and comparison purposes.
       Use `gpflow.kernels` instead for more efficient code.
    """

    def __init__(self, kernel: Kernel, output_dim: int) -> None:
        super().__init__()
        self.kernel = kernel
        self.output_dim = output_dim

    @property
    def num_latent_gps(self) -> int:
        # In this case number of latent GPs (L) == output_dim (P)
        return self.output_dim

    @property
    def latent_kernels(self) -> Tuple[Kernel, ...]:
        """The underlying kernels in the multioutput kernel"""
        return (self.kernel,)

    @inherit_check_shapes
    def K(
        self, X: TensorType, X2: Optional[TensorType] = None, full_output_cov: bool = True
    ) -> tf.Tensor:
        K = cs(self.kernel.K(X, X2), "[N..., N2...]")
        X_rank = tf.rank(X) - 1
        X2_rank = 1 if X2 is None else (tf.rank(X2) - 1)
        if full_output_cov:
            Ks = cs(leading_tile(K[..., None], [self.output_dim]), "[N..., N2..., P]")
            Ks = cs(tf.linalg.diag(Ks), "[N..., N2..., P, P]")

            P_rank = tf.ones((), dtype=tf.int32)
            ranks = [X_rank, X2_rank, P_rank, P_rank]
            indices = []
            i = tf.zeros((), dtype=tf.int32)
            for rank in ranks:
                indices.append(tf.range(i, i + rank))
                i += rank
            X_indices, X2_indices, P_indices, P2_indices = indices
            perm = tf.concat([X_indices, P_indices, X2_indices, P2_indices], axis=0)
            return cs(tf.transpose(Ks, perm), "[N..., P, N2..., P]")
        else:
            multiples = tf.concat(
                [[self.output_dim], tf.ones((X_rank + X2_rank), dtype=tf.int32)], axis=0
            )
            return cs(tf.tile(K[None, ...], multiples), "[P, N..., N2...]")

    @inherit_check_shapes
    def K_diag(self, X: TensorType, full_output_cov: bool = True) -> tf.Tensor:
        K = cs(self.kernel.K_diag(X), "[N...]")
        Ks = cs(leading_tile(K[..., None], [self.output_dim]), "[N..., P]")
        if full_output_cov:
            return cs(tf.linalg.diag(Ks), "[N..., P, P]")
        else:
            return cs(Ks, "[N..., P]")


class SeparateIndependent(MultioutputKernel, Combination):
    """
    - Separate: we use different kernel for each output latent
    - Independent: Latents are uncorrelated a priori.
    """

    def __init__(self, kernels: Sequence[Kernel], name: Optional[str] = None) -> None:
        super().__init__(kernels=kernels, name=name)

    @property
    def num_latent_gps(self) -> int:
        return len(self.kernels)

    @property
    def latent_kernels(self) -> Tuple[Kernel, ...]:
        """The underlying kernels in the multioutput kernel"""
        return tuple(self.kernels)

    @inherit_check_shapes
    def K(
        self, X: TensorType, X2: Optional[TensorType] = None, full_output_cov: bool = True
    ) -> tf.Tensor:
        if full_output_cov:
            Kxxs = tf.stack([k.K(X, X2) for k in self.kernels], axis=2)  # [N, N2, P]
            return tf.transpose(tf.linalg.diag(Kxxs), [0, 2, 1, 3])  # [N, P, N2, P]
        else:
            return tf.stack([k.K(X, X2) for k in self.kernels], axis=0)  # [P, N, N2]

    @inherit_check_shapes
    def K_diag(self, X: TensorType, full_output_cov: bool = False) -> tf.Tensor:
        stacked = tf.stack([k.K_diag(X) for k in self.kernels], axis=1)  # [N, P]
        return tf.linalg.diag(stacked) if full_output_cov else stacked  # [N, P, P]  or  [N, P]


class IndependentLatent(MultioutputKernel):
    """
    Base class for multioutput kernels that are constructed from independent
    latent Gaussian processes.

    It should always be possible to specify inducing variables for such kernels
    that give a block-diagonal Kuu, which can be represented as a [L, M, M]
    tensor. A reasonable (but not optimal) inference procedure can be specified
    by placing the inducing points in the latent processes and simply computing
    Kuu [L, M, M] and Kuf [N, P, M, L] and using `fallback_independent_latent_
    conditional()`. This can be specified by using `Fallback{Separate|Shared}
    IndependentInducingVariables`.
    """

    @abc.abstractmethod
    @check_shapes(
        "X: [N, D]",
        "X2: [N2, D]",
        "return: [L, N, N2]",
    )
    def Kgg(self, X: TensorType, X2: TensorType) -> tf.Tensor:
        raise NotImplementedError


class LinearCoregionalization(IndependentLatent, Combination):
    """
    Linear mixing of the latent GPs to form the output.
    """

    @check_shapes(
        "W: [P, L]",
    )
    def __init__(self, kernels: Sequence[Kernel], W: TensorType, name: Optional[str] = None):
        Combination.__init__(self, kernels=kernels, name=name)
        self.W = Parameter(W)  # [P, L]

    @property
    def num_latent_gps(self) -> int:
        return self.W.shape[-1]  # type: ignore  # L

    @property
    def latent_kernels(self) -> Tuple[Kernel, ...]:
        """The underlying kernels in the multioutput kernel"""
        return tuple(self.kernels)

    @inherit_check_shapes
    def Kgg(self, X: TensorType, X2: TensorType) -> tf.Tensor:
        return tf.stack([k.K(X, X2) for k in self.kernels], axis=0)  # [L, N, N2]

    @inherit_check_shapes
    def K(
        self, X: TensorType, X2: Optional[TensorType] = None, full_output_cov: bool = True
    ) -> tf.Tensor:
        Kxx = self.Kgg(X, X2)  # [L, N, N2]
        KxxW = Kxx[None, :, :, :] * self.W[:, :, None, None]  # [P, L, N, N2]
        if full_output_cov:
            # return tf.einsum('lnm,kl,ql->nkmq', Kxx, self.W, self.W)
            WKxxW = tf.tensordot(self.W, KxxW, [[1], [1]])  # [P, P, N, N2]
            return tf.transpose(WKxxW, [2, 0, 3, 1])  # [N, P, N2, P]
        else:
            # return tf.einsum('lnm,kl,kl->knm', Kxx, self.W, self.W)
            return tf.reduce_sum(self.W[:, :, None, None] * KxxW, [1])  # [P, N, N2]

    @inherit_check_shapes
    def K_diag(self, X: TensorType, full_output_cov: bool = True) -> tf.Tensor:
        K = tf.stack([k.K_diag(X) for k in self.kernels], axis=1)  # [N, L]
        if full_output_cov:
            # Can currently not use einsum due to unknown shape from `tf.stack()`
            # return tf.einsum('nl,lk,lq->nkq', K, self.W, self.W)  # [N, P, P]
            Wt = tf.transpose(self.W)  # [L, P]
            return tf.reduce_sum(
                K[:, :, None, None] * Wt[None, :, :, None] * Wt[None, :, None, :], axis=1
            )  # [N, P, P]
        else:
            # return tf.einsum('nl,lk,lk->nkq', K, self.W, self.W)  # [N, P]
            return tf.linalg.matmul(
                K, self.W ** 2.0, transpose_b=True
            )  # [N, L]  *  [L, P]  ->  [N, P]
