from typing import Callable

import gpytorch
import torch


class GibbsKernel(gpytorch.kernels.Kernel):
    def __init__(
        self,
        lengthscale_fn: Callable[
            [torch.Tensor], torch.Tensor
        ] = lambda x: torch.ones_like(x[..., :1]),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.lengthscale_fn = lengthscale_fn

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ):

        if diag:
            x1_lengthscale = self.lengthscale_fn(x1)
            x2_lengthscale = self.lengthscale_fn(x2)
            lengthscale = (x1_lengthscale**2 + x2_lengthscale**2) ** 0.5
            const = ((2 * x1_lengthscale * x2_lengthscale) / lengthscale**2) ** 0.5

            x1_ = x1.div(lengthscale)
            x2_ = x2.div(lengthscale)
            return const * self.covar_dist(
                x1_,
                x2_,
                square_dist=True,
                diag=True,
                dist_postprocess_func=gpytorch.kernels.rbf_kernel.postprocess_rbf,
                postprocess=True,
                last_dim_is_batch=last_dim_is_batch,
                **params,
            )

        assert not last_dim_is_batch

        x1_ = x1[..., None, :]
        x2_ = x2[..., None, :, :]
        diff = x1_ - x2_

        diff.where(diff == 0, torch.as_tensor(1e-8))

        x1_lengthscale = self.lengthscale_fn(x1_)
        x2_lengthscale = self.lengthscale_fn(x2_)
        lengthscale2 = x1_lengthscale**2 + x2_lengthscale**2
        const = ((2 * x1_lengthscale * x2_lengthscale) / lengthscale2) ** 0.5

        covar = const * (-(diff.pow(2) / lengthscale2)).exp()
        return covar[..., 0]


def gibbs_switching_lengthscale_fn(
    x: torch.Tensor,
    changepoint: float,
    direction: bool,
    lengthscale_high: float = 4.0,
    lengthscale_low: float = 0.1,
) -> torch.Tensor:
    if direction:
        return torch.where(
            x < changepoint,
            torch.ones_like(x) * lengthscale_high,
            torch.ones_like(x) * lengthscale_low,
        )
    return torch.where(
        x > changepoint,
        torch.ones_like(x) * lengthscale_high,
        torch.ones_like(x) * lengthscale_low,
    )

"""
Arsen and Nihar

"""

def gibbs_two_sigmoid_lengthscale_fn(
    x: torch.Tensor,
    b1: float,
    b2: float,
    width: float,
    ell_min: float,
    ell_max: float,
) -> torch.Tensor:
    """
    Continuous lengthscale l(x) using two sigmoids.

    combined = (sigmoid((x-b1)/width) - sigmoid((x-b2)/width))  (in [0,1])
    l(x) = ell_max + (ell_min - ell_max) * combined

    x: (..., D) but we use the first dim -> (..., 1)
    returns: (..., 1)
    """
    t = x[..., :1].to(dtype=torch.float32)

    s1 = torch.sigmoid((t - b1) / width)
    s2 = torch.sigmoid((t - b2) / width)

    combined = (s1 - s2)
    return ell_max + (ell_min - ell_max) * combined