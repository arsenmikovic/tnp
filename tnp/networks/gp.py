import random
from abc import ABC
from functools import partial
from typing import Tuple

import gpytorch
import torch

from tnp.networks.kernels import (
    GibbsKernel,
    gibbs_switching_lengthscale_fn,
    gibbs_two_sigmoid_lengthscale_fn,
)



class RandomHyperparameterKernel(ABC, gpytorch.kernels.Kernel):
    def sample_hyperparameters(self):
        pass


class ScaleKernel(gpytorch.kernels.ScaleKernel, RandomHyperparameterKernel):
    def __init__(
        self, min_log10_outputscale: float, max_log10_outputscale: float, **kwargs
    ):
        super().__init__(**kwargs)
        self.min_log10_outputscale = min_log10_outputscale
        self.max_log10_outputscale = max_log10_outputscale

    def sample_hyperparameters(self):
        # Sample outputscale.
        log10_outputscale = (
            torch.rand(()) * (self.max_log10_outputscale - self.min_log10_outputscale)
            + self.min_log10_outputscale
        )

        outputscale = 10.0**log10_outputscale
        self.outputscale = outputscale

        # Sample base kernel hyperparameters.
        self.base_kernel.sample_hyperparameters()


class RBFKernel(gpytorch.kernels.RBFKernel, RandomHyperparameterKernel):
    def __init__(
        self, min_log10_lengthscale: float, max_log10_lengthscale: float, **kwargs
    ):
        super().__init__(**kwargs)
        self.min_log10_lengthscale = min_log10_lengthscale
        self.max_log10_lengthscale = max_log10_lengthscale

    def sample_hyperparameters(self):
        # Sample lengthscale.
        shape = self.ard_num_dims if self.ard_num_dims is not None else ()
        log10_lengthscale = (
            torch.rand(shape)
            * (self.max_log10_lengthscale - self.min_log10_lengthscale)
            + self.min_log10_lengthscale
        )

        lengthscale = 10.0**log10_lengthscale
        self.lengthscale = lengthscale


class MaternKernel(gpytorch.kernels.MaternKernel, RandomHyperparameterKernel):
    def __init__(
        self, min_log10_lengthscale: float, max_log10_lengthscale: float, **kwargs
    ):
        super().__init__(**kwargs)
        self.min_log10_lengthscale = min_log10_lengthscale
        self.max_log10_lengthscale = max_log10_lengthscale

    def sample_hyperparameters(self):
        # Sample lengthscale.
        shape = self.ard_num_dims if self.ard_num_dims is not None else ()
        log10_lengthscale = (
            torch.rand(shape)
            * (self.max_log10_lengthscale - self.min_log10_lengthscale)
            + self.min_log10_lengthscale
        )

        lengthscale = 10.0**log10_lengthscale
        self.lengthscale = lengthscale


class PeriodicKernel(gpytorch.kernels.PeriodicKernel, RandomHyperparameterKernel):
    def __init__(
        self,
        min_log10_lengthscale: float,
        max_log10_lengthscale: float,
        min_log10_period: float,
        max_log10_period: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_log10_lengthscale = min_log10_lengthscale
        self.max_log10_lengthscale = max_log10_lengthscale
        self.min_log10_period = min_log10_period
        self.max_log10_period = max_log10_period

    def sample_hyperparameters(self):
        # Sample lengthscale.
        shape = self.ard_num_dims if self.ard_num_dims is not None else ()
        log10_lengthscale = (
            torch.rand(shape)
            * (self.max_log10_lengthscale - self.min_log10_lengthscale)
            + self.min_log10_lengthscale
        )

        lengthscale = 10.0**log10_lengthscale
        self.lengthscale = lengthscale

        # Sample period.
        log10_period = (
            torch.rand(shape) * (self.max_log10_period - self.min_log10_period)
            + self.min_log10_period
        )

        period = 10.0**log10_period
        self.period_length = period


class CosineKernel(gpytorch.kernels.CosineKernel, RandomHyperparameterKernel):
    def __init__(self, min_log10_period: float, max_log10_period: float, **kwargs):
        super().__init__(**kwargs)
        self.min_log10_period = min_log10_period
        self.max_log10_period = max_log10_period

    def sample_hyperparameters(self):
        # Sample period.
        log10_period = (
            torch.rand(()) * (self.max_log10_period - self.min_log10_period)
            + self.min_log10_period
        )

        period = 10.0**log10_period
        self.period_length = period


class RandomGibbsKernel(GibbsKernel, RandomHyperparameterKernel):
    def __init__(
        self,
        changepoints: Tuple[float, ...],
        directions: Tuple[bool, ...] = (True, False),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.changepoints = tuple(changepoints)
        self.directions = tuple(directions)

    def sample_hyperparameters(self):
        # Sample changepoint.
        direction = random.choice(self.directions)
        changepoint = random.choice(self.changepoints)

        self.lengthscale_fn = partial(
            gibbs_switching_lengthscale_fn,
            changepoint=changepoint,
            direction=direction,
        )



"""
Arsen and Nihar

"""

class RandomContinuousGibbsKernel(GibbsKernel, RandomHyperparameterKernel):
    """
    Gibbs kernel with a continuous two-sigmoid window lengthscale.

    b1,b2 sampled uniformly from ranges (with b1<b2 enforced).
    width is a fixed float (from YAML).
    ell_min, ell_max sampled from provided candidate lists (from YAML).
    """

    def __init__(
        self,
        *,
        # sample uniformly
        b1_range: Tuple[float, float] = (-2.0, 0.0),
        b2_range: Tuple[float, float] = (0.0, 2.0),

        # fixed
        width: float = 0.1,

        # sample from discrete sets
        ell_min_values: Tuple[float, ...] = (0.1, 0.3, 1.0),
        ell_max_values: Tuple[float, ...] = (2.0, 4.0, 8.0),

        **kwargs,
    ):
        super().__init__(**kwargs)
        self.b1_range = tuple(b1_range)
        self.b2_range = tuple(b2_range)
        self.width = float(width)

        self.ell_min_values = tuple(ell_min_values)
        self.ell_max_values = tuple(ell_max_values)

        # placeholders for inspection
        self.register_buffer("b1", torch.tensor(0.0))
        self.register_buffer("b2", torch.tensor(0.0))
        self.register_buffer("ell_min", torch.tensor(0.0))
        self.register_buffer("ell_max", torch.tensor(0.0))
        
        self.lengthscale_fn = self._dynamic_lengthscale_fn

    @property
    def device(self):
        return self.b1.device
    
    def _sample_uniform(self, lo: float, hi: float) -> float:
        return float((lo + (hi - lo) * torch.rand(())).item())

    def _dynamic_lengthscale_fn(self, x):
        return gibbs_two_sigmoid_lengthscale_fn(
            x,
            b1=self.b1,
            b2=self.b2,
            width=self.width,
            ell_min=self.ell_min,
            ell_max=self.ell_max,
        )

    def sample_hyperparameters(self):
        # sample b1, b2 and enforce b1 < b2
        b1 = self._sample_uniform(*self.b1_range)
        b2 = self._sample_uniform(*self.b2_range)
        if b2 < b1:
            b1, b2 = b2, b1

        # sample ell_min/ell_max from discrete candidate lists
        ell_min = float(random.choice(self.ell_min_values))
        ell_max = float(random.choice(self.ell_max_values))
        if ell_max < ell_min:
            ell_min, ell_max = ell_max, ell_min

        # store for logging/inspection
        device = self.device
        self.b1.copy_(torch.tensor(b1, device=device))
        self.b2.copy_(torch.tensor(b2, device=device))
        self.ell_min.copy_(torch.tensor(ell_min, device=device))
        self.ell_max.copy_(torch.tensor(ell_max, device=device))

# kernel:
#   _target_: tnp.networks.gp.RandomContinuousGibbsKernel
#   b1_range: [-7.0, -5.0]
#   b2_range: [5.0, 7.0]
#   width: 0.1
#   ell_min_values: [0.1, 0.2, 0.5]
#   ell_max_values: [0.5, 1.0, 2.0]
