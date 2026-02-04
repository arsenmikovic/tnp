import random
from abc import ABC
from typing import Dict, Iterable, Optional, Tuple, Union

import einops
import gpytorch
import torch

from ..networks.gp import RandomHyperparameterKernel
from .base import GroundTruthPredictor
from .synthetic import SyntheticGeneratorUniformInput, SyntheticBatch


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        kernel: gpytorch.kernels.Kernel,
        train_inputs: Optional[torch.Tensor] = None,
        train_targets: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            train_inputs=train_inputs,
            train_targets=train_targets,
            likelihood=likelihood,
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(  # pylint: disable=arguments-differ
        self, x: torch.Tensor
    ) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPGroundTruthPredictor(GroundTruthPredictor):
    def __init__(
        self,
        kernel: gpytorch.kernels.Kernel,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
    ):
        self.kernel = kernel
        self.likelihood = likelihood

        self._result_cache: Optional[Dict[str, torch.Tensor]] = None

    def __call__(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
        yt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        # Move devices.
        old_device = xc.device
        device = self.kernel.device
        xc = xc.to(device)
        yc = yc.to(device)
        xt = xt.to(device)
        if yt is not None:
            yt = yt.to(device)

        if yt is not None and self._result_cache is not None:
            # Return cached results.
            return (
                self._result_cache["mean"],
                self._result_cache["std"],
                self._result_cache["gt_loglik"],
            )

        mean_list = []
        std_list = []
        gt_loglik_list = []

        # Compute posterior.
        for i, (xc_, yc_, xt_) in enumerate(zip(xc, yc, xt)):
            gp_model = GPRegressionModel(
                likelihood=self.likelihood,
                kernel=self.kernel,
                train_inputs=xc_,
                train_targets=yc_[..., 0],
            )
            gp_model = gp_model.to(device)
            gp_model.eval()
            gp_model.likelihood.eval()
            with torch.no_grad():

                dist = gp_model(xt_)
                pred_dist = gp_model.likelihood.marginal(dist)
                if yt is not None:
                    gt_loglik = pred_dist.to_data_independent_dist().log_prob(
                        yt[i, ..., 0]
                    )
                    gt_loglik_list.append(gt_loglik)

                mean_list.append(pred_dist.mean)
                try:
                    std_list.append(pred_dist.stddev)
                except RuntimeError:
                    std_list.append(pred_dist.covariance_matrix.diagonal() ** 0.5)

        mean = torch.stack(mean_list, dim=0)
        std = torch.stack(std_list, dim=0)
        gt_loglik = torch.stack(gt_loglik_list, dim=0) if gt_loglik_list else None

        # Cache for deterministic validation batches.
        # Note yt is not specified when passing x_plot.
        if yt is not None:
            self._result_cache = {
                "mean": mean,
                "std": std,
                "gt_loglik": gt_loglik,
            }

        # Move back.
        xc = xc.to(old_device)
        yc = yc.to(old_device)
        xt = xt.to(old_device)
        if yt is not None:
            yt = yt.to(old_device)

        mean = mean.to(old_device)
        std = std.to(old_device)
        if gt_loglik is not None:
            gt_loglik = gt_loglik.to(old_device)

        return mean, std, gt_loglik
        
    def sample_outputs(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:

        gp_model = GPRegressionModel(
            likelihood=self.likelihood,
            kernel=self.kernel,
        )
        gp_model.eval()
        gp_model.likelihood.eval()

        # Sample from prior.
        with torch.no_grad():
            dist = gp_model.forward(x)
            f = dist.sample(sample_shape=sample_shape)
            dist = gp_model.likelihood(f)
            y = dist.sample()
            return y[..., None]


class GPGenerator(ABC):
    def __init__(
        self,
        *,
        kernel: Union[
            RandomHyperparameterKernel,
            Tuple[RandomHyperparameterKernel, ...],
        ],
        noise_std: float,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.kernel = kernel
        if isinstance(self.kernel, Iterable):
            self.kernel = tuple(self.kernel)

        self.noise_std = noise_std

    def set_up_gp(self) -> GPGroundTruthPredictor:
        if isinstance(self.kernel, tuple):
            kernel = random.choice(self.kernel)
        else:
            kernel = self.kernel

        kernel = kernel()
        kernel.sample_hyperparameters()
        kernel = kernel.cpu()

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood = likelihood.cpu()
        likelihood.noise = self.noise_std**2.0

        return GPGroundTruthPredictor(kernel=kernel, likelihood=likelihood)

    def sample_outputs(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, GroundTruthPredictor]:
        gt_pred = self.set_up_gp()
        y = gt_pred.sample_outputs(x)
        return y, gt_pred


class RandomScaleGPGenerator(GPGenerator, SyntheticGeneratorUniformInput):
    pass


class RandomScaleGPGeneratorSameInputs(RandomScaleGPGenerator):

    def sample_inputs(
        self,
        nc: int,
        batch_shape: torch.Size,
        nt: Optional[int] = None,
    ) -> torch.Tensor:
        x = super().sample_inputs(nc=nc, batch_shape=torch.Size(), nt=nt)
        x = einops.repeat(x, "n d -> b n d", b=batch_shape[0])
        return x

    def sample_outputs(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        gt_pred = self.set_up_gp()
        sample_shape = x.shape[:-2]
        return gt_pred.sample_outputs(x[0], sample_shape=sample_shape), gt_pred


"""
Arsen and Nihar work bellow.

"""

class ReversedGPGroundTruthPredictor(GroundTruthPredictor):
    def __init__(
            self, 
            base_gt_pred: GPGroundTruthPredictor,
            reversal_point: float, 
            context_range: torch.Tensor, 
            **kwargs):
        super().__init__(**kwargs)
        self.base_gt_pred = base_gt_pred
        self.reversal_point = reversal_point
        self.context_range = context_range
        self.min_context = context_range[:, 0]
        self.max_context = context_range[:, 1]
    
    def __call__(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
        yt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        # Ensure boundary tensors are on the same device as inputs
        min_c = self.min_context.to(xc.device)
        max_c = self.max_context.to(xc.device)
        reversal_point = self.reversal_point

        def transform_inputs(x):
            # Create a boolean mask: True where elements are inside the original context range
            # Broadcasting handles shape (Batch, N, Dim) vs (Dim,)
            is_in_range = (x >= min_c) & (x <= max_c)
            
            # Calculate the reflection for ALL points: x' = 2 * r - x
            x_reflected = 2 * reversal_point - x
            
            # Select element-wise and flip inputs outside of context range
            return torch.where(is_in_range, x, x_reflected)
        
        # Apply transformation to both context and target inputs
        xc_transformed = transform_inputs(xc)
        xt_transformed = transform_inputs(xt)

        # Call the parent GPGroundTruthPredictor with the transformed inputs.
        return self.base_gt_pred.__call__(xc_transformed, yc, xt_transformed, yt)

    def sample_outputs(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        pass


class ReversedContextGPGenerator(RandomScaleGPGenerator):
    """
    Generates batches of GP data where the GP is reversed at a reversal_point
    and the context points have direct counterparts in the target points.
    """

    def __init__(
        self,
        *,
        min_nc: int,
        max_nc: int,
        min_nt: int,
        max_nt: int,
        batch_size: int,
        reversal_point: float = 0.0,
        **kwargs,
    ):
        super().__init__(min_nc=min_nc, max_nc=max_nc, min_nt = min_nt, 
                         max_nt = max_nt, batch_size=batch_size, **kwargs)
        self.reversal_point = reversal_point

    def generate_batch(self) -> SyntheticBatch:
        # Sample number of context = number of target points.
        nc = torch.randint(low=self.min_nc, high=self.max_nc + 1, size=())
        nt = nc

        # Sample batch using parent method
        batch = self.sample_batch(
            nc=nc,
            nt=nt,
            batch_shape=torch.Size([self.batch_size])
        )

        return batch
    
    def sample_batch(
        self,
        nc: int,
        nt: int,
        batch_shape: torch.Size,
    ) -> SyntheticBatch:
        
        # Randomly flip the context range around the reversal point
        if torch.rand(1) > 0.5:
            current_range = 2 * self.reversal_point - self.context_range
            current_range = current_range.flip(dims=[1])
        else:
            current_range = self.context_range

        # Sample context inputs
        xc = self.sample_inputs(
            nc=nc,
            context_range=current_range,
            batch_shape=batch_shape)
        yc, non_reversed_gt_pred = self.sample_outputs(x=xc)

        # Create target inputs by reversing context inputs around reversal_point
        xt = 2 * self.reversal_point - xc
        xt = xt.flip(dims=[-2])
        yt = yc.flip(dims=[-2])

        x = torch.concat([xc, xt], axis=1)
        y = torch.concat([yc, yt], axis=1)

        reversed_gt_pred = ReversedGPGroundTruthPredictor(
            base_gt_pred=non_reversed_gt_pred,
            reversal_point=self.reversal_point,
            context_range=current_range
        )

        return SyntheticBatch(
            x=x,
            y=y,
            xc=xc,
            yc=yc,
            xt=xt,
            yt=yt,
            gt_pred=reversed_gt_pred,
        )
    
    def sample_inputs(
        self,
        nc: int,
        context_range: torch.Tensor,
        batch_shape: torch.Size,
    ) -> torch.Tensor:

        # Sample context inputs
        xc = (
            torch.rand((*batch_shape, nc, self.dim))
            * (context_range[:, 1] - context_range[:, 0])
            + context_range[:, 0]
        )

        return xc




"""
Arsen and Nihar reversal generalisation task.

"""

class IndependentReversedGPGroundTruthPredictor(GroundTruthPredictor):
    def __init__(
            self, 
            base_gt_pred: GPGroundTruthPredictor,
            reversal_point: float, 
            context_range: torch.Tensor, 
            **kwargs):
        super().__init__(**kwargs)
        self.base_gt_pred = base_gt_pred
        self.reversal_point = reversal_point
        self.context_range = context_range
        self.min_context = context_range[:, 0]
        self.max_context = context_range[:, 1]
    
    def __call__(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
        yt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        # Ensure boundary tensors are on the same device as inputs
        min_c = self.min_context.to(xc.device)
        max_c = self.max_context.to(xc.device)
        reversal_point = self.reversal_point

        def transform_inputs(x):
            # Create a boolean mask: True where elements are inside the original context range
            # Broadcasting handles shape (Batch, N, Dim) vs (Dim,)
            is_in_range = (x >= min_c) & (x <= max_c)
            
            # Calculate the reflection for ALL points: x' = 2 * r - x
            x_reflected = 2 * reversal_point - x
            
            # Select element-wise and flip inputs outside of context range
            return torch.where(is_in_range, x, x_reflected)
        
        # Apply transformation to both context and target inputs
        xc_transformed = transform_inputs(xc)
        xt_transformed = transform_inputs(xt)

        # Call the parent GPGroundTruthPredictor with the transformed inputs.
        return self.base_gt_pred.__call__(xc_transformed, yc, xt_transformed, yt)

    def sample_outputs(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        pass


class IndependentReversedContextGPGenerator(RandomScaleGPGenerator):
    """
    Generates batches of GP data where the GP is reversed at a reversal_point
    and the context points have direct counterparts in the target points.
    """

    def __init__(
        self,
        *,
        min_nc: int,
        max_nc: int,
        min_nt: int,
        max_nt: int,
        batch_size: int,
        reversal_point: float = 0.0,
        **kwargs,
    ):
        super().__init__(min_nc=min_nc, max_nc=max_nc, min_nt = min_nt, 
                         max_nt = max_nt, batch_size=batch_size, **kwargs)
        self.reversal_point = reversal_point

    def generate_batch(self) -> SyntheticBatch:
        # Sample number of context = number of target points.
        nc = torch.randint(low=self.min_nc, high=self.max_nc + 1, size=())
        nt = nc

        # Sample batch using parent method
        batch = self.sample_batch(
            nc=nc,
            nt=nt,
            batch_shape=torch.Size([self.batch_size])
        )

        return batch
    
    
    def sample_batch(
        self,
        nc: int,
        nt: int,
        batch_shape: torch.Size,
    ) -> SyntheticBatch:

        current_range = self.context_range
        def sample_in_range(n: int) -> torch.Tensor:
            # returns [B, n, dim]
            return (
                torch.rand((*batch_shape, n, self.dim))
                * (current_range[:, 1] - current_range[:, 0])
                + current_range[:, 0]
            )

        xc = sample_in_range(nc)
        xt_src = sample_in_range(nt)
        # 1) Sample context inputs on the base side
        #xc = self.sample_inputs(nc=nc, context_range=current_range, batch_shape=batch_shape)

        # 2) Sample target *source* inputs on the SAME base side (independent of xc)
        #xt_src = self.sample_inputs(nc=nt, context_range=current_range, batch_shape=batch_shape)

        # 3) Sample outputs from the GP on the base-side points (context + target-source)
        x_src = torch.cat([xc, xt_src], dim=1)          # [B, nc+nt, d]
        y_src, non_reversed_gt_pred = self.sample_outputs(x=x_src)  # y on base-side

        yc = y_src[:, :nc, :]   # [B, nc, 1]
        yt = y_src[:, nc:, :]   # [B, nt, 1]

        # 4) Mirror ONLY the target x’s to the other side for the *observed* target inputs
        xt = 2 * self.reversal_point - xt_src  # reflects about reversal_point (0 => x -> -x)

        # 5) Combine into a SyntheticBatch in observed coordinates
        x = torch.cat([xc, xt], dim=1)
        y = torch.cat([yc, yt], dim=1)

        reversed_gt_pred = ReversedGPGroundTruthPredictor(
            base_gt_pred=non_reversed_gt_pred,
            reversal_point=self.reversal_point,
            context_range=current_range,
        )

        return SyntheticBatch(
            x=x,
            y=y,
            xc=xc,
            yc=yc,
            xt=xt,
            yt=yt,
            gt_pred=reversed_gt_pred,
        )



"""
Arsen and Nihar reversal generalisation task with turn point inference.

"""
class ReversedInferenceGPGroundTruthPredictor(GroundTruthPredictor):
    def __init__(
        self,
        base_gt_pred: GPGroundTruthPredictor,
        reversal_point: float,
        context_range: torch.Tensor,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_gt_pred = base_gt_pred
        self.reversal_point = reversal_point
        self.context_range = context_range
        self.min_context = context_range[:, 0]
        self.max_context = context_range[:, 1]

    def __call__(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
        yt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        min_c = self.min_context.to(xc.device)
        max_c = self.max_context.to(xc.device)
        r = self.reversal_point

        def transform_inputs(x: torch.Tensor) -> torch.Tensor:
            # keep points inside [min_c, max_c], reflect everything else about r
            is_in_range = (x >= min_c) & (x <= max_c)
            x_reflected = 2.0 * r - x
            return torch.where(is_in_range, x, x_reflected)

        xc_t = transform_inputs(xc)
        xt_t = transform_inputs(xt)

        return self.base_gt_pred.__call__(xc_t, yc, xt_t, yt)

    def sample_outputs(self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        pass



class PartialRevealRandomFlipGPGenerator(RandomScaleGPGenerator):
    def __init__(
        self,
        *,
        min_nc: int,
        max_nc: int,
        min_nt: int,
        max_nt: int,
        batch_size: int,
        reversal_point_range: Tuple[float, float] = (-1.0, 1.0),
        reveal_frac_range: Optional[Tuple[float, float]] = None,
        **kwargs,
    ):
        super().__init__(
            min_nc=min_nc,
            max_nc=max_nc,
            min_nt=min_nt,
            max_nt=max_nt,
            batch_size=batch_size,
            **kwargs,
        )
        self.reversal_point_range = reversal_point_range
        self.reveal_frac_range = reveal_frac_range

    def generate_batch(self) -> SyntheticBatch:
        nc = torch.randint(low=self.min_nc, high=self.max_nc + 1, size=())
        nt = nc  # equal by construction
        return self.sample_batch(nc=nc, nt=nt, batch_shape=torch.Size([self.batch_size]))

    def sample_batch(self, nc: int, nt: int, batch_shape: torch.Size) -> SyntheticBatch:
        # 1) sample reversal point r
        # r0, r1 = self.reversal_point_range
        # r = (r0 + (r1 - r0) * torch.rand(())).item()
        r = torch.rand(()).item()  # uniform in [0, 1)


        # 2) sample context xc in [r-4, r]
        lo = r - 4.0
        hi = r
        if hi <= lo:
            hi = lo + 1e-6

        xc = lo + (hi - lo) * torch.rand((*batch_shape, nc, 1))
        xc, _ = torch.sort(xc, dim=1)

        # 3) sample yc from ONE coherent GP draw at xc
        yc, base_gt_pred = self.sample_outputs(x=xc)

        # 4) targets are the mirrored context points => nt == nc
        xtp = 2.0 * r - xc           # (B, nc, 1) but descending if xc ascending
        xtp = xtp.flip(dims=[-2])    # make ascending
        ytp = yc.flip(dims=[-2])     # match order

        # 5) reveal p% of target-prime with smallest x into context (unchanged)
        if self.reveal_frac_range is None:
            p = 0.0
        else:
            p0, p1 = self.reveal_frac_range
            p = (p0 + (p1 - p0) * torch.rand(())).item()

        # safety clamp (optional but recommended)
        p = float(max(0.0, min(1.0, p)))
        k = int(p * nc)
        k = max(0, min(k, nc - 1))   # keep at least 1 target point

        xc_new = torch.cat([xc, xtp[:, :k, :]], dim=1)
        yc_new = torch.cat([yc, ytp[:, :k, :]], dim=1)

        # --- sort the NEW context after concatenation (critical for sequential models) ---
        idx = xc_new[..., 0].argsort(dim=1)   # (B, nc+k)
        idx_x = idx.unsqueeze(-1)            # (B, nc+k, 1)
        xc_new = xc_new.gather(1, idx_x)
        yc_new = yc_new.gather(1, idx_x)

        xt_new = xtp[:, k:, :]
        yt_new = ytp[:, k:, :]


        # Ground-truth predictor: source side is exactly [r-4, r]
        src_range = torch.tensor([[r - 4.0, r]], device=xc_new.device, dtype=xc_new.dtype)

        gt_pred = ReversedInferenceGPGroundTruthPredictor(
            base_gt_pred=base_gt_pred,
            reversal_point=r,
            context_range=src_range,
        )

        x = torch.cat([xc_new, xt_new], dim=1)
        y = torch.cat([yc_new, yt_new], dim=1)

        return SyntheticBatch(
            x=x,
            y=y,
            xc=xc_new,
            yc=yc_new,
            xt=xt_new,
            yt=yt_new,
            gt_pred=gt_pred,
        )



"""
Arsen and Nihar varying lengthscale kernel.

"""

class B2SplitGPGroundTruthPredictor(GroundTruthPredictor):
    def __init__(self, base_gt_pred: GPGroundTruthPredictor, b2: float):
        self.base_gt_pred = base_gt_pred
        self.b2 = float(b2)

    @property
    def kernel(self):
        return self.base_gt_pred.kernel

    @property
    def likelihood(self):
        return self.base_gt_pred.likelihood

    @staticmethod
    def _module_device(m: torch.nn.Module) -> torch.device:
        # robustly get device even if module has no parameters
        for p in m.parameters(recurse=True):
            return p.device
        for b in m.buffers(recurse=True):
            return b.device
        return torch.device("cpu")

    def _ensure_base_on(self, device: torch.device):
        # Move BOTH kernel and likelihood together
        if self._module_device(self.base_gt_pred.kernel) != device:
            self.base_gt_pred.kernel = self.base_gt_pred.kernel.to(device)
            self.base_gt_pred._result_cache = None  # cached tensors may be on old device

        if self._module_device(self.base_gt_pred.likelihood) != device:
            self.base_gt_pred.likelihood = self.base_gt_pred.likelihood.to(device)
            self.base_gt_pred._result_cache = None

    def __call__(self, xc, yc, xt, yt=None):
        self._ensure_base_on(xc.device)
        return self.base_gt_pred(xc=xc, yc=yc, xt=xt, yt=yt)

    def sample_outputs(self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()):
        self._ensure_base_on(x.device)
        return self.base_gt_pred.sample_outputs(x)





def _unwrap_base_kernel(k):
    # unwrap ScaleKernel etc.
    while hasattr(k, "base_kernel"):
        k = k.base_kernel
    return k


class B2SplitRevealGPGenerator(RandomScaleGPGenerator):
    """
    Mixed generator with two cases (50/50 if enabled):

    Case A (b2-split):
      1) sample GP + hyperparams
      2) read b2 from kernel
      3) sample nc points in [x_min, b2]
      4) sample nt points in [b2, x_max]
      5) append lowest p% of right points to context, predict on the rest

    Case B (zero-split):
      1) sample nc points in [x_min, 0]
      2) sample nt points in [0, x_max]
      3) append lowest p% of right points to context, predict on the rest

    In both cases: y is sampled coherently from the same GP on (xc_left + xr).
    """

    def __init__(
        self,
        *,
        min_nc: int,
        max_nc: int,
        min_nt: int,
        max_nt: int,
        batch_size: int,
        reveal_frac_range: Optional[Tuple[float, float]] = None,
        mix_with_zero_split: bool = True,
        zero_split_point: float = 0.0,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(
            min_nc=min_nc,
            max_nc=max_nc,
            min_nt=min_nt,
            max_nt=max_nt,
            batch_size=batch_size,
            **kwargs,
        )
        self.reveal_frac_range = reveal_frac_range
        self.mix_with_zero_split = bool(mix_with_zero_split)
        self.zero_split_point = float(zero_split_point)
        self.eps = float(eps)

    def generate_batch(self) -> SyntheticBatch:
        nc = int(torch.randint(self.min_nc, self.max_nc + 1, size=()).item())
        nt = int(torch.randint(self.min_nt, self.max_nt + 1, size=()).item())
        return self.sample_batch(nc=nc, nt=nt, batch_shape=torch.Size([self.batch_size]))

    def sample_batch(self, nc: int, nt: int, batch_shape: torch.Size) -> SyntheticBatch:
        device = self.context_range.device
        dtype = self.context_range.dtype

        # overall region [x_min, x_max]
        x_min = float(self.context_range[0, 0].item())
        x_max = float(self.context_range[0, 1].item())

        # 1) sample GP predictor FIRST (so b2 exists if we need it, and y is coherent)
        gt_pred_base = self.set_up_gp()
        k = _unwrap_base_kernel(gt_pred_base.kernel)

        # Wrap so kernel/likelihood always move to the same device as xc/xt during validation/plotting/DDP
        b2_for_logging = float(getattr(k, "b2", 0.0)) if getattr(k, "b2", None) is not None else None
        gt_pred = B2SplitGPGroundTruthPredictor(base_gt_pred=gt_pred_base, b2=b2_for_logging)

        # 2) choose split rule (50/50 if enabled)
        use_zero_split = False
        if self.mix_with_zero_split and (torch.rand((), device=device).item() < 0.5):
            use_zero_split = True

        if use_zero_split:
            split = self.zero_split_point  # typically 0.0
            nt = nc
        else:
            if not hasattr(k, "b2") or k.b2 is None:
                raise RuntimeError(
                    "Kernel does not expose attribute `.b2` (needed for b2-split case). "
                    "Either use a kernel that stores b2 (e.g. RandomContinuousGibbsKernel), "
                    "or set mix_with_zero_split=True so the zero-split case can be used."
                )
            split = float(k.b2)

        # 3) sample left context x in [x_min, split]
        xc_left = x_min + (split - x_min) * torch.rand((*batch_shape, nc, 1), device=device, dtype=dtype)
        xc_left, _ = torch.sort(xc_left, dim=1)

        # 4) sample right candidates x in [split, x_max]
        xr = split + (x_max - split) * torch.rand((*batch_shape, nt, 1), device=device, dtype=dtype)
        xr, _ = torch.sort(xr, dim=1)

        # 5) decide reveal fraction p (or 0)
        if self.reveal_frac_range is None:
            p = 0.0
        else:
            p0, p1 = self.reveal_frac_range
            p = float((p0 + (p1 - p0) * torch.rand((), device=device)).item())
            p = max(0.0, min(1.0, p))

        k_reveal = int(p * nt)

        # enforce: reveal at least 2 points if possible, and keep at least 1 target to predict
        if nt >= 3:
            k_reveal = max(2, min(k_reveal, nt - 1))
        else:
            k_reveal = max(0, min(k_reveal, nt))

        # 6) build final context + targets (lowest x on right get revealed)
        xr_reveal = xr[:, :k_reveal, :]
        xr_test = xr[:, k_reveal:, :]

        # 7) coherent GP draw on (xc_left + xr)
        x_all = torch.cat([xc_left, xr], dim=1)
        y_all = gt_pred.sample_outputs(x_all)

        yc_left = y_all[:, :nc, :]
        yr = y_all[:, nc:, :]
        yr_reveal = yr[:, :k_reveal, :]
        yr_test = yr[:, k_reveal:, :]

        xc_new = torch.cat([xc_left, xr_reveal], dim=1)
        yc_new = torch.cat([yc_left, yr_reveal], dim=1)
        xt_new = xr_test
        yt_new = yr_test

        if torch.rand((), device=device) < 0.5:
            # flip x
            xc_new = -xc_new
            xt_new = -xt_new

        x = torch.cat([xc_new, xt_new], dim=1)
        y = torch.cat([yc_new, yt_new], dim=1)

        return SyntheticBatch(
            x=x,
            y=y,
            xc=xc_new,
            yc=yc_new,
            xt=xt_new,
            yt=yt_new,
            gt_pred=gt_pred,
        )





"""
    Arsen and Nihar reversal generalisation task with turn point inference and
    varying context range size.
    
"""
class ReversedContextGPGenerator1(RandomScaleGPGenerator):
    """
    Generates batches of GP data where the GP is reversed at a reversal_point
    and the context points have direct counterparts in the target points.
    """

    def __init__(
        self,
        *,
        min_nc: int,
        max_nc: int,
        min_nt: int,
        max_nt: int,
        batch_size: int,
        reversal_point: float = 0.0,
        same_targets: bool = True, # whether target points are same as context points reversed
        shared_noise: bool = True, # whether the noise is also shared between context and target
        bidirectional_reversal: bool = True, # whether to randomly reverse around reversal point or not
        sort_xs: bool = False, # whether to sort the context points along x axis
        targets_only_outside_context: bool = True,
        **kwargs,
    ):
        super().__init__(min_nc=min_nc, max_nc=max_nc, min_nt = min_nt, 
                         max_nt = max_nt, batch_size=batch_size, **kwargs)
        self.reversal_point = reversal_point
        self.same_targets = same_targets
        self.shared_noise = shared_noise
        self.bidirectional_reversal = bidirectional_reversal
        self.sort_xs = sort_xs
        self.targets_only_outside_context = targets_only_outside_context

    def generate_batch(self) -> SyntheticBatch:
        # Sample number of context = number of target points.
        nc = torch.randint(low=self.min_nc, high=self.max_nc + 1, size=())

        if self.same_targets:
            nt = nc
        else:
            nt = torch.randint(low=self.min_nt, high=self.max_nt + 1, size=())

        # Sample batch using parent method
        batch = self.sample_batch(
            nc=nc,
            nt=nt,
            batch_shape=torch.Size([self.batch_size])
        )

        return batch
    
    def sample_batch(
        self,
        nc: int,
        nt: int,
        batch_shape: torch.Size,
    ) -> SyntheticBatch:
        
        # Randomly flip the context range around the reversal point
        if torch.rand(1) > 0.5 and self.bidirectional_reversal:
            current_range = 2 * self.reversal_point - self.context_range
            current_range = current_range.flip(dims=[1])
            flipped = True
        else:
            current_range = self.context_range
            flipped = False

        # Sample context inputs
        xc = self.sample_inputs(
            n=nc,
            context_range=current_range,
            batch_shape=batch_shape)
        
        if self.sort_xs:
            xc, _ = torch.sort(xc, dim = -2)

        if self.same_targets and self.shared_noise:
            # Create target inputs by reversing context inputs around reversal_point
            xt = 2 * self.reversal_point - xc
            yc, non_reversed_gt_pred = self.sample_outputs(x=xc)
            yt = yc

        elif not self.shared_noise:
            if self.same_targets:
                xt = 2 * self.reversal_point - xc
                xquery = torch.concat([xc, xc], axis = 1)
            else:
                xt_reversed = self.sample_inputs(
                    n=nt,
                    context_range=current_range,
                    batch_shape=batch_shape)
                
                if self.targets_only_outside_context:
                    mask = torch.ones_like(xt_reversed, dtype=torch.bool)
                else:
                    mask = torch.rand_like(xt_reversed) < 0.5

                xt = torch.where(mask, 2 * self.reversal_point - xt_reversed, xt_reversed)
                xquery = torch.concat([xc, xt_reversed], axis = 1)
                
            yquery, non_reversed_gt_pred = self.sample_outputs(x=xquery)
            yc = yquery[:, :nc, :]
            yt = yquery[:, nc:, :]

        else:
            raise NotImplementedError("Noise can only be shared if targets are flipped contexts.")

        if self.sort_xs:
            xt = xt.flip(dims=[-2])
            yt = yt.flip(dims=[-2])

        x = torch.concat([xc, xt], axis=1)
        y = torch.concat([yc, yt], axis=1)

        reversed_gt_pred = ReversedGPGroundTruthPredictor(
            base_gt_pred=non_reversed_gt_pred,
            reversal_point=self.reversal_point,
            context_range=current_range
        )
        
        return SyntheticBatch(
            x=x,
            y=y,
            xc=xc,
            yc=yc,
            xt=xt,
            yt=yt,
            gt_pred=reversed_gt_pred,
        )
    
    def sample_inputs(
        self,
        n: int,
        context_range: torch.Tensor,
        batch_shape: torch.Size,
    ) -> torch.Tensor:

        # Sample context inputs
        xc = (
            torch.rand((*batch_shape, n, self.dim))
            * (context_range[:, 1] - context_range[:, 0])
            + context_range[:, 0]
        )

        return xc

class RandomReversalGPGeneratorv2(ReversedContextGPGenerator1):
    def __init__(
        self,
        *,
        min_nc: int,
        max_nc: int,
        min_nt: int,
        max_nt: int,
        batch_size: int,
        reversal_range: Tuple[float, float],
        priming_frac_range: Tuple[float, float],
        same_targets: bool = False,
        shared_noise: bool = True,
        context_in_targets: bool = False,
        **kwargs,
    ):
        super().__init__(
            min_nc=min_nc,
            max_nc=max_nc,
            min_nt=min_nt,
            max_nt=max_nt,
            batch_size=batch_size,
            **kwargs,   # includes kernel, noise_std, dim, ranges, etc
        )
        self.reversal_range = reversal_range
        self.priming_frac_range = priming_frac_range
        self.original_context_range = self.context_range.clone()
        self.context_in_targets = context_in_targets
        self.same_targets = same_targets
        self.shared_noise = shared_noise

        

    def generate_batch(self) -> SyntheticBatch:
        # Sample number of context = number of target points.
        
        reversal_point = float(torch.rand(1)) * (self.reversal_range[1] - self.reversal_range[0]) + self.reversal_range[0]        
        current_range = self.original_context_range.clone()
        current_range[:, 1] = reversal_point

        self.reversal_point = reversal_point
        self.context_range = current_range

        
        batch = super().generate_batch()

        priming_frac_low = self.priming_frac_range[0]
        priming_frac_high = self.priming_frac_range[1]
        priming_frac = float(torch.rand(1)) * (priming_frac_high - priming_frac_low) + priming_frac_low
        n_priming = int(priming_frac * batch.xc.shape[1])

        xc_priming = batch.xc[:, batch.xc.shape[1] - n_priming:, :]
        yc_priming = batch.yc[:, batch.yc.shape[1] - n_priming:, :]
        xc_priming = 2 * self.reversal_point - xc_priming
        xc_priming = xc_priming.flip(dims=[-2])
        yc_priming = yc_priming.flip(dims=[-2])

        xc = torch.concat([batch.xc, xc_priming], axis=1)
        yc = torch.concat([batch.yc, yc_priming], axis=1)

        if self.same_targets and self.context_in_targets:
            xt = batch.x
            yt = batch.y
        elif self.same_targets:
            xt = batch.x[:, batch.xc.shape[1] + n_priming:, :]
            yt = batch.y[:, batch.yc.shape[1] + n_priming:, :]
        else:
            xt = batch.xt
            yt = batch.yt

        assert isinstance(batch.gt_pred, ReversedGPGroundTruthPredictor)
        batch.gt_pred.priming_frac = priming_frac

        return SyntheticBatch(
            x=torch.concat([xc, xt], axis=1),
            y=torch.concat([yc, yt], axis=1),
            xc=xc,
            yc=yc,
            xt=xt,
            yt=yt,
            gt_pred=batch.gt_pred
        )
