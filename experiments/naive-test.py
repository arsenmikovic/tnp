import argparse
import copy
import os
import random
import time
from typing import Tuple

import lightning.pytorch as pl
import torch
import wandb

from plot import plot
from tnp.utils.experiment_utils import initialize_experiment


def get_one_batch(gen):
    if hasattr(gen, "__iter__"):
        return next(iter(gen))
    if hasattr(gen, "generate_batch"):
        return gen.generate_batch()
    if hasattr(gen, "__getitem__"):
        return gen[0]
    raise TypeError("Generator has no __iter__/generate_batch/__getitem__.")


def to_device(batch, device: torch.device):
    for k in ["xc", "yc", "xt", "yt"]:
        if hasattr(batch, k):
            v = getattr(batch, k)
            if torch.is_tensor(v):
                setattr(batch, k, v.to(device))
    return batch


def load_model_weights_from_litwrapper_ckpt(model: torch.nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model_sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    if not model_sd:
        print("ERROR: couldn't find keys starting with 'model.' in checkpoint.")
        print("Here are some checkpoint keys:", list(sd.keys())[:40])
        raise SystemExit(1)
    model.load_state_dict(model_sd, strict=False)


def _sort_by_x(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    idx = torch.argsort(x[..., 0], dim=1)  # [B, N]
    idx_x = idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    idx_y = idx.unsqueeze(-1).expand(-1, -1, y.shape[-1])
    return torch.gather(x, 1, idx_x), torch.gather(y, 1, idx_y)


@torch.no_grad()
def compute_nll(model, pred_fn, batch) -> float:
    dist = pred_fn(model, batch)
    return -(dist.log_prob(batch.yt).sum() / batch.yt[..., 0].numel()).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--nc", type=int, required=True)

    p.add_argument("--use_val_generator", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=-1)

    # Plot grid params
    p.add_argument("--x_min", type=float, default=-4.0)
    p.add_argument("--x_max", type=float, default=4.0)
    p.add_argument("--points_per_dim", type=int, default=64)

    # W&B
    p.add_argument("--wandb_project", type=str, default="gp-reversal")
    p.add_argument("--wandb_name", type=str, default="context-extension-percent-targets")
    p.add_argument("--wandb_dir", type=str, default="wandb")

    args, _ = p.parse_known_args()

    # Build experiment (this may set misc.seed internally)
    experiment = initialize_experiment()
    pred_fn = experiment.misc.pred_fn
    model = experiment.model

    # Reseed AFTER initialize_experiment so each run can differ
    if args.seed is None or args.seed < 0:
        seed = (int(time.time() * 1000) ^ (os.getpid() << 16) ^ random.getrandbits(32)) % (2**32)
    else:
        seed = int(args.seed) % (2**32)

    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Load trained weights
    load_model_weights_from_litwrapper_ckpt(model, args.ckpt)
    model.eval()

    device = torch.device(args.device if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    model.to(device)

    # Pick generator
    gen = experiment.generators.val if args.use_val_generator else experiment.generators.train
    if hasattr(gen, "deterministic"):
        gen.deterministic = False  # avoid repeating

    # Force Nc=Nt=args.nc (your generator uses min_nc/max_nc/min_nt/max_nt)
    for attr in ["min_nc", "max_nc", "min_nt", "max_nt"]:
        if hasattr(gen, attr):
            setattr(gen, attr, args.nc)

    # Generate ONE batch, take ONE task
    batch0 = get_one_batch(gen)
    batch0 = to_device(copy.deepcopy(batch0), device)
    batch0.xc, batch0.yc, batch0.xt, batch0.yt = batch0.xc[:1], batch0.yc[:1], batch0.xt[:1], batch0.yt[:1]

    # Sort by x (time-series look)
    batch0.xc, batch0.yc = _sort_by_x(batch0.xc, batch0.yc)
    batch0.xt, batch0.yt = _sort_by_x(batch0.xt, batch0.yt)

    # Quick leakage check (logs to console + W&B)
    xc_min, xc_max = batch0.xc[..., 0].min().item(), batch0.xc[..., 0].max().item()
    xt_min, xt_max = batch0.xt[..., 0].min().item(), batch0.xt[..., 0].max().item()
    print(f"xc range: [{xc_min:.3f}, {xc_max:.3f}] | xt range: [{xt_min:.3f}, {xt_max:.3f}]")

    # W&B init
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        dir=args.wandb_dir,
        config={
            "ckpt": args.ckpt,
            "seed_used": int(seed),
            "requested_nc": int(args.nc),
            "actual_nc": int(batch0.xc.shape[1]),
            "actual_nt_full": int(batch0.xt.shape[1]),
            "xc_range": [xc_min, xc_max],
            "xt_range": [xt_min, xt_max],
            "fractions": [0.33, 0.66, 1.0],
        },
    )

    Nt_full = int(batch0.xt.shape[1])

    # SIMPLE LOOP: just truncate targets, no autoregressive updates
    for frac in [0.33, 0.66, 1.0]:
        k = max(1, int(round(frac * Nt_full)))

        b = copy.deepcopy(batch0)
        b.xt = b.xt[:, :k, :]
        b.yt = b.yt[:, :k, :]

        nll = compute_nll(model, pred_fn, b)
        wandb.log({f"nll/{int(frac*100)}pct": nll, f"k_targets/{int(frac*100)}pct": k})

        plot(
            model=model,
            batches=[b],
            num_fig=1,
            x_range=(args.x_min, args.x_max),
            points_per_dim=args.points_per_dim,
            name=f"percent_targets/{int(frac*100)}pct",
            logging=True,
            savefig=False,
            pred_fn=pred_fn,
        )

    wandb.finish()
    print(f"Done. Logged 3 plots (33/66/100% targets). seed={seed}")


if __name__ == "__main__":
    main()
