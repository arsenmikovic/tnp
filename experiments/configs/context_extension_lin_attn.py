#!/usr/bin/env python3
"""
compare_two_models_separate_plots.py

- Build experiment A (config_a) and experiment B (config_b)
- Load ckpt_a into model A, ckpt_b into model B
- Generate ONE batch from generator in experiment A (train or val)
- Evaluate BOTH models on the SAME batch
- Produce TWO SEPARATE plots (one per model) using your existing plot() util

Example:
python experiments/compare_two_models_separate_plots.py \
  --ckpt_a path/to/A.ckpt \
  --ckpt_b path/to/B.ckpt \
  --config_a experiments/configs/models/tnp.yml experiments/configs/generators/gp-reversal.yml \
  --config_b experiments/configs/models/tnp_big.yml experiments/configs/generators/gp-reversal.yml \
  --nc 128 \
  --device cuda \
  --wandb_project gp-reversal \
  --wandb_name compare-separate
"""

import argparse
import copy
import os
import random
import sys
import time
from typing import List

import lightning.pytorch as pl
import torch
import wandb

# your plot util
from plot import plot

from tnp.utils.experiment_utils import initialize_experiment


# ---------------------------
# utilities
# ---------------------------

def get_one_batch(gen):
    if hasattr(gen, "__iter__"):
        return next(iter(gen))
    if hasattr(gen, "generate_batch"):
        return gen.generate_batch()
    if hasattr(gen, "__getitem__"):
        return gen[0]
    raise TypeError("Generator has no __iter__/generate_batch/__getitem__.")


def to_device(batch, device: torch.device):
    for k in ["x", "y", "xc", "yc", "xt", "yt"]:
        if hasattr(batch, k):
            v = getattr(batch, k)
            if torch.is_tensor(v):
                setattr(batch, k, v.to(device))
    return batch


def load_model_weights_from_litwrapper_ckpt(model: torch.nn.Module, ckpt_path: str):
    """
    Your checkpoints were saved from a LitWrapper that has `self.model`.
    Those weights appear under keys starting with 'model.' in checkpoint['state_dict'].
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    model_sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    if not model_sd:
        print("ERROR: couldn't find keys starting with 'model.' in checkpoint.")
        print("Here are some checkpoint keys:", list(sd.keys())[:40])
        raise SystemExit(1)

    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    if missing or unexpected:
        print("Warning: non-empty load_state_dict report.")
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)


def init_experiment_from_configs(config_paths: List[str]):
    """
    initialize_experiment() in your repo likely parses CLI configs.
    We create experiments by temporarily spoofing sys.argv.

    Tries:
      1) prog --config <paths...>
      2) prog <paths...>
    """
    old_argv = sys.argv[:]
    prog = old_argv[0] if old_argv else "compare_two_models_separate_plots.py"
    last_err = None

    for argv in ([prog, "--config", *config_paths], [prog, *config_paths]):
        try:
            sys.argv = argv
            exp = initialize_experiment()
            return exp
        except SystemExit as e:
            last_err = e
        except Exception as e:
            last_err = e
        finally:
            sys.argv = old_argv

    raise RuntimeError(f"Failed to initialize experiment from configs={config_paths}. Last error: {last_err}")


def force_nc_nt(gen, n: int):
    for attr in ["min_nc", "max_nc", "min_nt", "max_nt"]:
        if hasattr(gen, attr):
            setattr(gen, attr, n)


@torch.no_grad()
def compute_nll(model, pred_fn, batch) -> float:
    dist = pred_fn(model, batch)
    return -(dist.log_prob(batch.yt).sum() / batch.yt[..., 0].numel()).item()


# ---------------------------
# main
# ---------------------------

def main():
    p = argparse.ArgumentParser()

    p.add_argument("--ckpt_a", type=str, required=True)
    p.add_argument("--ckpt_b", type=str, required=True)

    p.add_argument("--config_a", nargs="+", required=True)
    p.add_argument("--config_b", nargs="+", required=True)

    p.add_argument("--label_a", type=str, default="Model A")
    p.add_argument("--label_b", type=str, default="Model B")

    p.add_argument("--nc", type=int, required=True)
    p.add_argument("--use_val_generator", action="store_true")
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--seed", type=int, default=-1)

    p.add_argument("--x_min", type=float, default=-4.0)
    p.add_argument("--x_max", type=float, default=4.0)
    p.add_argument("--points_per_dim", type=int, default=64)

    # W&B
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="gp-reversal")
    p.add_argument("--wandb_name", type=str, default="compare-separate-plots")
    p.add_argument("--wandb_dir", type=str, default="wandb")

    args = p.parse_args()

    # seed
    if args.seed is None or args.seed < 0:
        seed = (int(time.time() * 1000) ^ (os.getpid() << 16) ^ random.getrandbits(32)) % (2**32)
    else:
        seed = int(args.seed) % (2**32)

    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device(args.device if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")

    # build experiments
    exp_a = init_experiment_from_configs(args.config_a)
    exp_b = init_experiment_from_configs(args.config_b)

    model_a = exp_a.model
    model_b = exp_b.model
    pred_fn_a = exp_a.misc.pred_fn
    pred_fn_b = exp_b.misc.pred_fn

    # load weights
    load_model_weights_from_litwrapper_ckpt(model_a, args.ckpt_a)
    load_model_weights_from_litwrapper_ckpt(model_b, args.ckpt_b)

    model_a.eval().to(device)
    model_b.eval().to(device)

    # generate ONE batch (from exp_a generator)
    gen = exp_a.generators.val if args.use_val_generator else exp_a.generators.train

    if hasattr(gen, "deterministic"):
        gen.deterministic = False

    force_nc_nt(gen, args.nc)

    batch = get_one_batch(gen)
    batch = to_device(copy.deepcopy(batch), device)

    # keep only first task
    for k in ["x", "y", "xc", "yc", "xt", "yt"]:
        if hasattr(batch, k):
            v = getattr(batch, k)
            if torch.is_tensor(v):
                setattr(batch, k, v[:1])

    # W&B
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            dir=args.wandb_dir,
            config={
                "seed_used": int(seed),
                "nc": int(args.nc),
                "use_val_generator": bool(args.use_val_generator),
                "ckpt_a": args.ckpt_a,
                "ckpt_b": args.ckpt_b,
                "config_a": args.config_a,
                "config_b": args.config_b,
                "x_range": [float(args.x_min), float(args.x_max)],
                "points_per_dim": int(args.points_per_dim),
            },
        )

    # Evaluate both on SAME batch (use deep copies so nothing mutates between calls)
    nll_a = compute_nll(model_a, pred_fn_a, copy.deepcopy(batch))
    nll_b = compute_nll(model_b, pred_fn_b, copy.deepcopy(batch))

    if wandb.run is not None:
        wandb.log({"nll_a": nll_a, "nll_b": nll_b})

    # Plot A (separate figure)
    plot(
        model=model_a,
        batches=[copy.deepcopy(batch)],
        num_fig=1,
        x_range=(args.x_min, args.x_max),
        points_per_dim=args.points_per_dim,
        name=f"compare_sep/{args.label_a.replace(' ', '_')}/nc_{args.nc}",
        logging=(wandb.run is not None),
        savefig=False,
        pred_fn=pred_fn_a,
    )

    # Plot B (separate figure)
    plot(
        model=model_b,
        batches=[copy.deepcopy(batch)],
        num_fig=1,
        x_range=(args.x_min, args.x_max),
        points_per_dim=args.points_per_dim,
        name=f"compare_sep/{args.label_b.replace(' ', '_')}/nc_{args.nc}",
        logging=(wandb.run is not None),
        savefig=False,
        pred_fn=pred_fn_b,
    )

    if wandb.run is not None:
        wandb.finish()

    print(
        f"Done.\n"
        f"  seed={seed}\n"
        f"  Nc={batch.xc.shape[1]} Nt={batch.xt.shape[1]}\n"
        f"  {args.label_a} NLL={nll_a:.6f}\n"
        f"  {args.label_b} NLL={nll_b:.6f}"
    )


if __name__ == "__main__":
    main()


# python experiments/context_extension_lin_attn.py \
#   --ckpt_a gp-reversal/jykzqh3g/checkpoints/epoch=499-step=1000000.ckpt \
#   --ckpt_b gp-reversal-linear-attention/ihea0zgp/checkpoints/epoch=499-step=1000000.ckpt \
#   --config_a experiments/configs/models/tnp.yml experiments/configs/generators/gp-reversal.yml \
#   --config_b experiments/configs/models/lin_tnp.yml experiments/configs/generators/gp-reversal.yml \
#   --nc 64 \
#   --label_a "TNP" \
#   --label_b "TNP-linear attn" \
#   --wandb_project gp-reversal-linear-attention \
#   --wandb_name compare-separate
