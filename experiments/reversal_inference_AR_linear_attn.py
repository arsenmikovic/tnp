#!/usr/bin/env python3
"""
reversal_inference_compare_two_models.py

- Build experiment A from config_a list and experiment B from config_b list
- Load ckpt_a into model A and ckpt_b into model B
- Sample ONE batch ONCE from generator in experiment A (train or val)
- Run the same progressive reveal plots for BOTH models, on the SAME batch
- Plots are separate (distinct wandb image keys) per model

Example:

python experiments/reversal_inference_AR_linear_attn.py \
  --ckpt_a gp-reversal-inference/2ypx1r6i/checkpoints/epoch=499-step=1000000.ckpt \
  --ckpt_b gp-reversal-inference-linear-attention/vbp8okjt/checkpoints/epoch=499-step=1000000.ckpt \
  --config_a experiments/configs/models/tnp.yml experiments/configs/generators/gp-random-reversal.yml \
  --config_b experiments/configs/models/lin_tnp.yml experiments/configs/generators/gp-random-reversal.yml \
  --nc 64 \
  --label_a "TNP" \
  --label_b "TNP-linAttn" \
  --wandb_project gp-reversal-inference-linear-attention \
  --wandb_name compare-progressive \
  --wandb_dir gp-reversal-inference-linear-attention/wandb
"""

import argparse
import copy
import os
import random
import sys
import time
from typing import List, Tuple

import numpy as np
import lightning.pytorch as pl
import torch
import wandb

from plot import plot
from tnp.utils.experiment_utils import initialize_experiment


# ---------------------------
# helpers
# ---------------------------

def init_experiment_from_configs(config_paths: List[str]):
    """
    Your initialize_experiment() reads config paths from CLI.
    We spoof sys.argv to build two separate experiments in one script.
    """
    old_argv = sys.argv[:]
    prog = old_argv[0] if old_argv else "compare_two_models_3graphs.py"
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

    raise RuntimeError(f"Failed to init experiment from configs={config_paths}. Last error: {last_err}")


def load_model_weights_from_litwrapper_ckpt(model: torch.nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # LitWrapper stores under "model.*"
    model_sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    if not model_sd:
        print("ERROR: couldn't find keys starting with 'model.' in checkpoint.")
        print("Some checkpoint keys:", list(sd.keys())[:50])
        raise SystemExit(1)

    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    if missing or unexpected:
        print("Warning: load_state_dict had differences.")
        print("Missing:", missing)
        print("Unexpected:", unexpected)


def reseed_everything(seed: int):
    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def try_reseed_generator(gen, seed: int):
    """
    Best-effort reseed for various custom generator implementations.
    This is what usually fixes "same batch every run".
    """
    # common patterns
    for attr in ["seed", "base_seed", "rng_seed"]:
        if hasattr(gen, attr):
            try:
                setattr(gen, attr, int(seed))
            except Exception:
                pass

    # numpy rng
    for attr in ["rng", "np_rng"]:
        if hasattr(gen, attr):
            try:
                setattr(gen, attr, np.random.default_rng(int(seed)))
            except Exception:
                pass

    # python random state
    for attr in ["random_state"]:
        if hasattr(gen, attr):
            try:
                setattr(gen, attr, random.Random(int(seed)))
            except Exception:
                pass

    # explicit setter
    for fn in ["set_seed", "seed_everything", "reseed"]:
        if hasattr(gen, fn) and callable(getattr(gen, fn)):
            try:
                getattr(gen, fn)(int(seed))
            except Exception:
                pass


def get_one_batch(gen):
    # Prefer generate_batch if present, to avoid __iter__ caching behavior
    if hasattr(gen, "generate_batch") and callable(gen.generate_batch):
        return gen.generate_batch()
    if hasattr(gen, "__iter__"):
        return next(iter(gen))
    if hasattr(gen, "__getitem__"):
        return gen[0]
    raise TypeError("Generator has no generate_batch/__iter__/__getitem__.")


def to_device(batch, device: torch.device):
    for k in ["x", "y", "xc", "yc", "xt", "yt"]:
        if hasattr(batch, k):
            v = getattr(batch, k)
            if torch.is_tensor(v):
                setattr(batch, k, v.to(device))
    return batch


def force_nc_nt(gen, n: int):
    for attr in ["min_nc", "max_nc", "min_nt", "max_nt"]:
        if hasattr(gen, attr):
            setattr(gen, attr, n)


def sort_targets_inplace(batch):
    # sort xt and align yt
    xt_sorted, idx = torch.sort(batch.xt, dim=1)
    yt_sorted = torch.gather(batch.yt, 1, idx)
    batch.xt, batch.yt = xt_sorted, yt_sorted
    return batch


def make_no_context_batch(base_batch, k: int):
    b = copy.deepcopy(base_batch)
    if k > 0:
        b.xc = b.xt[:, :k, :]
        b.yc = b.yt[:, :k, :]
    b.xt = b.xt[:, k:, :]
    b.yt = b.yt[:, k:, :]
    return b


def make_reveal_batch(base_batch, k: int):
    b = copy.deepcopy(base_batch)
    if k > 0:
        b.xc = torch.cat([b.xc, b.xt[:, :k, :]], dim=1)
        b.yc = torch.cat([b.yc, b.yt[:, :k, :]], dim=1)
    b.xt = b.xt[:, k:, :]
    b.yt = b.yt[:, k:, :]
    return b


def make_window_reveal_batch(base_batch, k: int, e: int):
    b = copy.deepcopy(base_batch)
    nt = b.xt.shape[1]
    k = max(0, min(int(k), nt))
    e = max(k, min(int(e), nt))

    xt_win = b.xt[:, k:e, :]
    yt_win = b.yt[:, k:e, :]

    xt_left = b.xt[:, :k, :]
    yt_left = b.yt[:, :k, :]
    xt_right = b.xt[:, e:, :]
    yt_right = b.yt[:, e:, :]

    if xt_win.shape[1] > 0:
        b.xc = torch.cat([b.xc, xt_win], dim=1)
        b.yc = torch.cat([b.yc, yt_win], dim=1)

    b.xt = torch.cat([xt_left, xt_right], dim=1)
    b.yt = torch.cat([yt_left, yt_right], dim=1)
    return b


def strip_gt_if_requested(batch, disable_gt: bool):
    if disable_gt and hasattr(batch, "gt_pred"):
        b = copy.deepcopy(batch)
        b.gt_pred = None
        return b
    return batch


# ---------------------------
# main
# ---------------------------

def main():
    p = argparse.ArgumentParser()

    p.add_argument("--ckpt_a", type=str, required=True)
    p.add_argument("--ckpt_b", type=str, required=True)

    p.add_argument("--config_a", nargs="+", required=True)
    p.add_argument("--config_b", nargs="+", required=True)

    p.add_argument("--label_a", type=str, default="ModelA")
    p.add_argument("--label_b", type=str, default="ModelB")

    p.add_argument("--nc", type=int, required=True)
    p.add_argument("--use_val_generator", action="store_true")
    p.add_argument("--device", type=str, default="cuda")

    # If -1 -> random seed each run
    p.add_argument("--seed", type=int, default=-1)

    # slider granularity (0, step, 2*step, ... <100)
    p.add_argument("--step_pct", type=int, default=20)

    # plot grid params passed to plot.py
    p.add_argument("--x_min", type=float, default=-4.0)
    p.add_argument("--x_max", type=float, default=4.0)
    p.add_argument("--points_per_dim", type=int, default=64)

    # if plot.py GT call causes device mismatch, run with this
    p.add_argument("--disable_gt", action="store_true")

    # W&B
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="compare")
    p.add_argument("--wandb_name", type=str, default="two-models-3graphs")
    p.add_argument("--wandb_dir", type=str, default="wandb")

    args = p.parse_args()

    # Choose base seed
    if args.seed is None or args.seed < 0:
        seed = (int(time.time() * 1000) ^ (os.getpid() << 16) ^ random.getrandbits(32)) % (2**32)
    else:
        seed = int(args.seed) % (2**32)

    device = torch.device(args.device if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")

    # Build both experiments (configs)
    exp_a = init_experiment_from_configs(args.config_a)
    exp_b = init_experiment_from_configs(args.config_b)

    model_a, pred_fn_a = exp_a.model, exp_a.misc.pred_fn
    model_b, pred_fn_b = exp_b.model, exp_b.misc.pred_fn

    # Load weights
    load_model_weights_from_litwrapper_ckpt(model_a, args.ckpt_a)
    load_model_weights_from_litwrapper_ckpt(model_b, args.ckpt_b)

    model_a.eval().to(device)
    model_b.eval().to(device)

    # Pick generator (from exp_a)
    gen = exp_a.generators.val if args.use_val_generator else exp_a.generators.train
    if hasattr(gen, "deterministic"):
        gen.deterministic = False
    force_nc_nt(gen, args.nc)

    # IMPORTANT: reseed right before sampling batch, AND try to reseed the generator object
    batch_seed = (seed ^ 0x9E3779B9 ^ random.getrandbits(32)) % (2**32)
    reseed_everything(batch_seed)
    try_reseed_generator(gen, batch_seed)

    batch = get_one_batch(gen)
    batch = to_device(copy.deepcopy(batch), device)

    # Use B=1 only
    for k in ["x", "y", "xc", "yc", "xt", "yt"]:
        if hasattr(batch, k):
            v = getattr(batch, k)
            if torch.is_tensor(v):
                setattr(batch, k, v[:1])

    batch = sort_targets_inplace(batch)
    nt = int(batch.xt.shape[1])
    if nt == 0:
        raise RuntimeError("Sampled batch has Nt=0.")

    # W&B init
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            dir=args.wandb_dir,
            config={
                "seed": int(seed),
                "batch_seed": int(batch_seed),
                "nc": int(args.nc),
                "nt": int(nt),
                "use_val_generator": bool(args.use_val_generator),
                "ckpt_a": args.ckpt_a,
                "ckpt_b": args.ckpt_b,
                "config_a": args.config_a,
                "config_b": args.config_b,
                "labels": [args.label_a, args.label_b],
                "step_pct": int(args.step_pct),
                "disable_gt": bool(args.disable_gt),
            },
        )

    models = [
        (args.label_a, model_a, pred_fn_a),
        (args.label_b, model_b, pred_fn_b),
    ]


    # ---- per model: (1) window plot, (2) slider no_context, (3) slider append_context
    for label, model, pred_fn in models:
        slider_name_noctx = f"{label}/slider"  # constant name -> slider
        for pct in range(0, 100, args.step_pct):
            k = int((pct / 100.0) * nt)
            k = max(0, min(k, nt - 1))  # keep >=1 target to predict

            #b = make_no_context_batch(batch, k)
            b = make_reveal_batch(batch, k)
            #b = strip_gt_if_requested(b, args.disable_gt)

            plot(
                model=model,
                batches=[copy.deepcopy(b)],
                num_fig=1,
                x_range=(args.x_min, 5.0),
                points_per_dim=args.points_per_dim,
                name=slider_name_noctx,  # SAME key every time -> slider
                logging=(wandb.run is not None),
                savefig=False,
                pred_fn=pred_fn,)

    if wandb.run is not None:
        wandb.finish()

    # small debug to confirm batch differs across runs
    sig = batch.xc[0, :5, 0].detach().cpu().tolist()
    print("Done.")
    print(f"  seed={seed} batch_seed={batch_seed}")
    print(f"  Nc={int(batch.xc.shape[1])} Nt={int(batch.xt.shape[1])}")
    print(f"  debug xc[:5]={sig}")


if __name__ == "__main__":
    main()
