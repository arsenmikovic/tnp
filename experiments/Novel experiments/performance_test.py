#!/usr/bin/env python3
"""
eval_nll_stats.py

Sample batches from a generator and compute NLL statistics.

Example:
python experiments/performance_test.py \
  --ckpt gp-reversal/jykzqh3g/checkpoints/epoch=499-step=1000000.ckpt \
  --num_batches 200 \
  --use_val_generator \
  --device cuda \
  --force_n 64 \
  --config experiments/configs/models/tnp.yml experiments/configs/generators/gp-random-reversal.yml
"""

import argparse
import copy
import os
import random
import time
from typing import Optional

import lightning.pytorch as pl
import torch

from tnp.utils.experiment_utils import initialize_experiment


# ---------------------------
# utilities
# ---------------------------

def get_one_batch(gen):
    # Prefer generate_batch() to avoid iterator caching behavior
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
            setattr(gen, attr, int(n))


def load_model_weights_from_litwrapper_ckpt(model: torch.nn.Module, ckpt_path: str):
    """
    Loads weights saved from a LightningModule wrapper that stores model under 'model.*'
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    model_sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    if not model_sd:
        print("ERROR: couldn't find keys starting with 'model.' in checkpoint.")
        print("Some checkpoint keys:", list(sd.keys())[:40])
        raise SystemExit(1)

    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    if missing or unexpected:
        print("Warning: load_state_dict had differences.")
        print("Missing:", missing)
        print("Unexpected:", unexpected)


@torch.no_grad()
def batch_nll(pred_fn, model, batch) -> float:
    """
    Returns average NLL per target scalar in this batch.
    Works for yt shaped (B, Nt, 1) and dist.log_prob shaped compatibly.
    """
    dist = pred_fn(model, batch)              # distribution over y at xt
    lp = dist.log_prob(batch.yt)              # log prob tensor

    # sum all logprobs and normalize by number of target scalars
    denom = batch.yt[..., 0].numel()
    nll = -(lp.sum() / denom).item()
    return nll


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--num_batches", type=int, default=100)

    p.add_argument("--use_val_generator", action="store_true")
    p.add_argument("--device", type=str, default="cuda")

    # If -1: random seed each run. If >=0: fixed seed (reproducible batches).
    p.add_argument("--seed", type=int, default=-1)

    # Optional: force Nc=Nt=N for generators that support min/max nc/nt.
    p.add_argument("--force_n", type=int, default=-1)

    # Optional: if generator has deterministic=True in val, you can override it.
    # Default behavior: DO NOT touch it (matches your config).
    p.add_argument("--set_deterministic", type=str, default="leave", choices=["leave", "true", "false"])

    # Allow your config system args to follow after ours
    args, _ = p.parse_known_args()

    # Build experiment from YAML configs (initialize_experiment reads CLI args)
    experiment = initialize_experiment()
    pred_fn = experiment.misc.pred_fn
    model = experiment.model

    # Seed
    if args.seed is None or args.seed < 0:
        seed = (int(time.time() * 1000) ^ (os.getpid() << 16) ^ random.getrandbits(32)) % (2**32)
    else:
        seed = int(args.seed) % (2**32)

    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Load checkpoint
    load_model_weights_from_litwrapper_ckpt(model, args.ckpt)
    model.eval()

    device = torch.device(args.device if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    model.to(device)

    # Pick generator
    gen = experiment.generators.val if args.use_val_generator else experiment.generators.train

    # Optional deterministic override
    if hasattr(gen, "deterministic"):
        if args.set_deterministic == "true":
            gen.deterministic = True
        elif args.set_deterministic == "false":
            gen.deterministic = False
        # else: leave

    # Optional force nc/nt
    if args.force_n is not None and args.force_n > 0:
        force_nc_nt(gen, args.force_n)

    # Compute NLLs
    nlls = []
    for i in range(args.num_batches):
        b = get_one_batch(gen)
        b = to_device(copy.deepcopy(b), device)

        nll = batch_nll(pred_fn, model, b)
        nlls.append(nll)

        if (i + 1) % max(1, args.num_batches // 10) == 0:
            print(f"[{i+1}/{args.num_batches}] nll={nll:.6f}")

    nlls_t = torch.tensor(nlls, dtype=torch.float64)
    mean = nlls_t.mean().item()
    var = nlls_t.var(unbiased=True).item() if len(nlls) > 1 else 0.0
    std = var ** 0.5

    print("\n=== NLL stats ===")
    print(f"seed: {seed}")
    print(f"num_batches: {args.num_batches}")
    print(f"mean_nll: {mean:.6f}")
    print(f"var_nll:  {var:.6f}")
    print(f"std_nll:  {std:.6f}")


if __name__ == "__main__":
    main()
