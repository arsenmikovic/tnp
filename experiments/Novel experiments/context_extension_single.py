
import argparse
import copy
import os
import random
import time

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


@torch.no_grad()
def compute_nll(model, pred_fn, batch) -> float:
    dist = pred_fn(model, batch)
    return -(dist.log_prob(batch.yt).sum() / batch.yt[..., 0].numel()).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)

    # With ReversedContextGPGenerator, nt == nc (hard-coded), so set nc only.
    p.add_argument("--nc", type=int, required=True)

    p.add_argument("--use_val_generator", action="store_true")
    p.add_argument("--device", type=str, default="cuda")

    # optional: let user set a seed; if not provided, we generate a fresh one every run
    p.add_argument("--seed", type=int, default=-1)

    # Plot grid params
    p.add_argument("--x_min", type=float, default=-4.0)
    p.add_argument("--x_max", type=float, default=4.0)
    p.add_argument("--points_per_dim", type=int, default=64)

    # W&B
    p.add_argument("--wandb_project", type=str, default="gp-reversal")
    p.add_argument("--wandb_name", type=str, default="context-extension-single")
    p.add_argument("--wandb_dir", type=str, default="wandb")

    # Allow your config system args to follow after ours
    args, _ = p.parse_known_args()

    # Build experiment from YAML configs (this may set a fixed seed internally)
    experiment = initialize_experiment()
    pred_fn = experiment.misc.pred_fn
    model = experiment.model

    # --- IMPORTANT: force a new seed AFTER initialize_experiment so data differs each run ---
    if args.seed is None or args.seed < 0:
        seed = (int(time.time() * 1000) ^ (os.getpid() << 16) ^ random.getrandbits(32)) % (2**32)
    else:
        seed = int(args.seed) % (2**32)

    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Load trained weights into the model
    load_model_weights_from_litwrapper_ckpt(model, args.ckpt)
    model.eval()

    device = torch.device(args.device if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    model.to(device)

    # Pick generator (train or val)
    gen = experiment.generators.val if args.use_val_generator else experiment.generators.train

    # Make sure deterministic generator mode doesn't freeze samples (val has deterministic: True in your YAML)
    if hasattr(gen, "deterministic"):
        gen.deterministic = False

    # Force generator to produce exactly Nc=Nt=args.nc (no resplitting)
    if hasattr(gen, "min_nc"):
        gen.min_nc = args.nc
    if hasattr(gen, "max_nc"):
        gen.max_nc = args.nc
    if hasattr(gen, "min_nt"):
        gen.min_nt = args.nc
    if hasattr(gen, "max_nt"):
        gen.max_nt = args.nc

    # Generate EXACTLY one fresh batch
    batch = get_one_batch(gen)
    batch = to_device(copy.deepcopy(batch), device)

    # Use only first task (B=1) for plotting/eval (does NOT change split lengths)
    batch.xc, batch.yc, batch.xt, batch.yt = batch.xc[:1], batch.yc[:1], batch.xt[:1], batch.yt[:1]

    # keep only one target point test
    # batch.xt = batch.xt[:, :1, :]
    # batch.yt = batch.yt[:, :1, :]


    # Start W&B
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        dir=args.wandb_dir,
        config={
            "ckpt": args.ckpt,
            "seed_used": int(seed),
            "requested_nc": int(args.nc),
            "actual_nc": int(batch.xc.shape[1]),
            "actual_nt": int(batch.xt.shape[1]),
            "use_val_generator": bool(args.use_val_generator),
            "x_range": [float(args.x_min), float(args.x_max)],
            "points_per_dim": int(args.points_per_dim),
        },
    )

    # Log scalar + image plot
    nll = compute_nll(model, pred_fn, batch)
    wandb.log({"nll": nll})

    plot(
        model=model,
        batches=[batch],
        num_fig=1,
        x_range=(args.x_min, args.x_max),
        points_per_dim=args.points_per_dim,
        name=f"context_ext/nc_{args.nc}",
        logging=True,
        savefig=False,  # ensures no local fig saving
        pred_fn=pred_fn,
    )

    wandb.finish()
    print(
        f"Done. Logged to W&B. NLL={nll:.4f} | "
        f"Nc={batch.xc.shape[1]} Nt={batch.xt.shape[1]} | seed={seed}"
    )


if __name__ == "__main__":
    main()




# python experiments/context_extension_single.py \
#   --ckpt gp-reversal/jykzqh3g/checkpoints/epoch=499-step=1000000.ckpt \
#   --nc 128 \
#   --wandb_project gp-reversal \
#   --wandb_name ctx-ext-nc128-randseed \
#   --wandb_dir gp-reversal/jykzqh3g/wandb \
#   --config experiments/configs/models/tnp.yml \
#   experiments/configs/generators/gp-reversal.yml \

