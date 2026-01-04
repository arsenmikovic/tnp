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


def sort_targets_inplace(batch):
    """
    Sort xt (and permute yt) by x so progressive reveal (adding smallest-x targets first) is meaningful.
    Expects xt, yt of shape (B, Nt, 1).
    """
    xt = batch.xt
    yt = batch.yt
    xt_sorted, idx = torch.sort(xt, dim=1)      # (B, Nt, 1)
    yt_sorted = torch.gather(yt, 1, idx)        # align y with sorted x
    batch.xt, batch.yt = xt_sorted, yt_sorted
    return batch


def make_reveal_batch(base_batch, k: int):
    """
    Return a new batch where first k target points are appended to context,
    and remaining targets are the prediction targets.
    Assumes base_batch targets are already sorted by x.
    """
    b = copy.deepcopy(base_batch)

    if k > 0:
        b.xc = torch.cat([b.xc, b.xt[:, :k, :]], dim=1)
        b.yc = torch.cat([b.yc, b.yt[:, :k, :]], dim=1)

    b.xt = b.xt[:, k:, :]
    b.yt = b.yt[:, k:, :]

    return b

def make_no_context_batch(base_batch, k: int):
    """
    Return a new batch where first k target points are appended to context,
    and remaining targets are the prediction targets.
    Assumes base_batch targets are already sorted by x.
    """
    b = copy.deepcopy(base_batch)

    if k > 0:
        b.xc = b.xt[:, :k, :]
        b.yc = b.yt[:, :k, :]

    b.xt = b.xt[:, k:, :]
    b.yt = b.yt[:, k:, :]

    return b

def make_window_reveal_batch(base_batch, k: int, e: int):
    """
    Targets are already sorted.
    Append targets in the index window [k:e) to context,
    and predict on the remaining targets (i.e., all indices not in [k:e)).

    Shapes assumed: xc,yc ~ (B,Nc,1), xt,yt ~ (B,Nt,1)
    """
    b = copy.deepcopy(base_batch)
    nt = b.xt.shape[1]

    k = max(0, min(int(k), nt))
    e = max(k, min(int(e), nt))

    # window to append
    xt_win = b.xt[:, k:e, :]
    yt_win = b.yt[:, k:e, :]

    # remaining targets: [0:k) and [e:nt)
    xt_left = b.xt[:, :k, :]
    yt_left = b.yt[:, :k, :]
    xt_right = b.xt[:, e:, :]
    yt_right = b.yt[:, e:, :]

    # append window into context
    if xt_win.shape[1] > 0:
        b.xc = torch.cat([b.xc, xt_win], dim=1)
        b.yc = torch.cat([b.yc, yt_win], dim=1)

    # predict the rest
    b.xt = torch.cat([xt_left, xt_right], dim=1)
    b.yt = torch.cat([yt_left, yt_right], dim=1)

    return b



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)

    # With your reversal generator, nt == nc (by construction), so set nc only.
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
    p.add_argument("--wandb_name", type=str, default="progressive-reveal-10plots")
    p.add_argument("--wandb_dir", type=str, default="wandb")

    # Allow your config system args to follow after ours
    args, _ = p.parse_known_args()

    # Build experiment from YAML configs
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

    # Make sure deterministic generator mode doesn't freeze samples
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

    # Use only first task (B=1) for plotting/eval
    batch.xc, batch.yc, batch.xt, batch.yt = batch.xc[:1], batch.yc[:1], batch.xt[:1], batch.yt[:1]

    # Ensure targets sorted for progressive reveal
    batch = sort_targets_inplace(batch)

    nt = int(batch.xt.shape[1])
    if nt == 0:
        raise RuntimeError("Sampled batch has Nt=0; resample or adjust generator.")

    # optional: read reversal point if present
    r = None
    if hasattr(batch, "gt_pred") and hasattr(batch.gt_pred, "reversal_point"):
        r = float(batch.gt_pred.reversal_point)

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
            "reversal_point_r": r,
        },
    )

    # 10 plots: reveal 0%,10%,...,90% of targets into and context deletion
    for pct in range(0, 100, 20):
        frac = pct / 100.0
        k = int(frac * nt)
        k = max(0, min(k, nt - 1))  # keep >=1 target to predict

        b_reveal = make_no_context_batch(batch, k)

        nll = compute_nll(model, pred_fn, b_reveal)
        wandb.log({f"nll/reveal_{pct:02d}pct": nll})

        tag = f"reveal_{pct:02d}pct_nc{args.nc}_nt{nt}"
        if r is not None:
            tag += f"_r{r:.3f}"

        plot(
            model=model,
            batches=[b_reveal],
            num_fig=1,
            x_range=(args.x_min, args.x_max),
            points_per_dim=args.points_per_dim,
            name=f"progressive/{tag}",
            logging=True,
            savefig=False,
            pred_fn=pred_fn,
        )

    for pct in range(0, 100, 20):
        frac = pct / 100.0
        k = int(frac * nt)
        k = max(0, min(k, nt - 1))  # keep >=1 target to predict

        b_reveal = make_reveal_batch(batch, k)

        nll = compute_nll(model, pred_fn, b_reveal)
        wandb.log({f"nll/reveal_{pct:02d}pct": nll})

        tag = f"reveal_{pct:02d}pct_nc{args.nc}_nt{nt}"
        if r is not None:
            tag += f"_r{r:.3f}"

        plot(
            model=model,
            batches=[b_reveal],
            num_fig=1,
            x_range=(args.x_min, args.x_max),
            points_per_dim=args.points_per_dim,
            name=f"progressive/{tag}",
            logging=True,
            savefig=False,
            pred_fn=pred_fn,
        )

    k, e = 2 * nt // 5, 3 * nt // 5
    b_win = make_window_reveal_batch(batch, k, e)

    nll_win = compute_nll(model, pred_fn, b_win)
    wandb.log({"nll/window": nll_win})

    tag_win = f"window_k{k}_e{e}_nc{args.nc}_nt{nt}"
    if r is not None:
        tag_win += f"_r{r:.3f}"

    plot(
        model=model,
        batches=[b_win],
        num_fig=1,
        x_range=(args.x_min, args.x_max),
        points_per_dim=args.points_per_dim,
        name=f"progressive/{tag_win}",
        logging=True,
        savefig=False,
        pred_fn=pred_fn,
    )

    wandb.finish()
    print(
        f"Done. Logged 10 progressive plots to W&B. "
        f"Nc={batch.xc.shape[1]} Nt={batch.xt.shape[1]} | seed={seed}"
    )



if __name__ == "__main__":
    main()


# Example:
# python experiments/reversal_inference_AR.py \
#   --ckpt gp-reversal-inference/2ypx1r6i/checkpoints/epoch=499-step=1000000.ckpt \
#   --nc 64 \
#   --wandb_project gp-reversal-inference \
#   --wandb_name progressive-reveal-nc128 \
#   --wandb_dir gp-reversal-inference/2ypx1r6i/wandb \
#   --device cuda \
#   --config experiments/configs/models/tnp.yml \
#   experiments/configs/generators/gp-random-reversal.yml
