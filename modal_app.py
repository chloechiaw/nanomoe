"""
Modal harness for nanoMoE: single-GPU (A10G) training of the nanochat-derived MoE model.

Everything stateful (dataset shards, tokenizer, eval bundle, checkpoints, HF kernel cache)
lives on a single Modal Volume mounted at /data, so runs are resumable and the expensive
prepare steps happen exactly once.

Typical first-time flow:

    modal run modal_app.py::probe                      # ~1 min : is the GPU stack sane?
    modal run modal_app.py::prepare --shards 32        # ~20 min: data + tokenizer + eval bundle
    modal run modal_app.py::smoke                      # ~10 min: 30 real training steps, get tok/sec
    modal run modal_app.py::train --args "--depth=8"   # the real run

See README.md for the annotated version of the above.
"""

import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

# -----------------------------------------------------------------------------
# Constants

APP_NAME = "nano-moe"
# 24GB, GA102/sm_86, no FP8. Note: this tier has been observed to schedule a plain A10
# (72 SM, 125 TFLOPS dense bf16) rather than an A10G (80 SM, 140). `probe` reports which.
GPU_TYPE = "A10G"
REPO_ROOT = Path(__file__).parent
REMOTE_REPO = "/root/nano-moe"
VOLUME_NAME = "nano-moe-data"
DATA_ROOT = "/data"
NANOCHAT_BASE_DIR = f"{DATA_ROOT}/nanochat"  # nanochat writes all its artifacts here

HOURS = 60 * 60

# -----------------------------------------------------------------------------
# Image
#
# torch 2.9.1 from PyPI already ships the CUDA 12.8 runtime for linux/x86_64, so there is no
# need for the pytorch wheel index. build-essential is required by torch.compile's inductor
# backend, which shells out to a C++ compiler for the generated wrapper code.

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install("torch==2.9.1")
    .pip_install(
        # nanochat's own dependency set (pyproject.toml), plus the two it uses but does not
        # declare: requests (nanochat/dataset.py) and pyyaml (scripts/base_eval.py).
        "filelock>=3.19.0",
        "kernels>=0.11.7",
        "numpy>=1.26.0",
        "psutil>=7.1.0",
        "pyarrow>=21.0.0",
        "requests>=2.32.0",
        "rustbpe>=0.1.0",
        "tiktoken>=0.11.0",
        "wandb>=0.21.3",
        "pyyaml>=6.0",
        "matplotlib>=3.10.0",
        "pytest>=8.0.0",
    )
    .env(
        {
            "NANOCHAT_BASE_DIR": NANOCHAT_BASE_DIR,
            # keep the HF kernel cache (flash-attn3) on the volume so we download it once
            "HF_HOME": f"{DATA_ROOT}/hf",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "OMP_NUM_THREADS": "1",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_dir(
        REPO_ROOT,
        REMOTE_REPO,
        ignore=[
            "**/.git",
            "**/.venv",
            "**/__pycache__",
            "**/*.pyc",
            "**/.DS_Store",
            "**/dev/*.ipynb",
        ],
    )
)

app = modal.App(APP_NAME, image=image)

data_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
VOLUMES = {DATA_ROOT: data_vol}

# Pass a local WANDB_API_KEY through if one is set; otherwise this is an empty secret and
# wandb stays disabled (base_train's --run defaults to "dummy", which skips wandb entirely).
SECRETS = [modal.Secret.from_dict({"WANDB_API_KEY": os.environ.get("WANDB_API_KEY", "")})]


# -----------------------------------------------------------------------------
# Helpers (run inside the container)


def _run(*argv: str) -> None:
    """Run a repo command, streaming output, raising on nonzero exit."""
    cmd = [sys.executable, "-u", *argv]
    print(f"\n$ {shlex.join(cmd)}\n", flush=True)
    subprocess.run(cmd, cwd=REMOTE_REPO, check=True)


class _PeriodicCommit:
    """Commit the volume every `interval` seconds so a killed multi-hour run keeps its
    checkpoints. Modal only auto-commits when the function returns normally."""

    def __init__(self, interval: int = 10 * 60):
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.wait(self.interval):
            try:
                data_vol.commit()
                print(f"[modal] volume committed at {time.strftime('%H:%M:%S')}", flush=True)
            except Exception as e:  # a failed mid-run commit must never kill training
                print(f"[modal] volume commit failed (continuing): {e}", flush=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        data_vol.commit()


def _merge_args(defaults: list[str], overrides: str) -> list[str]:
    """argparse takes the last occurrence of a flag, so appending user args over the
    defaults gives override semantics for free."""
    return defaults + shlex.split(overrides)


# -----------------------------------------------------------------------------
# Diagnostics


@app.function(gpu=GPU_TYPE, volumes=VOLUMES, timeout=30 * 60)
def probe():
    """Report what the GPU stack actually resolved to. Needs no data — run this first."""
    import torch

    sys.path.insert(0, REMOTE_REPO)
    os.chdir(REMOTE_REPO)

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability()
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

    from nanochat.common import COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, get_peak_flops
    from nanochat.flash_attention import HAS_FA3, USE_FA3

    peak = get_peak_flops(name)

    print("=" * 70)
    print(f"GPU              : {name}  (sm_{cap[0]}{cap[1]}, {total_gb:.1f} GiB)")
    print(f"torch            : {torch.__version__}  cuda {torch.version.cuda}")
    print(f"compute dtype    : {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
    print(f"peak bf16 FLOPS  : {peak:.3e}")
    print(f"FA3 available    : {HAS_FA3}")
    print(f"FA3 in use       : {USE_FA3}")
    print("=" * 70)

    if not USE_FA3:
        print(
            "NOTE: falling back to PyTorch SDPA. SDPA has no fused sliding-window support, so\n"
            "      pass --window-pattern=L (full context on every layer) or utilization will\n"
            "      be poor. With FA3 active, the default SSSL pattern is the faster choice."
        )

    # measured bf16 matmul throughput, as a reality check on the datasheet number
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    t0 = time.time()
    iters = 30
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    measured = iters * 2 * 8192**3 / (time.time() - t0)
    print(f"measured 8k bf16 matmul: {measured:.3e} FLOP/s ({100 * measured / peak:.1f}% of peak)")

    return {"gpu": name, "sm": f"{cap[0]}{cap[1]}", "fa3": USE_FA3, "peak_flops": peak}


# -----------------------------------------------------------------------------
# Data preparation (CPU only)


@app.function(cpu=4.0, memory=8192, timeout=30 * 60)
def test(args: str = ""):
    """Run the MoE / Quantile Balancing test suite. CPU only, so effectively free — run this
    before spending GPU time on a bench or a training run."""
    _run("-m", "pytest", *(shlex.split(args) or ["tests/test_moe.py", "-v"]))


@app.function(gpu=GPU_TYPE, volumes=VOLUMES, timeout=1 * HOURS)
def bench(args: str = ""):
    """Warm up, then time training steps on random tokens and project against the budget.
    Needs no dataset, so it runs standalone.

        modal run modal_app.py::bench --args "--n-expert 8 --top-k 2"
        modal run modal_app.py::bench --args "--n-expert 1"     # dense, for the MoE/dense ratio
    """
    argv = _merge_args(BENCH_DEFAULTS, args)
    _run("-m", "scripts.bench_step", *argv)


@app.function(cpu=8.0, memory=16384, volumes=VOLUMES, timeout=4 * HOURS)
def download_data(shards: int = 32, workers: int = 16):
    """Download `shards` ClimbMix train shards (+ the pinned val shard) to the volume.

    Sizing: each shard is ~250M chars ≈ 60M tokens at nanochat's default vocab. The BOS
    best-fit dataloader discards ~35% of tokens to cropping, so budget ~1.5x. 32 shards is
    ~2B usable tokens, comfortably above the 500M–1B target with no epoch repeats.
    """
    _run("-m", "nanochat.dataset", "-n", str(shards), "-w", str(workers))
    data_vol.commit()


@app.function(cpu=16.0, memory=65536, volumes=VOLUMES, timeout=4 * HOURS)
def train_tokenizer(vocab_size: int = 32768, max_chars: int = 2_000_000_000):
    """Train the BPE tokenizer and cache token_bytes (needed for the vocab-invariant bpb metric).

    See BRIEF.md "Open questions": at a ~40M-param target the default 32768 vocab puts more
    params in wte+lm_head than in the MoE trunk. --vocab-size 8192 is the alternative.
    """
    _run("-m", "scripts.tok_train", f"--vocab-size={vocab_size}", f"--max-chars={max_chars}")
    _run("-m", "scripts.tok_eval")
    data_vol.commit()


@app.function(cpu=4.0, volumes=VOLUMES, timeout=1 * HOURS)
def download_eval_bundle():
    """Prefetch the CORE eval bundle so the first in-training CORE eval doesn't stall."""
    sys.path.insert(0, REMOTE_REPO)
    os.chdir(REMOTE_REPO)
    from nanochat.common import download_file_with_lock
    from scripts.base_eval import EVAL_BUNDLE_URL, place_eval_bundle

    download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)
    data_vol.commit()


# -----------------------------------------------------------------------------
# Training
#
# A10G defaults, and why:
#   --device-batch-size=8    24 GB is the binding constraint, and the logit tensor dominates:
#                            base_train materialises B*T*vocab in bf16, again in fp32, again
#                            after the tanh softcap, and again inside cross_entropy. At B=8,
#                            T=2048, vocab=32768 that chain alone is ~7.5 GiB. Drop to 4 if you
#                            OOM; a smaller --vocab-size at tokenizer time helps a lot here.
#                            MoE will add pressure too: 8x the FFN weights and 8x their
#                            optimizer state stay resident even though only top-2 are active.
#   --window-pattern=L       safe under the SDPA fallback. If `probe` reports FA3 in use, SSSL
#                            is faster — pass it explicitly.
#   (no --fp8)               sm_86 has no FP8 tensor cores.
#   --total-batch-size       left on auto (-1); with world_size=1 it is reached purely by
#                            gradient accumulation.

# --depth/--aspect-ratio/--head-dim give d_model = 8 * 32 = 256, n_head = 4: the config
# scripts/size_moe.py picks out as hitting the brief's ~40M total / ~15M active target.
# Leaving aspect-ratio at nanochat's default 64 would build d_model=512, a 243M-param model.
# --device-batch-size=8 is the value that is safe at either vocab size; at --vocab-size=8192
# raise it to 32 for ~25% more throughput (17.1 of 22.1 GiB, measured — see README).
TRAIN_DEFAULTS = [
    "--depth=8",
    "--aspect-ratio=32",
    "--head-dim=64",
    "--device-batch-size=8",
    "--window-pattern=L",
    "--n-expert=8",
    "--top-k=2",
    "--eval-every=250",
    "--core-metric-every=2000",
    "--sample-every=2000",
    "--save-every=1000",
    "--expert-load-every=100",
]

# bench_step.py takes the same architecture flags but drives them with random tokens
BENCH_DEFAULTS = [
    "--depth=8",
    "--aspect-ratio=32",
    "--head-dim=64",
    "--device-batch-size=8",
    "--window-pattern=L",
    "--n-expert=8",
    "--top-k=2",
    "--grad-accum=4",
    "--warmup=6",
    "--steps=20",
]


@app.function(
    gpu=GPU_TYPE,
    volumes=VOLUMES,
    secrets=SECRETS,
    timeout=8 * HOURS,  # the brief's budget is 5-7h; Modal's ceiling is 24h
)
def train(args: str = "", run: str = "dummy"):
    """Pretrain. `args` is a raw arg string forwarded to scripts.base_train, overriding
    TRAIN_DEFAULTS (argparse keeps the last occurrence of a flag).

        modal run modal_app.py::train --args "--depth=10 --device-batch-size=8"
    """
    argv = _merge_args(TRAIN_DEFAULTS, args) + [f"--run={run}"]
    with _PeriodicCommit():
        _run("-m", "scripts.base_train", *argv)


@app.function(gpu=GPU_TYPE, volumes=VOLUMES, timeout=1 * HOURS)
def smoke(args: str = ""):
    """A short but *real* training run: validates the full stack and, more importantly,
    measures tok/sec so you can size the token budget for a 5-7 hour run."""
    argv = _merge_args(
        [
            "--depth=8",
            "--device-batch-size=8",
            "--window-pattern=L",
            "--num-iterations=30",
            "--total-batch-size=32768",
            "--eval-every=-1",
            "--core-metric-every=-1",
            "--sample-every=-1",
            "--model-tag=smoke",
        ],
        args,
    )
    _run("-m", "scripts.base_train", *argv)
    print(
        "\nRead `tok/sec` off the last few steps above (ignore the first ~10, they include\n"
        "torch.compile warmup), then: tokens_in_6h = tok_per_sec * 21600."
    )


@app.function(gpu=GPU_TYPE, volumes=VOLUMES, timeout=4 * HOURS)
def evaluate(args: str = ""):
    """CORE metric + train/val bpb + samples for a saved checkpoint.

        modal run modal_app.py::evaluate --args "--model-tag=d8"
    """
    argv = _merge_args(["--device-batch-size=8"], args)
    _run("-m", "scripts.base_eval", *argv)


# -----------------------------------------------------------------------------
# Volume utilities


@app.function(volumes=VOLUMES, timeout=15 * 60)
def ls(path: str = ""):
    """List what's on the volume (checkpoints, shards, tokenizer)."""
    target = os.path.join(NANOCHAT_BASE_DIR, path)
    subprocess.run(["du", "-sh", target], check=False)
    subprocess.run(["ls", "-la", target], check=False)


# To pull a checkpoint down to your laptop, use the volume CLI rather than a function:
#   modal volume ls   nano-moe-data nanochat/base_checkpoints
#   modal volume get  nano-moe-data nanochat/base_checkpoints/d8 ./checkpoints


# -----------------------------------------------------------------------------
# Orchestration


@app.local_entrypoint()
def prepare(shards: int = 32, vocab_size: int = 32768, max_chars: int = 2_000_000_000):
    """One-time setup: shards -> tokenizer -> eval bundle. Idempotent; safe to re-run.

        modal run modal_app.py::prepare --shards 32 --vocab-size 8192
    """
    print(f"[1/3] downloading {shards} data shards ...")
    download_data.remote(shards=shards)
    print(f"[2/3] training tokenizer (vocab_size={vocab_size}) ...")
    train_tokenizer.remote(vocab_size=vocab_size, max_chars=max_chars)
    print("[3/3] fetching CORE eval bundle ...")
    download_eval_bundle.remote()
    print("done. next: modal run modal_app.py::smoke")
