"""
Modal harness for nanoMoE: single-GPU (H100) training of the nanochat-derived MoE model.

Everything stateful (dataset shards, tokenizer, eval bundle, checkpoints, HF kernel cache)
lives on a single Modal Volume mounted at /data, so runs are resumable and the expensive
prepare steps happen exactly once.

Typical first-time flow:

    modal run modal_app.py::prepare --shards 32        # ~20 min: data + tokenizer + eval bundle
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
# 80GB, sm_90. torch._grouped_mm needs a Hopper-class card, so this is the supported target.
# Override with NANOMOE_GPU if you have something else.
GPU_TYPE = os.environ.get("NANOMOE_GPU", "H100")
REPO_ROOT = Path(__file__).parent
REMOTE_REPO = "/root/nano-moe"
VOLUME_NAME = "nano-moe-data"
DATA_ROOT = "/data"
NANOCHAT_BASE_DIR = f"{DATA_ROOT}/nanochat"  # nanochat writes all its artifacts here

HOURS = 60 * 60

# -----------------------------------------------------------------------------
# Image
#
# torch 2.13 from the cu129 wheel index. Measured slightly faster than 2.9.1+cu128
# (28.6% vs 28.0% MFU at the training config) and FA3 still loads.
# build-essential is required by torch.compile's inductor backend, which shells out to a C++
# compiler for the generated wrapper code.

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install("torch==2.13.0", index_url="https://download.pytorch.org/whl/cu129")
    .pip_install(
        # pyproject.toml's set, plus the two it uses but does not declare:
        # requests (nanomoe/dataset.py) and pyyaml (scripts/base_eval.py).
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
        "pytest>=8.0.0",
        "quack-kernels>=0.6.4",
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


@app.function(cpu=4.0, memory=8192, timeout=30 * 60)
def test(args: str = ""):
    """Run the MoE / Quantile Balancing test suite. CPU only, so effectively free — run this
    before spending GPU time on a training run."""
    _run("-m", "pytest", *(shlex.split(args) or ["tests/test_moe.py", "-v"]))


@app.function(cpu=8.0, memory=16384, volumes=VOLUMES, timeout=4 * HOURS)
def download_data(shards: int = 32, workers: int = 16):
    """Download `shards` ClimbMix train shards (+ the pinned val shard) to the volume.

    Sizing: each shard is ~250M chars ≈ 60M tokens at nanochat's default vocab. The BOS
    best-fit dataloader discards ~35% of tokens to cropping, so budget ~1.5x. 32 shards is
    ~2B usable tokens, comfortably above the 500M–1B target with no epoch repeats.
    """
    _run("-m", "nanomoe.dataset", "-n", str(shards), "-w", str(workers))
    data_vol.commit()


@app.function(cpu=16.0, memory=65536, volumes=VOLUMES, timeout=4 * HOURS)
def train_tokenizer(vocab_size: int = 32768, max_chars: int = 2_000_000_000):
    """Train the BPE tokenizer and cache token_bytes (needed for the vocab-invariant bpb metric).

    See BRIEF.md "Open questions": at a ~40M-param target the default 32768 vocab puts more
    params in wte+lm_head than in the MoE trunk. --vocab-size 8192 is the alternative.
    """
    _run("-m", "scripts.tok_train", f"--vocab-size={vocab_size}", f"--max-chars={max_chars}")
    data_vol.commit()


@app.function(cpu=4.0, volumes=VOLUMES, timeout=1 * HOURS)
def download_eval_bundle():
    """Prefetch the CORE eval bundle so the first in-training CORE eval doesn't stall."""
    sys.path.insert(0, REMOTE_REPO)
    os.chdir(REMOTE_REPO)
    from nanomoe.common import download_file_with_lock
    from scripts.base_eval import EVAL_BUNDLE_URL, place_eval_bundle

    download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)
    data_vol.commit()


# -----------------------------------------------------------------------------
# Training
#
# H100 defaults: these are exactly the flags that trained moe-d16-h100 (val_bpb 0.7822,
# CORE 0.1901) in 5.1 hours, so a bare `modal run ::train` reproduces that run.
#
#   --depth=16 --aspect-ratio=40   d_model 640, the compute-optimal size for a 5 hour budget
#   --device-batch-size=32         73.8 of 79.2 GiB; 40 and above OOM
#   --window-pattern=L             full context; no FA3 build is published for this stack
#   --num-iterations=7350          3.85B tokens at total-batch 524,288
#
TRAIN_DEFAULTS = [
    "--depth=16",
    "--aspect-ratio=40",
    "--head-dim=64",
    "--window-pattern=L",
    "--n-expert=8",
    "--top-k=2",
    "--device-batch-size=32",
    "--num-iterations=7350",
    "--eval-every=500",
    "--eval-tokens=10485760",
    "--core-metric-every=1500",
    "--core-metric-max-per-task=200",
    "--expert-load-every=200",
    "--save-every=2500",
]



@app.function(gpu=GPU_TYPE, volumes=VOLUMES, timeout=30 * 60)
def smoke(args: str = ""):
    """A few real training steps. Cheap check that the whole stack still runs before
    committing to a long run: data, tokenizer, FA3, grouped dispatch, QB, checkpointing."""
    argv = _merge_args(
        TRAIN_DEFAULTS
        + [
            "--num-iterations=12",
            "--total-batch-size=65536",
            "--device-batch-size=8",
            "--eval-every=-1",
            "--core-metric-every=-1",
            "--save-every=-1",
            "--model-tag=smoke",
        ],
        args,
    )
    _run("-m", "scripts.base_train", *argv)


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
    print("done. next: modal run --detach modal_app.py::train")
