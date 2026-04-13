import logging
import os
import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

HF_TOKEN         = os.environ["HF_TOKEN"]
WANDB_TOKEN      = os.environ["WANDB_API_KEY"]
DO_SPACES_KEY    = os.environ["DO_SPACES_KEY"]
DO_SPACES_SECRET = os.environ["DO_SPACES_SECRET"]

DATASET_PATH     = "s3://lerobot-datasets/so101_pickplace_tools_v3"
DO_ENDPOINT      = "https://atl1.digitaloceanspaces.com"

CHECKPOINT_REPO  = "ankithreddy/pi05-so101-finetune"
RUN_NAME         = "pi05-so101-pickplace"
RUN_STORAGE_PATH = os.path.abspath("./checkpoints")

CAMERA_RENAME = {
    "observation.images.images.wrist": "observation.images.left_wrist_0_rgb",
    "observation.images.images.top":   "observation.images.base_0_rgb",
}

# ── Set DO Spaces credentials before any S3 calls ─────────────────────────────

os.environ["AWS_ACCESS_KEY_ID"]     = DO_SPACES_KEY
os.environ["AWS_SECRET_ACCESS_KEY"] = DO_SPACES_SECRET
os.environ["AWS_ENDPOINT_URL"]      = DO_ENDPOINT

# ── Ray init ──────────────────────────────────────────────────────────────────

import ray

os.environ["PATH"] = f"/root/.local/bin:{os.environ['PATH']}"

ray.init(
    _temp_dir="/tmp/ray_tmp",
    object_store_memory=20 * 1024 ** 3,
    runtime_env={
        "py_executable": "/root/.local/bin/uv run",
        "working_dir": ".",
        "env_vars": {
            "HF_TOKEN":              HF_TOKEN,
            "WANDB_API_KEY":         WANDB_TOKEN,
            "PATH":                  f"/usr/bin:/usr/local/bin:/root/.local/bin:{os.environ['PATH']}",
            "LD_LIBRARY_PATH":       "/usr/lib/x86_64-linux-gnu:/usr/local/lib",
            "AWS_ACCESS_KEY_ID":     DO_SPACES_KEY,
            "AWS_SECRET_ACCESS_KEY": DO_SPACES_SECRET,
            "AWS_ENDPOINT_URL":      DO_ENDPOINT,
        },
    },
    ignore_reinit_error=True,
)

log.info(f"ray resources: {ray.cluster_resources()}")

# ── Ray Data pipeline ─────────────────────────────────────────────────────────

import ray.data
import util
from lerobot_datasource import LeRobotDatasource, Partitioning


def rename_columns(row: dict, rename: dict[str, str]) -> dict:
    """Rename dataset camera columns to match PI0.5's expected feature names."""
    return {rename.get(k, k): v for k, v in row.items()}


def transpose_images(batch: dict, camera_keys: list[str]) -> dict:
    """Convert camera images from HWC uint8 to CHW float32."""
    result = dict(batch)
    for key in camera_keys:
        result[key] = np.transpose(np.stack(batch[key]), (0, 3, 1, 2)).astype(np.float32)
    return result


source     = LeRobotDatasource(DATASET_PATH, partitioning=Partitioning.FILE_GROUP)
stats      = util.extract_stats(source)
image_keys = util.renamed_image_keys(source, CAMERA_RENAME)

ds = (
    ray.data
    .read_datasource(source)
    .map(rename_columns, fn_args=(CAMERA_RENAME,))
    .map_batches(transpose_images, batch_size=32, fn_args=(image_keys,))
)

log.info(f"pipeline ready — {source.meta.total_frames} frames, cameras: {image_keys}")

# ── Training loop ─────────────────────────────────────────────────────────────

def train_loop_per_worker(config: dict):
    import ray.train
    import ray.train.torch
    import wandb as wandb_lib
    from ray.air.integrations.wandb import setup_wandb
    from lerobot.policies.factory import make_pre_post_processors

    wandb_lib.finish()
    wandb = setup_wandb(
        config, project=RUN_NAME, reinit=True, name=f"{RUN_NAME}-{os.getpid()}"
    )

    device = torch.device("cuda")

    policy = util.load_pi05_policy()
    policy = ray.train.torch.prepare_model(policy)

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=config.get("lr", 1e-4),
    )
    scaler = torch.amp.GradScaler("cuda")

    checkpoint = ray.train.get_checkpoint()
    if checkpoint:
        start_epoch, step = util.load_checkpoint(checkpoint, policy, optimizer, scaler)
    else:
        start_epoch, step = 0, 0

    inner_policy = policy.module if hasattr(policy, "module") else policy
    inner_policy.config.normalization_mapping = {
        "ACTION": "MEAN_STD",
        "STATE":  "MEAN_STD",
        "VISUAL": "IDENTITY",
    }

    preprocessor, _ = make_pre_post_processors(
        inner_policy.config,
        pretrained_path="lerobot/pi05_base",
        dataset_stats=config["stats"],
    )

    batch_size  = int(config.get("batch_size", 2))
    grad_accum  = int(config.get("grad_accum", 2))
    num_epochs  = int(config.get("num_epochs", 2))
    max_len     = int(config.get("max_len", 512))
    num_workers = ray.train.get_context().get_world_size()
    scheduler   = util.build_lr_scheduler(optimizer, config, num_workers, last_step=step)

    shard = ray.train.get_dataset_shard("train")

    for epoch in range(start_epoch, num_epochs):
        optimizer.zero_grad(set_to_none=True)
        epoch_loss, epoch_count, accum_count = 0.0, 0, 0

        for batch in shard.iter_torch_batches(
            batch_size=batch_size,
            collate_fn=util.NumpyToTorchCollate(device),
        ):
            loss_val = util.train_step(policy, batch, preprocessor, max_len, grad_accum, scaler)
            step += 1
            accum_count += 1
            epoch_loss += loss_val
            epoch_count += 1

            if accum_count % grad_accum == 0:
                util.optimizer_step(policy, optimizer, scaler, scheduler)
                accum_count = 0

            if step % 10 == 0:
                log.info(f"epoch={epoch} step={step} loss={loss_val:.4f} lr={scheduler.get_last_lr()[0]:.2e}")
                wandb.log({"loss": loss_val, "lr": scheduler.get_last_lr()[0], "step": step})

        if accum_count > 0:
            util.optimizer_step(policy, optimizer, scaler, scheduler)

        avg_loss = epoch_loss / max(epoch_count, 1)
        metrics  = {"epoch": epoch, "step": step, "loss": avg_loss, "lr": scheduler.get_last_lr()[0]}
        wandb.log({"epoch_loss": avg_loss, "epoch": epoch})

        if ray.train.get_context().get_world_rank() == 0:
            checkpoint = util.make_checkpoint(policy, optimizer, scaler, epoch, step)
            ray.train.report(metrics, checkpoint=checkpoint)
        else:
            ray.train.report(metrics)

# ── Launch ────────────────────────────────────────────────────────────────────

import ray.train
import ray.train.torch

train_loop_config = {
    "stats":       stats,
    "total_rows":  source.meta.total_frames,
    "num_epochs":  3,
    "batch_size":  28,
    "grad_accum":  1,
    "lr":          1e-4,
    "warmup_frac": 0.1,
    "max_len":     512,
}

scaling_config = ray.train.ScalingConfig(
    num_workers=1,
    use_gpu=True,
)

run_config = ray.train.RunConfig(
    name=RUN_NAME,
    storage_path=RUN_STORAGE_PATH,
    failure_config=ray.train.FailureConfig(max_failures=1),
)

trainer = ray.train.torch.TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config=train_loop_config,
    scaling_config=scaling_config,
    run_config=run_config,
    datasets={"train": ds},
)

result = trainer.fit()
log.info(f"training complete: {result}")

# ── Push checkpoint to HuggingFace ────────────────────────────────────────────

if result.checkpoint:
    import ray.cloudpickle as pickle
    from lerobot.policies.pi05 import PI05Policy

    with result.checkpoint.as_directory() as ckpt_dir:
        with open(os.path.join(ckpt_dir, "state.pkl"), "rb") as f:
            state = pickle.load(f)

    # Load the trained weights into the policy and push directly
    policy = PI05Policy.from_pretrained("lerobot/pi05_base")
    policy.load_state_dict(state["model"], strict=False)
    policy.push_to_hub(CHECKPOINT_REPO, token=HF_TOKEN)
    log.info(f"checkpoint pushed to: https://huggingface.co/{CHECKPOINT_REPO}")