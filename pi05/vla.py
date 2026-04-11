import logging
import os
import tempfile

import numpy as np
import util
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

HF_TOKEN    = os.environ["HF_TOKEN"]
WANDB_TOKEN = os.environ["WANDB_API_KEY"]

DATASET_ID   = "ankithreddy/so101_pickplace_tools"
REVISION     = "v3"
DATASET_PATH = f"hf://datasets/{DATASET_ID}@{REVISION}"

CHECKPOINT_REPO  = "ankithreddy/pi05-so101-pickplace"
RUN_NAME         = "pi05-so101-pickplace-finetune"
RUN_STORAGE_PATH = "/tmp/ray_runs"

# ============================================================================
# Connect to Ray
# ============================================================================

import ray

ray.init(
    _temp_dir="/tmp/ray_tmp",
    object_store_memory=20 * 1024 ** 3,
    runtime_env={
        "py_executable": "uv run",
        "working_dir": ".",
        "env_vars": {
            "HF_TOKEN":      HF_TOKEN,
            "WANDB_API_KEY": WANDB_TOKEN,
        },
    },
    ignore_reinit_error=True,
)

log.info("Ray cluster resources: %s", ray.cluster_resources())

# ============================================================================
# Ray Data — preprocessing pipeline
# ============================================================================

CAMERA_RENAME = {
    "observation.images.images.wrist": "observation.images.left_wrist_0_rgb",
    "observation.images.images.top":   "observation.images.base_0_rgb",
}

def rename_columns(row: dict, rename: dict[str, str]) -> dict:
    return {rename.get(k, k): v for k, v in row.items()}

def transpose_images(batch: dict, camera_keys: list[str]) -> dict:
    result = dict(batch)
    for key in camera_keys:
        result[key] = np.transpose(np.stack(batch[key]), (0, 3, 1, 2)).astype(np.float32)
    return result

from lerobot_datasource import LeRobotDatasource, Partitioning

source     = LeRobotDatasource(DATASET_PATH, partitioning=Partitioning.EPISODE)
stats      = util.extract_stats(source)
image_keys = util.renamed_image_keys(source, CAMERA_RENAME)

ds = (
    ray.data
    .read_datasource(source)
    .map(rename_columns, fn_args=(CAMERA_RENAME,))
    .map_batches(transpose_images, batch_size=8, fn_args=(image_keys,))
)

log.info("pipeline ready — %d frames, cameras: %s", source.meta.total_frames, image_keys)

# ============================================================================
# Ray Train — distributed training loop
# ============================================================================

import torch

def train_loop_per_worker(config: dict):
    import ray.train
    import ray.train.torch
    from ray.air.integrations.wandb import setup_wandb
    from lerobot.policies.factory import make_pre_post_processors

    wandb = setup_wandb(config, project=RUN_NAME)

    # Backbone frozen BEFORE DDP wrap — critical for float16 + GradScaler
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

    policy.module.config.normalization_mapping = {
        "ACTION": "MEAN_STD",
        "STATE":  "MEAN_STD",
        "VISUAL": "IDENTITY",
    }

    preprocessor, _ = make_pre_post_processors(
        policy.module.config,
        pretrained_path="lerobot/pi05_base",
        dataset_stats=config["stats"],
    )

    batch_size  = int(config.get("batch_size", 1))
    grad_accum  = int(config.get("grad_accum", 1))
    num_epochs  = int(config.get("num_epochs", 1))
    max_len     = int(config.get("max_len", 512))
    num_workers = ray.train.get_context().get_world_size()
    scheduler   = util.build_lr_scheduler(optimizer, config, num_workers, last_step=step)

    shard = ray.train.get_dataset_shard("train")

    for epoch in range(start_epoch, num_epochs):
        optimizer.zero_grad(set_to_none=True)
        epoch_loss_sum, epoch_loss_count = 0.0, 0
        accum_count = 0

        for batch in shard.iter_torch_batches(
            batch_size=batch_size,
            collate_fn=util.NumpyToTorchCollate(torch.device("cuda")),
        ):
            loss_val = util.train_step(policy, batch, preprocessor, max_len, grad_accum, scaler)
            step += 1
            accum_count += 1
            epoch_loss_sum += loss_val
            epoch_loss_count += 1

            if accum_count % grad_accum == 0:
                util.optimizer_step(policy, optimizer, scaler, scheduler)
                accum_count = 0

            if step % 10 == 0:
                log.info("epoch=%d  step=%d  loss=%.4f  lr=%.2e",
                         epoch, step, loss_val, scheduler.get_last_lr()[0])
                wandb.log({"loss": loss_val, "lr": scheduler.get_last_lr()[0], "step": step})

        if accum_count > 0:
            util.optimizer_step(policy, optimizer, scaler, scheduler)

        avg_loss = epoch_loss_sum / max(epoch_loss_count, 1)
        metrics  = {"epoch": epoch, "steps": step, "loss": avg_loss, "lr": scheduler.get_last_lr()[0]}
        wandb.log({"epoch_loss": avg_loss, "epoch": epoch})

        if ray.train.get_context().get_world_rank() == 0:
            checkpoint = util.make_checkpoint(policy, optimizer, scaler, epoch, step)
            ray.train.report(metrics, checkpoint=checkpoint)
        else:
            ray.train.report(metrics)

# ============================================================================
# Launch training
# ============================================================================

import ray.train
import ray.train.torch

train_loop_config = {
    "stats":       stats,
    "total_rows":  source.meta.total_frames,
    "num_epochs":  2,
    "batch_size":  4,
    "grad_accum":  2,
    "lr":          1e-4,
    "warmup_frac": 0.1,
    "max_len":     512,
}

scaling_config = ray.train.ScalingConfig(num_workers=2, use_gpu=True)

run_config = ray.train.RunConfig(
    name=RUN_NAME,
    storage_path=RUN_STORAGE_PATH,
    failure_config=ray.train.FailureConfig(max_failures=1),
)

data_config = ray.train.DataConfig(
    execution_options=ray.data.ExecutionOptions(
        resource_limits=ray.data.ExecutionResources(
            object_store_memory=10 * 1024 ** 3
        )
    )
)

trainer = ray.train.torch.TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config=train_loop_config,
    scaling_config=scaling_config,
    run_config=run_config,
    dataset_config=data_config,
    datasets={"train": ds},
)

result = trainer.fit()
log.info("Training complete: %s", result)

# ============================================================================
# Push checkpoint to HuggingFace
# ============================================================================

if result.checkpoint:
    import ray.cloudpickle as pickle
    from huggingface_hub import HfApi
    from lerobot.policies.pi05 import PI05Policy

    with result.checkpoint.as_directory() as ckpt_dir:
        with open(os.path.join(ckpt_dir, "state.pkl"), "rb") as f:
            state = pickle.load(f)

    # Load on CPU in float32 for saving — float16 not supported on CPU
    save_dir = tempfile.mkdtemp(prefix="pi05_hf_")
    policy = PI05Policy.from_pretrained(
        "lerobot/pi05_base",
        device="cpu",
        dtype=torch.float32,
    )
    policy.load_state_dict(state["model"])
    policy.save_pretrained(save_dir)

    api = HfApi(token=HF_TOKEN)
    api.create_repo(CHECKPOINT_REPO, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=save_dir,
        repo_id=CHECKPOINT_REPO,
        repo_type="model",
    )
    log.info("checkpoint pushed to: %s", CHECKPOINT_REPO)