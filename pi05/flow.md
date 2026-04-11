
Your HF Dataset (v3 format)
        ↓
StreamingLeRobotDataset   ← LeRobot handles decode, parquet, timestamps
        ↓
ray.data.from_iterable()  ← thin wrapper, that's it
        ↓
.map_batches(transforms)  ← HWC→CHW, normalize, rename cameras
        ↓
Ray Train TorchTrainer    ← DDP, NCCL, checkpointing
        ↓
Your model (PI0.5 or whatever)
