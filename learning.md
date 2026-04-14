Batch size = how many samples the model sees in one forward pass 
Grad accumulation = how many batches to run before actually updating weights — lets you simulate a larger batch without needing more GPU memory (set to 1 here, so weights update every batch)
Here's a clean learning breakdown:

---

**DDP (Distributed Data Parallel)**
- Runs same model on N GPUs simultaneously
- Each GPU gets different data shard, computes gradients independently
- Gradients are averaged across all GPUs before weight update
- `prepare_model()` wraps your model in DDP automatically

**Mixed Precision Training**
- Runs forward pass in float16 (faster, less memory)
- GradScaler scales loss up before backward to avoid float16 underflow, then scales gradients back down before optimizer step

**Gradient Accumulation**
- Simulates large batch by running multiple small batches before updating weights
- Effective batch = `batch_size × grad_accum × num_workers`

**Data Sharding**
- Each worker gets a non-overlapping slice of the full dataset
- Prevents duplicate training on same samples across GPUs

**Rank-0 Pattern**
- In DDP, only worker with rank=0 handles logging, checkpointing, pushing to HuggingFace
- All other workers just train

**LR Warmup**
- Starts learning rate very small, ramps up over first N steps, then decays
- Prevents large destructive updates at the start of finetuning transformers

**Checkpoint Resume**
- Saves optimizer state, scaler state, epoch, step — not just model weights
- Allows training to resume exactly where it left off after crash