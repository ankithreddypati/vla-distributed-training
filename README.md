# vla-distributed-training
Distributed fine-tuning of VLA models (PI0.5, GR00T N1.5) on custom SO-101 pick-and-place data using Ray Train. Cloud-agnostic infrastructure provisioned with Pulumi runs on RunPod, DigitalOcean, or any GPU cloud

Each GPU gets a copy of the data batch or a shard of the model, runs the forward pass, computes gradients, then NCCL synchronizes gradients across all GPUs via all-reduce, and the optimizer updates the weights. Frameworks like FSDP or DeepSpeed handle the memory optimization so you can train models too large to fit on a single GPU



nebuis , runpod and Digital ocean - terraform