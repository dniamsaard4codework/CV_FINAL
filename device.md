If your machine has multiple GPUs, torch.device('cuda') still works — but it will default to GPU 0 (cuda:0). To use multiple GPUs you either (1) pick a specific GPU, or (2) do multi-GPU training.

1) Check how many GPUs you have

import torch

if torch.cuda.is_available():
    n = torch.cuda.device_count()
    print("Num GPUs:", n)
    for i in range(n):
        print(i, torch.cuda.get_device_name(i))

2) Pick a specific GPU (single-GPU run)

device = torch.device("cuda:1")  # use 2nd GPU
model = model.to(device)

3) Use multiple GPUs (recommended way: DDP)

DistributedDataParallel (DDP) is the standard for performance + correctness.

Run with:

torchrun --nproc_per_node=NUM_GPUS train.py

Minimal train.py skeleton:

import os, torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

model = model.to(device)
model = DDP(model, device_ids=[local_rank])

4) Quick-and-dirty multi-GPU (often slower): DataParallel

model = torch.nn.DataParallel(model).cuda()

It’s simple, but DDP is usually faster and preferred.

5) Control which GPUs are visible

In your shell:

CUDA_VISIBLE_DEVICES=1,3 python train.py

Then inside your script, PyTorch sees those as cuda:0 and cuda:1.

If you tell me whether you’re doing training or inference, I can suggest the best pattern (DDP vs single GPU selection vs model-parallel).