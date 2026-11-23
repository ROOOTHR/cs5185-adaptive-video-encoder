"""Small helper executed only by the dedicated CPU PyTorch alignment venv."""
from __future__ import annotations
import json
import sys
import numpy as np
import torch
import importlib.metadata
from pytorch_msssim import ms_ssim

source, output = sys.argv[1:3]
torch.set_num_threads(1)
data = np.load(source)
values = []
for a, b in zip(data["reference"], data["distorted"]):
    x = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None, None]
    y = torch.from_numpy(np.ascontiguousarray(b, dtype=np.float32))[None, None]
    values.append(float(ms_ssim(x, y, data_range=255, size_average=True).item()))
with open(output, "w", encoding="utf-8") as stream:
    json.dump({"values": values, "torch": torch.__version__, "pytorch_msssim": importlib.metadata.version("pytorch-msssim")}, stream)
