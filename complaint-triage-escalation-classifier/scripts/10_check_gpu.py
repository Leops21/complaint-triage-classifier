"""Check whether PyTorch can see and use the local GPU."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allows running this file directly without installing the package first.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from complaint_triage.utils.device import get_torch_device_summary


def main() -> None:
    summary = get_torch_device_summary()
    print(json.dumps(summary, indent=2))

    if not torch.cuda.is_available():
        print(
            "\nCUDA is not available to PyTorch. If this machine has an NVIDIA GPU, reinstall "
            "PyTorch using a CUDA-enabled wheel, then run this script again."
        )
        return

    device = torch.device("cuda")
    x = torch.randn((1024, 1024), device=device)
    y = x @ x
    torch.cuda.synchronize()
    print(
        f"\nGPU smoke test passed on {torch.cuda.get_device_name(0)}. "
        f"Result mean: {y.mean().item():.6f}"
    )


if __name__ == "__main__":
    main()
