"""检查Python、PyTorch、timm和计算设备。"""

import importlib
import sys


def main():
    print("Python", sys.version)
    for name in [
        "torch",
        "torchvision",
        "timm",
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "PIL",
        "matplotlib",
        "seaborn",
    ]:
        m = importlib.import_module(name)
        print(name, getattr(m, "__version__", "unknown"))
    import torch

    print("CUDA available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()
