"""使用随机输入和随机初始化模型测量参数量、FLOPs和前向耗时。"""

import argparse
import statistics
import tempfile
import time
from pathlib import Path

import torch

from bca_net.model import BCAStudent
from common.config import load_config
from common.runtime import device_for, json_write, seed_all


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--iterations", type=int, default=100)
    a = p.parse_args(argv)
    if a.iterations < 2 or a.warmup < 0:
        raise ValueError("Need >=2 timing iterations and nonnegative warmup")
    cfg = load_config(a.config)
    seed_all(42, cfg)
    device = device_for(cfg)
    model = BCAStudent(cfg, pretrained=False).to(device).eval()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    with tempfile.TemporaryDirectory() as folder:
        weight = Path(folder) / "model.pt"
        torch.save(model.state_dict(), weight)
        file_bytes = weight.stat().st_size
    report = {
        "scope": "student forward pass with random inputs and initialization",
        "config": cfg.dict(),
        "device": str(device),
        "torch": str(torch.__version__),
        "parameters": params,
        "trainable_parameters": trainable,
        "state_dict_bytes": file_bytes,
        "registered_but_unused": "lat_c4 and the deepest backbone output",
        "timing_sd": "SD over repeated forward timing calls",
        "batches": [],
    }
    for batch in [1, 16]:
        image = torch.randn(batch, 3, cfg.image_size, cfg.image_size, device=device)
        nir = torch.randn(batch, cfg.nir_dim, device=device)
        with torch.inference_mode():
            for _ in range(a.warmup):
                model(image, nir)
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(a.iterations):
                if device.type == "cuda":
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    model(image, nir)
                    end.record()
                    end.synchronize()
                    elapsed = start.elapsed_time(end)
                else:
                    start = time.perf_counter()
                    model(image, nir)
                    elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
        mean = statistics.mean(times)
        row = {
            "batch": batch,
            "latency_ms_per_batch": mean,
            "latency_sd_ms": statistics.stdev(times),
            "throughput_samples_s": batch * 1000 / mean,
            "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20
            if device.type == "cuda"
            else None,
        }
        # 使用PyTorch FLOP计数器，分别记录每批次与每样本FLOPs。
        try:
            from torch.utils.flop_counter import FlopCounterMode

            with torch.inference_mode(), FlopCounterMode(display=False) as counter:
                model(image, nir)
            flops = int(counter.get_total_flops())
            row.update(
                flops_per_batch=flops,
                flops_per_sample=flops / batch,
                flop_backend="torch.utils.flop_counter",
            )
        except Exception as e:
            row["flop_counter_error"] = str(e)
        report["batches"].append(row)
    json_write(report, out / "profile.json")
    print("Saved", out / "profile.json")


if __name__ == "__main__":
    main()
