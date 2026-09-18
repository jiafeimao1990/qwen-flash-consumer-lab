"""Measure the small-tensor cross-GPU path needed by MoE expert co-processing.

This intentionally allocates only a few MiB and can run beside the stable server.
It measures latency rather than bulk bandwidth because decode sends only one or a
few hidden-state rows per layer.
"""

from __future__ import annotations

import json
import statistics
import time

import torch


def sync_all() -> None:
    for device in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device)


def bench_direct(rows: int, hidden: int, rounds: int = 2000) -> dict:
    src = torch.randn((rows, hidden), dtype=torch.float16, device="cuda:0")
    for _ in range(100):
        remote = src.to("cuda:1", non_blocking=True)
        returned = remote.to("cuda:0", non_blocking=True)
    sync_all()

    samples = []
    for _ in range(rounds):
        start = time.perf_counter_ns()
        remote = src.to("cuda:1", non_blocking=True)
        returned = remote.to("cuda:0", non_blocking=True)
        sync_all()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "rows": rows,
        "one_way_bytes": src.numel() * src.element_size(),
        "median_ms": statistics.median(samples),
        "p90_ms": statistics.quantiles(samples, n=10)[8],
        "p99_ms": statistics.quantiles(samples, n=100)[98],
        "mean_ms": statistics.fmean(samples),
    }


def bench_local(rows: int, hidden: int, rounds: int = 2000) -> dict:
    src = torch.randn((rows, hidden), dtype=torch.float16, device="cuda:0")
    dst = torch.empty_like(src)
    for _ in range(100):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize(0)
    samples = []
    for _ in range(rounds):
        start = time.perf_counter_ns()
        dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize(0)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "rows": rows,
        "median_ms": statistics.median(samples),
        "p90_ms": statistics.quantiles(samples, n=10)[8],
        "p99_ms": statistics.quantiles(samples, n=100)[98],
        "mean_ms": statistics.fmean(samples),
    }


def main() -> None:
    assert torch.cuda.device_count() >= 2
    result = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "devices": [torch.cuda.get_device_name(i) for i in range(2)],
        "hidden": 2560,
        "direct_round_trip": [bench_direct(r, 2560) for r in (1, 2, 4, 8)],
        "local_copy": [bench_local(r, 2560) for r in (1, 2, 4, 8)],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
