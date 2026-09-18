from __future__ import annotations

import argparse
import heapq
import json
import re
from pathlib import Path


def layer_number(name: str) -> int | None:
    match = re.search(r"model\.language_model\.layers\.(\d+)\.mlp$", name)
    return int(match.group(1)) if match else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Allocate a fixed per-GPU MoE expert-slot budget across layers"
    )
    parser.add_argument("profile", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--default-slots", type=int, default=160)
    parser.add_argument("--min-slots", type=int, default=128)
    parser.add_argument("--max-slots", type=int, default=192)
    parser.add_argument("--split-after-layer", type=int, default=22)
    parser.add_argument("--prefix-safe", action="store_true")
    args = parser.parse_args()

    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    layers = {
        layer_number(name): (name, [int(v) for v in counts])
        for name, counts in profile.items()
        if name != "_meta" and layer_number(name) is not None
    }
    if not layers:
        raise SystemExit("profile contains no main-model MoE layers")

    groups = [
        sorted(i for i in layers if i <= args.split_after_layer),
        sorted(i for i in layers if i > args.split_after_layer),
    ]
    allocation: dict[str, int] = {}
    meta = {"groups": [], "objective": "maximize observed router hits at fixed slots"}

    for group in groups:
        ranked = {
            i: sorted(layers[i][1], reverse=True)
            for i in group
        }
        target = len(group) * args.default_slots
        if args.prefix_safe:
            # DP over cumulative slot delta. Constraining every prefix to <= 0 means
            # autosplit never sees more expert bytes at any load point than the uniform
            # baseline; the final state must return to zero to preserve total VRAM.
            dp: dict[int, tuple[int, list[int]]] = {0: (0, [])}
            for i in group:
                utility = {
                    c: sum(ranked[i][:c])
                    for c in range(args.min_slots, args.max_slots + 1)
                }
                nxt: dict[int, tuple[int, list[int]]] = {}
                for delta, (score, chosen) in dp.items():
                    for c, value in utility.items():
                        new_delta = delta + c - args.default_slots
                        if new_delta > 0:
                            continue
                        candidate = (score + value, chosen + [c])
                        if new_delta not in nxt or candidate[0] > nxt[new_delta][0]:
                            nxt[new_delta] = candidate
                dp = nxt
            if 0 not in dp:
                raise SystemExit(f"no prefix-safe allocation for group {group}")
            slots = dict(zip(group, dp[0][1]))
        else:
            slots = {i: args.min_slots for i in group}
            remaining = target - sum(slots.values())
            heap: list[tuple[int, int]] = []
            for i in group:
                if slots[i] < min(args.max_slots, len(ranked[i])):
                    heapq.heappush(heap, (-ranked[i][slots[i]], i))
            while remaining > 0 and heap:
                _, i = heapq.heappop(heap)
                slots[i] += 1
                remaining -= 1
                if slots[i] < min(args.max_slots, len(ranked[i])):
                    heapq.heappush(heap, (-ranked[i][slots[i]], i))
            if remaining:
                raise SystemExit(f"unable to allocate {remaining} slots in group {group}")
        for i in group:
            allocation[layers[i][0]] = slots[i]
        meta["groups"].append({
            "layers": [group[0], group[-1]],
            "total_slots": sum(slots.values()),
            "min": min(slots.values()),
            "max": max(slots.values()),
        })

    meta["prefix_safe"] = args.prefix_safe
    allocation["_meta"] = meta
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(allocation, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
