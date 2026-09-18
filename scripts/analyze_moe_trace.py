from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path


def layer_number(name: str) -> int:
    match = re.search(r"(?:layers|layer)\.(\d+)", name)
    return int(match.group(1)) if match else 1_000_000


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def load_records(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def simulate_lru(rows, capacity, initial):
    cache = collections.OrderedDict((int(e), None) for e in initial[:capacity])
    hits = total = 0
    for row in rows:
        protected = {int(e) for e in row}
        for raw in row:
            e = int(raw)
            total += 1
            if e in cache:
                hits += 1
                cache.move_to_end(e)
                continue
            while len(cache) >= capacity:
                victim = next(iter(cache))
                if victim not in protected or all(v in protected for v in cache):
                    cache.pop(victim)
                    break
                cache.move_to_end(victim)
            cache[e] = None
    return hits, total


def static_recall(train_rows, test_rows, widths):
    counts = collections.Counter(int(e) for row in train_rows for e in row)
    ranked = [e for e, _ in counts.most_common()]
    total = sum(len(row) for row in test_rows)
    return {
        width: pct(sum(e in set(ranked[:width]) for row in test_rows for e in row), total)
        for width in widths
    }


def transition_recall(source_rows, target_rows, widths):
    n = min(len(source_rows), len(target_rows))
    cut = max(1, int(n * 0.7))
    transitions = collections.defaultdict(collections.Counter)
    fallback = collections.Counter()
    for source, target in zip(source_rows[:cut], target_rows[:cut]):
        fallback.update(int(e) for e in target)
        for s in source:
            transitions[int(s)].update(int(e) for e in target)
    fallback_ranked = [e for e, _ in fallback.most_common()]
    hits = {w: 0 for w in widths}
    total = 0
    for source, target in zip(source_rows[cut:n], target_rows[cut:n]):
        scores = collections.Counter()
        for s in source:
            scores.update(transitions.get(int(s), {}))
        ranked = [e for e, _ in scores.most_common()]
        ranked.extend(e for e in fallback_ranked if e not in scores)
        total += len(target)
        for width in widths:
            pred = set(ranked[:width])
            hits[width] += sum(int(e) in pred for e in target)
    return {w: pct(hits[w], total) for w in widths}


def _stable_rank(counter, size):
    """Rank the complete expert id domain by count, then id for deterministic ties."""
    return sorted(range(size), key=lambda expert: (-counter[expert], expert))


def build_transition_profile(by_sequence_module, metadata, width):
    """Build per-adjacent-layer transition tables without crossing sequence boundaries."""
    if width <= 0:
        raise ValueError("transition width must be positive")

    pairs = {}
    for sequence in sorted({seq for seq, _ in by_sequence_module}):
        names = sorted(
            (name for seq, name in by_sequence_module if seq == sequence), key=layer_number
        )
        for source_name, target_name in zip(names, names[1:]):
            source_rows = by_sequence_module[(sequence, source_name)]
            target_rows = by_sequence_module[(sequence, target_name)]
            n = min(len(source_rows), len(target_rows))
            cut = max(1, int(n * 0.7))
            state = pairs.setdefault(
                target_name,
                {
                    "source": source_name,
                    "fallback": collections.Counter(),
                    "transitions": collections.defaultdict(collections.Counter),
                },
            )
            if state["source"] != source_name:
                raise ValueError(
                    f"target module {target_name!r} has multiple adjacent sources: "
                    f"{state['source']!r} and {source_name!r}"
                )
            for source, target in zip(source_rows[:cut], target_rows[:cut]):
                target_ids = [int(expert) for expert in target]
                state["fallback"].update(target_ids)
                for source_expert in source:
                    state["transitions"][int(source_expert)].update(target_ids)

    profile = {}
    for target_name in sorted(pairs, key=layer_number):
        state = pairs[target_name]
        source_name = state["source"]
        num_source = int(metadata[source_name]["num_experts"])
        num_target = int(metadata[target_name]["num_experts"])
        out_width = min(width, num_target)
        fallback = _stable_rank(state["fallback"], num_target)
        by_source = []
        for source_expert in range(num_source):
            ranked = _stable_rank(state["transitions"][source_expert], num_target)
            # _stable_rank already covers the complete target domain. Re-sorting zero-count
            # entries by the global fallback gives unseen transitions a useful prior.
            seen = {
                expert
                for expert, count in state["transitions"][source_expert].items()
                if count
            }
            ranked = [expert for expert in ranked if expert in seen]
            ranked.extend(expert for expert in fallback if expert not in seen)
            by_source.append(ranked[:out_width])
        profile[target_name] = {
            "source": source_name,
            "width": out_width,
            "num_source_experts": num_source,
            "num_target_experts": num_target,
            "fallback": fallback,
            "by_source": by_source,
        }
    return profile


def main():
    parser = argparse.ArgumentParser(description="Analyze EXL3 split-MoE routing traces")
    parser.add_argument("trace", type=Path)
    parser.add_argument("--json", type=Path, dest="json_path")
    parser.add_argument("--placement-stats", type=Path)
    parser.add_argument("--min-sequence", type=int, default=0)
    parser.add_argument("--component", choices=("main", "mtp", "all"), default="main")
    parser.add_argument("--transition-profile", type=Path)
    parser.add_argument("--transition-width", type=int, default=16)
    args = parser.parse_args()

    if args.transition_width <= 0:
        parser.error("--transition-width must be positive")

    records = load_records(args.trace)
    records = [r for r in records if int(r["sequence"]) >= args.min_sequence]
    if args.component == "main":
        records = [r for r in records if r["module"].startswith("model.")]
    elif args.component == "mtp":
        records = [r for r in records if r["module"].startswith("mtp.")]
    if not records:
        raise SystemExit("trace contains no records")

    if args.placement_stats:
        placement_counts = {}
        placement_meta = {}
        for record in records:
            name = record["module"]
            size = int(record["num_experts"])
            counts = placement_counts.setdefault(name, [0] * size)
            for row in record["selected"]:
                for expert in row:
                    counts[int(expert)] += 1
            placement_meta[name] = {
                "samples": sum(counts),
                "num_experts": size,
                "top_k": int(record["top_k"]),
            }
        args.placement_stats.parent.mkdir(parents=True, exist_ok=True)
        args.placement_stats.write_text(
            json.dumps({**placement_counts, "_meta": placement_meta}, indent=2),
            encoding="utf-8",
        )

    by_module = collections.defaultdict(list)
    records_by_module = collections.defaultdict(list)
    by_sequence_module = collections.defaultdict(list)
    metadata = {}
    for record in records:
        name = record["module"]
        by_module[name].extend(record["selected"])
        records_by_module[name].append(record)
        by_sequence_module[(record["sequence"], name)].extend(record["selected"])
        metadata[name] = record

    if args.transition_profile:
        transition_profile = build_transition_profile(
            by_sequence_module, metadata, args.transition_width
        )
        transition_profile["_meta"] = {
            "schema_version": 1,
            "trace": str(args.trace),
            "component": args.component,
            "training_fraction": 0.7,
        }
        args.transition_profile.parent.mkdir(parents=True, exist_ok=True)
        args.transition_profile.write_text(
            json.dumps(transition_profile, indent=2), encoding="utf-8"
        )

    layer_results = []
    totals = collections.Counter()
    lru_hits = lru_total = 0
    previous_hits = previous_total = 0
    for name in sorted(by_module, key=layer_number):
        rows = by_module[name]
        meta = metadata[name]
        placement = [int(v) for v in records_by_module[name][0]["placement"]]
        slots = int(meta["gpu_slots"])
        counts = collections.Counter(int(e) for row in rows for e in row)
        total = sum(len(row) for row in rows)
        # Placement can change between generations when dynamic expert swapping is
        # enabled. Score every trace record against the placement that was active
        # for that record instead of applying the final placement to all rows.
        observed_hits = 0
        for record in records_by_module[name]:
            record_placement = [int(v) for v in record["placement"]]
            record_slots = int(record["gpu_slots"])
            observed_hits += sum(
                record_placement[int(e)] < record_slots
                for row in record["selected"]
                for e in row
            )
        hottest = {e for e, _ in counts.most_common(slots)}
        oracle_hits = sum(int(e) in hottest for row in rows for e in row)
        initial = [router for router, physical in enumerate(placement) if physical < slots]
        lh, lt = simulate_lru(rows, slots, initial)
        ph = pt = 0
        for previous, current in zip(rows, rows[1:]):
            prev_set = {int(e) for e in previous}
            ph += sum(int(e) in prev_set for e in current)
            pt += len(current)
        lru_hits += lh
        lru_total += lt
        previous_hits += ph
        previous_total += pt
        totals.update(total=total, observed=observed_hits, oracle=oracle_hits)
        layer_results.append({
            "module": name,
            "device": int(meta["device"]),
            "rows": len(rows),
            "gpu_slots": slots,
            "observed_gpu_hit_pct": pct(observed_hits, total),
            "static_oracle_hit_pct": pct(oracle_hits, total),
            "per_token_lru_hit_pct": pct(lh, lt),
            "previous_token_recall_pct": pct(ph, pt),
        })

    widths = (10, 20, 40)
    static_acc = collections.Counter()
    static_layers = 0
    for rows in by_module.values():
        cut = max(1, int(len(rows) * 0.7))
        recalls = static_recall(rows[:cut], rows[cut:], widths)
        for width, value in recalls.items():
            static_acc[width] += value
        static_layers += 1

    transition_acc = collections.Counter()
    transition_pairs = 0
    for sequence in sorted({seq for seq, _ in by_sequence_module}):
        names = sorted(
            (name for seq, name in by_sequence_module if seq == sequence), key=layer_number
        )
        for source_name, target_name in zip(names, names[1:]):
            result = transition_recall(
                by_sequence_module[(sequence, source_name)],
                by_sequence_module[(sequence, target_name)],
                widths,
            )
            for width, value in result.items():
                transition_acc[width] += value
            transition_pairs += 1

    summary = {
        "records": len(records),
        "layers": len(by_module),
        "router_picks": totals["total"],
        "observed_gpu_hit_pct": pct(totals["observed"], totals["total"]),
        "static_oracle_hit_pct": pct(totals["oracle"], totals["total"]),
        "per_token_lru_hit_pct": pct(lru_hits, lru_total),
        "previous_token_recall_pct": pct(previous_hits, previous_total),
        "static_train_test_recall_pct": {
            str(w): static_acc[w] / static_layers for w in widths
        },
        "cross_layer_transition_recall_pct": {
            str(w): transition_acc[w] / transition_pairs if transition_pairs else 0.0
            for w in widths
        },
        "layers_detail": layer_results,
    }

    print(json.dumps({k: v for k, v in summary.items() if k != "layers_detail"}, indent=2))
    print("\nWorst observed GPU-hit layers:")
    for row in sorted(layer_results, key=lambda x: x["observed_gpu_hit_pct"])[:8]:
        print(
            f"  {row['module']}: observed={row['observed_gpu_hit_pct']:.1f}% "
            f"oracle={row['static_oracle_hit_pct']:.1f}% "
            f"LRU={row['per_token_lru_hit_pct']:.1f}% prev={row['previous_token_recall_pct']:.1f}%"
        )
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
