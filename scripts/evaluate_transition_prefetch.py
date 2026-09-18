from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from analyze_moe_trace import layer_number, load_records


def evaluate(records, profile, stage_widths, expert_bytes):
    by_seq_mod = collections.defaultdict(list)
    rec_by_seq_mod = {}
    for record in records:
        if not record["module"].startswith("model."):
            continue
        key = (int(record["sequence"]), record["module"])
        by_seq_mod[key].extend(record["selected"])
        rec_by_seq_mod[key] = record

    totals = {
        width: collections.Counter(rows=0, cpu_picks=0, hits=0, staged=0, useful_rows=0)
        for width in stage_widths
    }
    layers = {width: collections.defaultdict(collections.Counter) for width in stage_widths}
    total_test_tokens = 0

    for sequence in sorted({seq for seq, _ in by_seq_mod}):
        names = sorted((name for seq, name in by_seq_mod if seq == sequence), key=layer_number)
        if names:
            first_rows = by_seq_mod[(sequence, names[0])]
            total_test_tokens += max(0, len(first_rows) - max(1, int(len(first_rows) * 0.7)))
        for source_name, target_name in zip(names, names[1:]):
            entry = profile.get(target_name)
            if not entry or entry["source"] != source_name:
                continue
            source_rows = by_seq_mod[(sequence, source_name)]
            target_rows = by_seq_mod[(sequence, target_name)]
            n = min(len(source_rows), len(target_rows))
            cut = max(1, int(n * 0.7))
            target_record = rec_by_seq_mod[(sequence, target_name)]
            placement = [int(value) for value in target_record["placement"]]
            gpu_slots = int(target_record["gpu_slots"])
            fallback = [int(value) for value in entry["fallback"]]
            table = entry["by_source"]

            for source, target in zip(source_rows[cut:n], target_rows[cut:n]):
                scores = collections.Counter()
                for source_expert in source:
                    ranked = table[int(source_expert)]
                    for rank, candidate in enumerate(ranked):
                        scores[int(candidate)] += len(ranked) - rank
                ranked = sorted(scores, key=lambda expert: (-scores[expert], expert))
                ranked.extend(expert for expert in fallback if expert not in scores)
                cpu_ranked = [expert for expert in ranked if placement[expert] >= gpu_slots]
                cpu_actual = [int(expert) for expert in target if placement[int(expert)] >= gpu_slots]
                for width in stage_widths:
                    predicted = set(cpu_ranked[:width])
                    hits = sum(expert in predicted for expert in cpu_actual)
                    row = totals[width]
                    row.update(rows=1, cpu_picks=len(cpu_actual), hits=hits,
                               staged=min(width, len(cpu_ranked)), useful_rows=bool(hits))
                    layer = layers[width][target_name]
                    layer.update(rows=1, cpu_picks=len(cpu_actual), hits=hits,
                                 staged=min(width, len(cpu_ranked)), useful_rows=bool(hits))

    result = {"expert_bytes": expert_bytes, "stage_widths": {}}
    for width in stage_widths:
        row = totals[width]
        result["stage_widths"][str(width)] = {
            "rows": row["rows"],
            "cpu_picks": row["cpu_picks"],
            "cpu_pick_recall_pct": 100.0 * row["hits"] / row["cpu_picks"] if row["cpu_picks"] else 0.0,
            "useful_row_pct": 100.0 * row["useful_rows"] / row["rows"] if row["rows"] else 0.0,
            "mean_staged_experts_per_layer_row": row["staged"] / row["rows"] if row["rows"] else 0.0,
            "mean_transfer_mib_per_token": (
                row["staged"] * expert_bytes / max(1, total_test_tokens) / (1024 * 1024)
            ),
            "best_layers": [
                {
                    "module": name,
                    "cpu_pick_recall_pct": 100.0 * stat["hits"] / stat["cpu_picks"] if stat["cpu_picks"] else 0.0,
                    "useful_row_pct": 100.0 * stat["useful_rows"] / stat["rows"] if stat["rows"] else 0.0,
                    "cpu_picks": stat["cpu_picks"],
                }
                for name, stat in sorted(
                    layers[width].items(),
                    key=lambda item: (-(item[1]["hits"] / item[1]["cpu_picks"] if item[1]["cpu_picks"] else 0),
                                      layer_number(item[0])),
                )[:12]
            ],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate CPU-tail transition-prefetch coverage")
    parser.add_argument("trace", type=Path)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--stage-widths", default="1,2,4,8")
    parser.add_argument("--expert-bytes", type=int, default=2476736)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    widths = tuple(int(value) for value in args.stage_widths.split(",") if value.strip())
    if not widths or any(width <= 0 for width in widths):
        parser.error("--stage-widths must contain positive integers")
    result = evaluate(
        load_records(args.trace),
        json.loads(args.profile.read_text(encoding="utf-8")),
        widths,
        args.expert_bytes,
    )
    text = json.dumps(result, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
