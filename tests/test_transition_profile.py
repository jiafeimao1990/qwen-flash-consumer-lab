import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYZER = ROOT / "scripts" / "analyze_moe_trace.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_moe_trace", ANALYZER
)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def record(sequence, module, selected, num_experts=4):
    return {
        "sequence": sequence,
        "module": module,
        "selected": selected,
        "num_experts": num_experts,
        "top_k": len(selected[0]),
        "placement": list(range(num_experts)),
        "gpu_slots": 2,
        "device": 0,
    }


class TransitionProfileTests(unittest.TestCase):
    def build(self, records, width=3):
        by = {}
        meta = {}
        for row in records:
            by.setdefault((row["sequence"], row["module"]), []).extend(row["selected"])
            meta[row["module"]] = row
        return MOD.build_transition_profile(by, meta, width)

    def test_stable_ties_fallback_fill_and_no_duplicates(self):
        rows = [
            record(1, "model.layers.0.mlp", [[0], [0], [1], [1]]),
            record(1, "model.layers.1.mlp", [[2], [1], [3], [0]]),
        ]
        entry = self.build(rows)["model.layers.1.mlp"]
        self.assertEqual(entry["fallback"], [1, 2, 0, 3])
        self.assertEqual(entry["by_source"][0], [1, 2, 0])
        self.assertEqual(len(entry["by_source"][0]), len(set(entry["by_source"][0])))

    def test_sequences_do_not_cross_and_width_is_capped(self):
        rows = [
            record(1, "model.layers.0.mlp", [[0], [0]]),
            record(1, "model.layers.1.mlp", [[1], [1]]),
            record(2, "model.layers.0.mlp", [[0], [0]]),
            record(2, "model.layers.1.mlp", [[2], [2]]),
        ]
        entry = self.build(rows, width=99)["model.layers.1.mlp"]
        self.assertEqual(entry["width"], 4)
        self.assertEqual(entry["by_source"][0][:2], [1, 2])

    def test_invalid_width(self):
        with self.assertRaises(ValueError):
            self.build([], width=0)

    def test_cli_backward_compatible_and_emits_profile(self):
        rows = [
            record(1, "model.layers.0.mlp", [[0], [1]]),
            record(1, "model.layers.1.mlp", [[2], [3]]),
        ]
        with tempfile.TemporaryDirectory() as td:
            trace = Path(td) / "trace.jsonl"
            trace.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            plain = subprocess.run(
                [sys.executable, str(ANALYZER), str(trace)],
                capture_output=True, text=True, check=True,
            )
            self.assertIn('"records": 2', plain.stdout)
            out = Path(td) / "profile.json"
            subprocess.run(
                [sys.executable, str(ANALYZER), str(trace),
                 "--transition-profile", str(out), "--transition-width", "2"],
                capture_output=True, text=True, check=True,
            )
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(data["_meta"]["schema_version"], 1)
            self.assertEqual(data["model.layers.1.mlp"]["width"], 2)


if __name__ == "__main__":
    unittest.main()
