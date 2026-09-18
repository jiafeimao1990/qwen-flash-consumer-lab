from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge EXL3 MoE placement profiles")
    parser.add_argument("base", type=Path)
    parser.add_argument("overlay", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    base = json.loads(args.base.read_text(encoding="utf-8"))
    overlay = json.loads(args.overlay.read_text(encoding="utf-8"))
    base_meta = base.setdefault("_meta", {})
    overlay_meta = overlay.get("_meta", {})
    for key, value in overlay.items():
        if key == "_meta":
            continue
        base[key] = value
        if key in overlay_meta:
            base_meta[key] = overlay_meta[key]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(base, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
