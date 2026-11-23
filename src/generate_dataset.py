"""Generate the relief dataset (thin wrapper over synth.generate)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/relief_dataset")
    ap.add_argument("--n-scenes", type=int, default=4000)
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()
    from synth.generate import contact_sheet, generate_dataset, qc_assertions

    out = generate_dataset(args.out, n_scenes=args.n_scenes, seed0=args.seed0)
    print("QC:", qc_assertions(out))
    sheet = contact_sheet(out, n=24)
    print("contact sheet:", sheet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
