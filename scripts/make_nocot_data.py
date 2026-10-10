#!/usr/bin/env python3
"""Data folder for the No-CoT (direct-answer) SFT baseline.

    python scripts/make_nocot_data.py --out data_nocot
    GPU=0 DATA_DIR=$PWD/data_nocot bash sft/run_sft.sh scales qwen3.5-4b original 0     # then train as usual

Same 3,369 training and 422 validation records, same prompt, same pairing and record order as the CoT runs; only the
assistant target changes from the CoT to "<answer>{reference SMILES}</answer>". The No-CoT target sits on the
"original" side of train_sft.py's pairing, so the baseline is trained with --variant original; the curated and test
files are symbolic links to data/ (used only for the pairing and the evaluation).
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(REPO / "data"))
    ap.add_argument("--out", default=str(REPO / "data_nocot"))
    a = ap.parse_args()
    src, out = Path(a.data_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("train_data.jsonl", "val_data.jsonl"):
        rows = [json.loads(line) for line in open(src / name, encoding="utf-8")]
        with open(out / name, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps({**r, "cot": f"<answer>{r['smile']}</answer>"}, ensure_ascii=False) + "\n")
        print(f"{out / name}: {len(rows)} records with the direct-answer target")
    for name in ("curated_train_data.jsonl", "curated_val_data.jsonl", "test_data.jsonl", "curated_test_data.jsonl"):
        link = out / name
        if not link.exists():
            link.symlink_to((src / name).resolve())


if __name__ == "__main__":
    main()
