#!/usr/bin/env python3
"""Strict greedy exact match on an external test set, restricted to molecules that do not occur in the
MolCoT-Paired-4K training split (the setting of the paper's external evaluation).

    python eval/external_unseen_em.py runs_external/chebi20/sft_curated/curated/run_00*/test_predictions.json

Prints, per prediction file and as the mean over files (runs): the number of unseen records, the share of answered
generations (a closed <answer> tag in the model's own turn), and the strict exact match.
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_predictions import ANSWER_RE, TURN_RE, canonicalize, score  # noqa: E402


def first_tag_answer(text):
    """Strict answer of the external evaluation: own turn only, FIRST closed <answer> tag, None without a tag.
    (For the trained models this equals the last-tag rule of score_predictions.py up to <= 0.01 points.)"""
    boundary = TURN_RE.search(text or "")
    turn = (text or "") if boundary is None else text[: boundary.start()]
    matches = ANSWER_RE.findall(turn)
    if not matches:
        return None
    candidate = matches[0].strip().replace("```smiles", "").replace("```", "").strip()
    return candidate.splitlines()[0].strip() if candidate else ""

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--train", default=str(REPO / "data" / "train_data.jsonl"))
    ap.add_argument("--all-records", action="store_true", help="do not restrict to unseen molecules")
    a = ap.parse_args()
    train_mols = {canonicalize(json.loads(line)["smile"]) for line in open(a.train, encoding="utf-8")}
    ems = []
    for f in a.files:
        preds = json.loads(Path(f).read_text(encoding="utf-8"))["predictions"]
        keep = [p for p in preds if a.all_records or canonicalize(p["reference"]) not in train_mols]
        answered = [first_tag_answer(p["greedy"]["generation"]) is not None for p in keep]
        em = [score(first_tag_answer(p["greedy"]["generation"]), canonicalize(p["reference"]))[0] for p in keep]
        ems.append(100 * st.mean(em))
        print(f"{f}: {len(keep)} records, answered {100 * st.mean(answered):.1f}%, strict greedy EM {ems[-1]:.2f}")
    if len(ems) > 1:
        print(f"mean over {len(ems)} runs: {st.mean(ems):.2f} (sd {st.stdev(ems):.2f})")


if __name__ == "__main__":
    main()
