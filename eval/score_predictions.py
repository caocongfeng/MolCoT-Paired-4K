#!/usr/bin/env python3
"""Score prediction files written by sft/train_sft.py (test_predictions_epoch_*.json) or train_grpo.py.

    python eval/score_predictions.py runs/qwen3.5-4b-base/curated/run_000/test_predictions_epoch_004.json [...]
    python eval/score_predictions.py runs/ --out scores.csv          # every test_predictions*.json below runs/

One output row per file. Each file holds, per test record, the greedy answer and N sampled answers (N = 16 in the
paper). Two readings of every generation are reported:

  stored  the answer rule of train_sft.py (last <answer>...</answer> of the whole generation; without a tag, the
          last non-empty line). These are the numbers in the figures of the paper.
  strict  only the model's own turn (cut at the first line that is just a role name), the last CLOSED <answer> tag
          in that turn, no fallback: a generation without a tag has no answer.

Metrics (percent, averaged over test records):
  greedy_{EM,validity,similarity}   the greedy answer
  pass@1_{EM,validity,similarity}   mean over the N samples (the N = 1 point of a pass@N curve)
  pass@k_EM, pass@k_validity        unbiased pass@k from N samples (Chen et al., 2021), k = 2, 4, 8, 16
  pass@k_similarity                 unbiased expected maximum similarity of k samples drawn from the N
EM = exact match of the RDKit-canonical SMILES; validity = RDKit can parse the answer; similarity = Tanimoto
similarity of Morgan fingerprints (radius 2) to the reference, 0 for invalid answers.
"""
import argparse
import csv
import json
import re
import sys
from functools import lru_cache
from math import comb
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
K_VALUES = (1, 2, 4, 8, 16)

# --- answer extraction: identical to sft/train_sft.py ------------------------------------------------------------
ANSWER_RE = re.compile(r"<\s*answer\s*>\s*(.*?)\s*<\s*/\s*answer\s*>", re.IGNORECASE | re.DOTALL)
# a chat-turn boundary after the special tokens were stripped: a line that holds nothing but a role name
TURN_RE = re.compile(r"^(?:user|assistant|system)[ \t]*$", re.MULTILINE)


def extract_answer(text):
    """Stored rule of train_sft.py: last tag of the whole text, else the last non-empty line."""
    matches = ANSWER_RE.findall(text or "")
    candidate = matches[-1].strip() if matches else ""
    if not candidate and text:
        nonempty = [line.strip() for line in text.splitlines() if line.strip()]
        candidate = nonempty[-1] if nonempty else ""
    candidate = candidate.replace("```smiles", "").replace("```", "").strip()
    return candidate.splitlines()[0].strip() if candidate else ""


def strict_answer(text):
    """Strict rule: own turn only, last closed tag, None without a tag."""
    boundary = TURN_RE.search(text or "")
    turn = (text or "") if boundary is None else text[: boundary.start()]
    matches = ANSWER_RE.findall(turn)
    if not matches:
        return None
    candidate = matches[-1].strip().replace("```smiles", "").replace("```", "").strip()
    return candidate.splitlines()[0].strip() if candidate else ""


@lru_cache(maxsize=None)
def canonicalize(smiles):
    if not smiles:
        return None
    molecule = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(molecule, canonical=True) if molecule else None


_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2)


@lru_cache(maxsize=None)
def tanimoto(canonical_a, canonical_b):
    if not canonical_a or not canonical_b:
        return 0.0
    a, b = Chem.MolFromSmiles(canonical_a), Chem.MolFromSmiles(canonical_b)
    if a is None or b is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(_MORGAN.GetFingerprint(a), _MORGAN.GetFingerprint(b)))


def score(answer, reference_canonical):
    """(exact, valid, similarity) of one extracted answer; answer None = no answer."""
    canonical = canonicalize(answer) if answer else None
    if canonical is None:
        return 0, 0, 0.0
    return int(canonical == reference_canonical), 1, tanimoto(canonical, reference_canonical)


# --- estimators ---------------------------------------------------------------------------------------------------
def pass_at_k(n, c, k):
    """Unbiased pass@k (Chen et al., 2021) from n samples with c successes."""
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


def expected_max(values, k):
    """Unbiased E[max of k values drawn without replacement from the n values]."""
    v = np.sort(np.asarray(values, dtype=float)); n = len(v)
    weights = np.array([comb(i - 1, k - 1) / comb(n, k) for i in range(1, n + 1)])
    return float(weights @ v)


def score_file(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    predictions = payload["predictions"]
    n = len(predictions[0]["sampled"])
    row = {"file": str(path), "n_records": len(predictions), "n_samples": n}
    for rule, extract in (("stored", extract_answer), ("strict", strict_answer)):
        greedy, per_item = [], []
        for record in predictions:
            reference = record.get("canonical_reference") or canonicalize(record["reference"])
            greedy.append(score(extract(record["greedy"]["generation"]), reference))
            per_item.append([score(extract(s["generation"]), reference) for s in record["sampled"]])
        g = np.array(greedy, dtype=float)
        row[f"{rule}_greedy_EM"], row[f"{rule}_greedy_validity"], row[f"{rule}_greedy_similarity"] = 100 * g.mean(axis=0)
        s = np.array(per_item, dtype=float)                          # records x samples x (em, valid, sim)
        row[f"{rule}_pass@1_EM"], row[f"{rule}_pass@1_validity"], row[f"{rule}_pass@1_similarity"] = 100 * s.mean(axis=(0, 1))
        for k in K_VALUES[1:]:
            if k > n:
                continue
            row[f"{rule}_pass@{k}_EM"] = 100 * np.mean([pass_at_k(n, int(c), k) for c in s[:, :, 0].sum(axis=1)])
            row[f"{rule}_pass@{k}_validity"] = 100 * np.mean([pass_at_k(n, int(c), k) for c in s[:, :, 1].sum(axis=1)])
            row[f"{rule}_pass@{k}_similarity"] = 100 * np.mean([expected_max(sim, k) for sim in s[:, :, 2]])
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="prediction files or folders (searched for test_predictions*.json)")
    ap.add_argument("--out", help="write all rows to this CSV file")
    a = ap.parse_args()
    files = []
    for p in map(Path, a.paths):
        files.extend(sorted(p.rglob("test_predictions*.json")) if p.is_dir() else [p])
    rows = []
    for f in files:
        rows.append(score_file(f))
        r = rows[-1]
        print(f"{f}: stored greedy EM {r['stored_greedy_EM']:.1f} | pass@1 {r['stored_pass@1_EM']:.1f} | "
              f"pass@16 {r.get('stored_pass@16_EM', float('nan')):.1f}   (strict greedy EM {r['strict_greedy_EM']:.1f})", flush=True)
    if a.out and rows:
        keys = list(rows[0])
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
        print(f"wrote {a.out} ({len(rows)} rows)")


if __name__ == "__main__":
    sys.exit(main())
