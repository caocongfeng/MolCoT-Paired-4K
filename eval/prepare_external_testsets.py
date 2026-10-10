#!/usr/bin/env python3
"""Build the two external molecule-generation test sets in the format of data/test_data.jsonl.

    python eval/prepare_external_testsets.py \
        --chebi20-test  path/to/ChEBI-20/test.txt \
        --smolinstruct-mg-test path/to/SMolInstruct/raw/test/molecule_generation.jsonl \
        --out data_external

Sources (public, not redistributed here):
  ChEBI-20 test split (3,300 description-molecule pairs; Edwards et al., 2021): test.txt, tab-separated
      SMILES and description (an optional header line with the columns SMILES / description is accepted).
  SMolInstruct molecule-generation test split (Yu et al., 2024): raw/test/molecule_generation.jsonl of the
      released data archive (fields "input" = description, "output" = SMILES). Records whose reference SMILES
      RDKit cannot parse are dropped.

Each output folder holds test_data.jsonl (fields smile, explain, cot = "n/a", source, index) and symbolic links to
the MolCoT-Paired-4K train / val files, which sft/train_sft.py reads for its data pairing also in --eval-only mode.
The paper reports exact match on the records whose molecule does not occur in the MolCoT-Paired-4K training split
(eval/external_unseen_em.py). Note that most SMolInstruct test descriptions repeat ChEBI-20 test descriptions.
"""
import argparse
import csv
import json
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
REPO = Path(__file__).resolve().parent.parent
LINKED = ["train_data.jsonl", "val_data.jsonl", "curated_train_data.jsonl", "curated_val_data.jsonl"]


def write_set(rows, folder, data_dir):
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "test_data.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    for name in LINKED:
        link = folder / name
        if not link.exists():
            link.symlink_to((data_dir / name).resolve())
    print(f"{folder}: {len(rows)} records")


def read_chebi20(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        lines = [line.rstrip("\n") for line in fh if line.strip()]
    header = [c.strip().lower() for c in lines[0].split("\t")]
    if "smiles" in header and "description" in header:
        i_s, i_d = header.index("smiles"), header.index("description")
        lines = lines[1:]
    else:
        i_s, i_d = 0, 1
    for k, line in enumerate(lines):
        cols = line.split("\t")
        smiles = cols[i_s].replace("\\\\", "\\")          # the release writes bond-direction backslashes doubled
        rows.append({"smile": smiles, "explain": cols[i_d], "cot": "n/a", "source": "chebi20_test", "index": k})
    return rows


def read_smolinstruct(path):
    rows, dropped = [], 0
    with open(path, encoding="utf-8") as fh:
        for k, line in enumerate(fh):
            r = json.loads(line)
            if Chem.MolFromSmiles(r["output"]) is None:
                dropped += 1
                continue
            rows.append({"smile": r["output"], "explain": r["input"].strip(), "cot": "n/a", "source": "smolinstruct_mg_test", "index": k})
    print(f"SMolInstruct: dropped {dropped} records with an RDKit-invalid reference")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chebi20-test")
    ap.add_argument("--smolinstruct-mg-test")
    ap.add_argument("--data-dir", default=str(REPO / "data"), help="MolCoT-Paired-4K data folder")
    ap.add_argument("--out", default=str(REPO / "data_external"))
    a = ap.parse_args()
    out, data_dir = Path(a.out), Path(a.data_dir)
    if a.chebi20_test:
        write_set(read_chebi20(a.chebi20_test), out / "chebi20", data_dir)
    if a.smolinstruct_mg_test:
        write_set(read_smolinstruct(a.smolinstruct_mg_test), out / "smolinstruct_mg", data_dir)
    if not (a.chebi20_test or a.smolinstruct_mg_test):
        ap.error("give --chebi20-test and/or --smolinstruct-mg-test")


if __name__ == "__main__":
    main()
