"""
split_report_v2.py  v2  —  read-only check of a train/val/test split
=================================================================
Scans the masks in your train/val/test folders and tells you, per MODEL class
(the classes the network is trained on):

  * patches   how many patches contain it, per split
  * tiles     how many SOURCE TILES contain it, per split  (this is what matters:
              patches of one tile are near-duplicates)
  * % pixels  what share of the class's pixels sits in train / val / test
              (target is ~80 / 10 / 10)

and flags the problems that wreck validation:
  NO DATA                the class has no pixels anywhere — untrainable
  MISSING in val/test    the class cannot be measured
  THIN                   < 10 patches or < 3 tiles in val or test
  pixel share outside 3-20%  val or test holds < 3% or > 20% of the class pixels (target ~10%)
  TILE LEAKAGE           the same source tile appears in more than one split

New in v2: tile counts and pixel shares per class, class role (KEY = water
habitats and wet vegetated habitats), raw ids that are ignored in training
(e.g. EXCLUDE_RAW classes), unknown raw values, and a faster scan.

The class scheme is IMPORTED from model_class_scheme_v2.py (never hardcoded).
Writes per_image_class_inventory.csv (read by smart_split_v2.py --inventory-csv)
and split_report_summary.json.

Usage
-----
    python split_report_v2.py --train OUT/train --val OUT/val --test OUT/test
    python split_report_v2.py --train ... --val ... --test ... --out-dir ./split_analysis
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from model_class_scheme_v2 import (RAW_TO_MODEL, CLASS_NAMES, NUM_CLASSES, CLASS_ROLES,
                                    KEY_CLASS_IDS)
    from lulc_class_scheme_v2 import RAW_NAMES
except ModuleNotFoundError as e:
    print("ERROR: could not import model_class_scheme_v2.py / lulc_class_scheme_v2.py.")
    print("Put split_report_v2.py in the same folder as those two files.")
    print(f"Underlying error: {e}")
    sys.exit(1)

SPLITS = ("train", "val", "test")
TARGET = {"train": 80.0, "val": 10.0, "test": 10.0}

RAW_LUT = np.full(256, -1, dtype=np.int16)
for _raw, _mid in RAW_TO_MODEL.items():
    if 0 <= _raw < 256:
        RAW_LUT[_raw] = _mid


def resolve_mask_dir(split_path: Path) -> Path:
    sub = split_path / "masks"
    if sub.is_dir() and any(sub.glob("*.png")):
        return sub
    return split_path


def source_tile_key(filename: str) -> str:
    stem = Path(filename).stem
    if "_p" in stem:
        head, tail = stem.rsplit("_p", 1)
        if tail.isdigit():
            stem = head
    return stem


def dataset_key(filename: str) -> str:
    name = Path(filename).name
    return name.split("__", 1)[0] if "__" in name else "unknown_dataset"


def scan(mask_dir: Path):
    patch_presence = np.zeros(NUM_CLASSES, dtype=np.int64)
    pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    tile_presence = defaultdict(set)              # model id -> {tile}
    raw_ignored = np.zeros(256, dtype=np.int64)   # raw values with no model channel
    tiles, per_image = set(), []
    n_no_fg = 0

    files = sorted(mask_dir.glob("*.png"))
    for mp in files:
        counts = np.bincount(np.array(Image.open(mp)).ravel(), minlength=256)[:256]
        tile, ds = source_tile_key(mp.name), dataset_key(mp.name)
        tiles.add(tile)
        valid = RAW_LUT >= 0
        px = np.zeros(NUM_CLASSES, dtype=np.int64)
        np.add.at(px, RAW_LUT[valid], counts[valid])
        raw_ignored += np.where(valid, 0, counts)

        fg = bool(px[1:].sum() > 0)
        n_no_fg += (not fg)
        patch_presence += (px > 0)
        pixel_counts += px
        for mid in np.nonzero(px)[0]:
            tile_presence[int(mid)].add(tile)
        per_image.append({"filename": mp.name, "tile": tile, "dataset": ds,
                          "model_px": px, "ignore_px": int(counts[~valid].sum()),
                          "has_foreground": fg})
    return {"n_patches": len(files), "n_tiles": len(tiles), "tiles": tiles,
            "patch_presence": patch_presence, "pixel_counts": pixel_counts,
            "tile_presence": dict(tile_presence), "raw_ignored": raw_ignored,
            "per_image": per_image, "n_no_foreground": n_no_fg}


def print_report(scans):
    print("\n" + "=" * 112)
    print("SPLIT REPORT")
    print("=" * 112)
    print("\nCURRENT SPLIT SIZES")
    total = 0
    for sp in SPLITS:
        s = scans[sp]
        total += s["n_patches"]
        print(f"  {sp:5s}: {s['n_patches']:7,} patches from {s['n_tiles']:4,} source tiles "
              f"({s['n_no_foreground']:,} patches with no trainable foreground)")
    print(f"  TOTAL: {total:,} patches")

    tr, va, te = (scans[s]["tiles"] for s in SPLITS)
    leaks = (tr & va) | (tr & te) | (va & te)
    print("\nTILE LEAKAGE (same source tile in >1 split)")
    if leaks:
        print(f"  WARNING: {len(leaks)} tile(s) appear in more than one split — val/test scores are inflated.")
        for t in sorted(leaks)[:12]:
            print(f"    - {t}")
        if len(leaks) > 12:
            print(f"    ... and {len(leaks) - 12} more")
    else:
        print("  None. Good.")

    tot_px = sum(scans[s]["pixel_counts"] for s in SPLITS)
    print("\nPER-CLASS DISTRIBUTION (rarest first)")
    hdr = (f"  {'mdl':>3} {'class':<24} {'role':<4} | {'patches tr/va/te':>20} | {'tiles tr/va/te':>14} | "
           f"{'% of class pixels tr/va/te':>28}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows = []
    for mid in range(1, NUM_CLASSES):
        pp = [int(scans[s]["patch_presence"][mid]) for s in SPLITS]
        tt = [len(scans[s]["tile_presence"].get(mid, ())) for s in SPLITS]
        px = [int(scans[s]["pixel_counts"][mid]) for s in SPLITS]
        rows.append((mid, pp, tt, px))
    no_data = []
    for mid, pp, tt, px in sorted(rows, key=lambda r: sum(r[2])):
        role = "KEY" if mid in KEY_CLASS_IDS else ""
        if sum(px) == 0:
            no_data.append(mid)
            print(f"  {mid:>3} {CLASS_NAMES[mid]:<24} {role:<4} | NO DATA — no pixels in any split")
            continue
        sh = [100.0 * px[i] / tot_px[mid] for i in range(3)]
        flag = ""
        if tt[1] == 0 or tt[2] == 0:
            flag = "  <-- MISSING in val/test"
        elif tt[0] == 0:
            flag = "  <-- MISSING in train"
        elif pp[1] < 10 or pp[2] < 10 or tt[1] < 3 or tt[2] < 3:
            flag = "  <-- THIN in val/test"
        elif not (3 <= sh[1] <= 20 and 3 <= sh[2] <= 20):
            flag = "  <-- val/test pixel share outside 3-20%"
        print(f"  {mid:>3} {CLASS_NAMES[mid]:<24} {role:<4} | "
              f"{pp[0]:>8,}{pp[1]:>6,}{pp[2]:>6,}   | {tt[0]:>4}{tt[1]:>4}{tt[2]:>4}   | "
              f"{sh[0]:>8.1f}{sh[1]:>9.1f}{sh[2]:>9.1f}{flag}")

    if no_data:
        print("\n  NO DATA classes: " + ", ".join(CLASS_NAMES[m] for m in no_data)
              + "\n  -> annotate them, or list them in model_class_scheme_v2.EXCLUDE_RAW so the head "
                "does not carry untrainable channels.")

    ign = sum(scans[s]["raw_ignored"] for s in SPLITS)
    shown = [(r, int(ign[r])) for r in range(256) if ign[r] > 0 and r != 255]
    print("\nIGNORED IN TRAINING (raw values with no model channel)")
    if shown:
        for r, c in sorted(shown, key=lambda x: -x[1]):
            print(f"  raw {r:>3} {RAW_NAMES.get(r, 'unknown value'):<26} {c:>14,} px")
    print(f"  raw 255 unlabelled / unmapped          {int(ign[255]):>14,} px")
    print("\n  (Background = model 0 is not judged. KEY = water habitats + wet vegetated habitats.)")


def write_outputs(scans, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "per_image_class_inventory.csv"
    cols = [f"m{mid}_{CLASS_NAMES.get(mid, mid)}".replace(" ", "_") for mid in range(NUM_CLASSES)]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "filename", "dataset", "source_tile", "has_foreground", "ignore_px"] + cols)
        for sp in SPLITS:
            for rec in scans[sp]["per_image"]:
                w.writerow([sp, rec["filename"], rec["dataset"], rec["tile"],
                            int(rec["has_foreground"]), rec["ignore_px"]] + rec["model_px"].tolist())

    summary = {
        "num_model_classes": NUM_CLASSES,
        "class_names": {str(k): v for k, v in CLASS_NAMES.items()},
        "class_roles": {str(k): v for k, v in CLASS_ROLES.items()},
        "splits": {sp: {
            "n_patches": scans[sp]["n_patches"],
            "n_tiles": scans[sp]["n_tiles"],
            "n_no_foreground": scans[sp]["n_no_foreground"],
            "patch_presence": {str(k): int(v) for k, v in enumerate(scans[sp]["patch_presence"])},
            "tile_presence": {str(k): len(v) for k, v in scans[sp]["tile_presence"].items()},
            "pixel_counts": {str(k): int(v) for k, v in enumerate(scans[sp]["pixel_counts"])},
        } for sp in SPLITS},
    }
    with open(out_dir / "split_report_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote per-image inventory : {csv_path}")
    print(f"Wrote summary JSON        : {out_dir / 'split_report_summary.json'}")


def parse_args():
    p = argparse.ArgumentParser(description="Report the class distribution of a train/val/test split.",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--train", required=True, type=Path)
    p.add_argument("--val", required=True, type=Path)
    p.add_argument("--test", required=True, type=Path)
    p.add_argument("--out-dir", type=Path, default=Path("./split_analysis"))
    return p.parse_args()


def main():
    a = parse_args()
    scans = {}
    for sp, path in {"train": a.train, "val": a.val, "test": a.test}.items():
        if not path.exists():
            print(f"ERROR: {sp} path does not exist: {path}")
            sys.exit(1)
        mdir = resolve_mask_dir(path)
        if not any(mdir.glob("*.png")):
            print(f"ERROR: no .png masks found for {sp} in {mdir}")
            sys.exit(1)
        print(f"Scanning {sp}: {mdir}")
        scans[sp] = scan(mdir)
    print_report(scans)
    write_outputs(scans, a.out_dir)


if __name__ == "__main__":
    main()
