"""
smart_split_v2.py  v2  —  tile-grouped, class- and pixel-balanced split
=====================================================================
Decides which SOURCE TILE goes to train / val / test, and (optionally) moves
or copies already-cut patches accordingly.

Why the unit is the tile
------------------------
All patches cut from one original image must land in the SAME split.
Overlapping patches from one tile spread across splits leak, and make val/test
scores look better than they are.

What is better than v1
----------------------
  1. PIXEL balance, not just tile counts. Every class gets ~80/10/10 of its
     pixels (not only "at least N tiles"), so a class cannot end up with 60% of
     its pixels in val because one huge tile happened to be assigned there.
  2. KEY CLASSES count double. Water habitats and wet vegetated habitats
     (role from lulc_class_scheme_v2.CLASS_ROLE) get --key-weight (default 2.0)
     in the objective, so Fish Ponds, Rainwater Puddles, Marshes, rice, etc.
     are protected first.
  3. Smarter minimums. Each class wants min(--min-tiles, what is feasible) tiles
     in EACH of val and test, but at least half of its tiles always stay in train
     (v1 could leave a 2-tile class with 0 tiles in train).
  4. Search, not one greedy pass: --trials random restarts (rarest-first order
     with jitter) + a local search that moves single tiles while the objective
     improves. Best trial wins. Deterministic for a given --seed.
  5. Presence needs a real footprint: a tile only "has" a class if it holds at
     least --min-class-px pixels of it (stray pixels do not count).
  6. Overrides can be class NAMES:  --min-tiles-overrides "Marshes:3,Fish Ponds:2"
     (v1 used model ids, which silently change whenever the scheme is reordered).
  7. Works at two levels:
       --masks-dir   raw integer masks of the SOURCE TILES, before patching
                     (what prepare_patches_v2.py --split-strategy smart uses)
       --train/--val/--test   already-cut patch folders (re-split, as v1)

Classes are judged in the MODEL output space (model_class_scheme_v2.RAW_TO_MODEL).
Raw ids with no model channel (e.g. EXCLUDE_RAW) and 255 are IGNORE.

Usage
-----
    # 1) decide the split from the source-tile masks (nothing is copied)
    python smart_split_v2.py --masks-dir ./lulc_int_masks --output ./split_plan

    # 2) re-split patch folders you already have (inspect first):
    python smart_split_v2.py --train P/train --val P/val --test P/test \
        --output P_smart --dry-run
    #    then for real, reusing split_report_v2's CSV so masks are not rescanned:
    python smart_split_v2.py --train P/train --val P/val --test P/test --output P_smart \
        --inventory-csv ./split_analysis/per_image_class_inventory.csv

Key options
-----------
    --ratios 0.8 0.1 0.1     target fractions (train val test), applied to tiles AND pixels
    --min-tiles 4            wanted tiles per class in EACH of val and test (when feasible)
    --min-tiles-overrides    "Class name:n,Class name:n"  (or model ids "9:3")
    --min-class-px 500       pixels a tile needs for a class to count as present
    --key-weight 2.0         extra objective weight for key (malaria-relevant) classes
    --trials 100             random restarts
    --max-patches-per-tile N density cap (patch mode only; keeps rare-class patches first)
    --seed 42  --move  --dry-run
"""

import argparse
import collections
import csv
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from model_class_scheme_v2 import (RAW_TO_MODEL, CLASS_NAMES, NUM_CLASSES,
                                    CLASS_ROLES, KEY_CLASS_IDS)
except ModuleNotFoundError as e:
    print("ERROR: could not import model_class_scheme_v2.py / lulc_class_scheme_v2.py.")
    print("Keep smart_split_v2.py in the same folder as those two files.")
    print(f"Underlying error: {e}")
    sys.exit(1)

SPLITS = ("train", "val", "test")
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# raw pixel value -> model id (-1 = ignore)
RAW_LUT = np.full(256, -1, dtype=np.int16)
for _raw, _mid in RAW_TO_MODEL.items():
    if 0 <= _raw < 256:
        RAW_LUT[_raw] = _mid


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def source_tile_key(filename: str) -> str:
    """'<dataset>__<tilestem>_pNNNN.png' -> '<dataset>__<tilestem>'."""
    stem = Path(filename).stem
    if "_p" in stem:
        head, tail = stem.rsplit("_p", 1)
        if tail.isdigit():
            stem = head
    return stem


def resolve_subdir(split_path: Path, kind: str) -> Path:
    sub = split_path / kind
    return sub if sub.is_dir() else split_path


def find_image(img_dir: Path, filename: str):
    """Image for a patch: same name first, then same stem with any image extension."""
    p = img_dir / filename
    if p.exists():
        return p
    stem = Path(filename).stem
    for ext in IMG_EXTS:
        q = img_dir / f"{stem}{ext}"
        if q.exists():
            return q
    return None


def model_px(mask_arr: np.ndarray):
    """-> (per-model-class pixel counts [NUM_CLASSES], ignore pixel count)."""
    counts = np.bincount(mask_arr.ravel(), minlength=256)[:256]
    out = np.zeros(NUM_CLASSES, dtype=np.int64)
    valid = RAW_LUT >= 0
    np.add.at(out, RAW_LUT[valid], counts[valid])
    return out, int(counts[~valid].sum())


def parse_overrides(s):
    """'Marshes:3,Fish Ponds:2' or '9:3' -> {model_id: n}."""
    out = {}
    if not s:
        return out
    by_name = {v.strip().lower(): k for k, v in CLASS_NAMES.items()}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        key, val = part.rsplit(":", 1)
        key = key.strip()
        if key.isdigit():
            mid = int(key)
        elif key.lower() in by_name:
            mid = by_name[key.lower()]
        else:
            raise SystemExit(f"--min-tiles-overrides: unknown class '{key}'. "
                             f"Known: {sorted(by_name)}")
        out[mid] = int(val)
    return out


# ---------------------------------------------------------------------------
# inventories
# ---------------------------------------------------------------------------
class Patch:
    __slots__ = ("filename", "old_split", "tile", "px")

    def __init__(self, filename, old_split, tile, px):
        self.filename = filename
        self.old_split = old_split
        self.tile = tile
        self.px = px                    # np.int64[NUM_CLASSES]

    @property
    def classes(self):
        return {int(i) for i in np.nonzero(self.px)[0]}


def inventory_from_csv(csv_path: Path):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"inventory CSV is empty: {csv_path}")
    cls_cols = [c for c in rows[0].keys() if c.startswith("m") and c[1:2].isdigit()]
    mid_of = {c: int(c[1:].split("_")[0]) for c in cls_cols}
    # guard against a CSV written under a different class scheme
    if max(mid_of.values()) >= NUM_CLASSES or len(cls_cols) != NUM_CLASSES:
        raise SystemExit(
            f"inventory CSV has {len(cls_cols)} class columns but the current scheme has "
            f"{NUM_CLASSES}. It was made with a different model_class_scheme_v2 — rerun split_report_v2.py.")
    for c, mid in mid_of.items():
        want = CLASS_NAMES[mid].replace(" ", "_")
        got = c.split("_", 1)[1] if "_" in c else ""
        if got != want:
            raise SystemExit(f"inventory CSV column '{c}' does not match scheme class "
                             f"{mid} '{CLASS_NAMES[mid]}'. Rerun split_report_v2.py.")
    patches = []
    for r in rows:
        px = np.zeros(NUM_CLASSES, dtype=np.int64)
        for c, mid in mid_of.items():
            px[mid] = int(float(r[c]))
        patches.append(Patch(r["filename"], r["split"], r["source_tile"], px))
    return patches


def inventory_from_scan(split_paths: dict):
    patches = []
    for split, path in split_paths.items():
        mdir = resolve_subdir(path, "masks")
        files = sorted(mdir.glob("*.png"))
        print(f"  scanning {split}: {len(files)} masks in {mdir}")
        for mp in files:
            px, _ = model_px(np.array(Image.open(mp)))
            patches.append(Patch(mp.name, split, source_tile_key(mp.name), px))
    return patches


def tile_matrix_from_patches(patches):
    tiles = sorted({p.tile for p in patches})
    idx = {t: i for i, t in enumerate(tiles)}
    PX = np.zeros((len(tiles), NUM_CLASSES), dtype=np.int64)
    for p in patches:
        PX[idx[p.tile]] += p.px
    return tiles, PX


def tile_matrix_from_masks(mask_dir: Path, stems=None):
    files = sorted(mask_dir.glob("*.png"))
    if stems is not None:
        stems = set(stems)
        files = [f for f in files if f.stem in stems]
    if not files:
        raise SystemExit(f"no .png masks in {mask_dir}")
    tiles, rows = [], []
    for mp in files:
        px, _ = model_px(np.array(Image.open(mp)))
        tiles.append(mp.stem)
        rows.append(px)
    return tiles, np.vstack(rows)


# ---------------------------------------------------------------------------
# the assignment core
# ---------------------------------------------------------------------------
def class_need(total_tiles: int, nominal: int):
    """(val_need, test_need): tiles wanted in EACH of val/test for a class that
    lives in `total_tiles` tiles. At least half of its tiles stay in train."""
    if total_tiles < 2:
        return 0, 0
    if total_tiles == 2:
        return 1, 0                       # one tile can be measured; test stays empty (warned)
    train_floor = int(np.ceil(total_tiles * 0.5))
    spare = total_tiles - train_floor
    each = max(1, min(nominal, spare // 2))
    return each, each


def assign_tiles(tiles, PX, ratios=(0.8, 0.1, 0.1), min_tiles=4, min_overrides=None,
                 min_class_px=500, key_weight=2.0, trials=100, seed=42,
                 mu=0.3, nu=20.0, verbose=True):
    """
    tiles : list of tile ids          PX : int array [n_tiles, NUM_CLASSES] model-class pixels
    Returns (assign {tile: split}, info dict).
    Objective J (lower is better):
        sum_c w_c * sum_sp (share_{c,sp} - ratio_sp)^2          pixel balance
      + mu * sum_c w_c * sum_{val,test} max(0, need - tiles)    unmet minimum tiles
      + nu * sum_sp (n_sp / N - ratio_sp)^2                     tile-count balance
    """
    min_overrides = min_overrides or {}
    T, C = PX.shape
    r = np.asarray(ratios, dtype=float)
    r = r / r.sum()
    PX = PX.astype(np.float64)
    PRES = (PX >= min_class_px)
    PRES[:, 0] = False                       # background is never constrained
    tot_px = PX.sum(0)
    tot_tiles = PRES.sum(0)
    active = (tot_px > 0) & (tot_tiles > 0)
    active[0] = False

    # class weights: rarer (by tile count) = heavier; key classes extra
    w = np.zeros(C)
    for c in range(C):
        if active[c]:
            w[c] = 1.0 / np.sqrt(tot_tiles[c])
    if w[active].size:
        w[active] /= w[active].mean()
    for c in KEY_CLASS_IDS:
        if c < C and active[c]:
            w[c] *= key_weight

    need = np.zeros((3, C))
    for c in range(C):
        if not active[c]:
            continue
        nv, nt = class_need(int(tot_tiles[c]), min_overrides.get(c, min_tiles))
        need[1, c], need[2, c] = nv, nt

    inv_tot = np.where(active, 1.0 / np.maximum(tot_px, 1.0), 0.0)

    def J(S, K, N):
        share = S * inv_tot
        pix = (w * (share - r[:, None]) ** 2).sum()
        miss = mu * (w * np.maximum(0.0, need - K)).sum()
        cnt = nu * ((N / T - r) ** 2).sum()
        return pix + miss + cnt

    # rarity key: tile's rarest present class (ascending = rarest first)
    big = float(T + 1)
    rarity = np.array([min([tot_tiles[c] for c in np.nonzero(PRES[t])[0]] or [big])
                       for t in range(T)], dtype=float)

    def put(S, K, N, t, sp, sign):
        S[sp] += sign * PX[t]
        K[sp] += sign * PRES[t]
        N[sp] += sign

    def state_from(a):
        S = np.zeros((3, C)); K = np.zeros((3, C)); N = np.zeros(3)
        for t in range(T):
            put(S, K, N, t, a[t], +1)
        return S, K, N

    def moves(a, S, K, N, rng, max_pass=25):
        """Single-tile moves between splits while J improves."""
        cur = J(S, K, N)
        for _ in range(max_pass):
            improved = False
            tl = list(range(T))
            rng.shuffle(tl)
            for t in tl:
                src = a[t]
                put(S, K, N, t, src, -1)
                bj, bs = None, src
                for sp in range(3):
                    put(S, K, N, t, sp, +1)
                    j = J(S, K, N)
                    put(S, K, N, t, sp, -1)
                    if bj is None or j < bj - 1e-15:
                        bj, bs = j, sp
                if bs != src and bj < cur - 1e-12:
                    cur, improved = bj, True
                else:
                    bs = src
                put(S, K, N, t, bs, +1)
                a[t] = bs
            if not improved:
                break
        return cur

    def swaps(a, S, K, N, max_pass=4):
        """Swap two tiles that sit in different splits (keeps the tile counts) while J improves."""
        cur = J(S, K, N)
        for _ in range(max_pass):
            improved = False
            for t in [t for t in range(T) if a[t] != 0]:
                for u in range(T):
                    st, su = a[t], a[u]
                    if st == su:
                        continue
                    put(S, K, N, t, st, -1); put(S, K, N, u, su, -1)
                    put(S, K, N, t, su, +1); put(S, K, N, u, st, +1)
                    j = J(S, K, N)
                    if j < cur - 1e-12:
                        cur, improved = j, True
                        a[t], a[u] = su, st
                    else:
                        put(S, K, N, t, su, -1); put(S, K, N, u, st, -1)
                        put(S, K, N, t, st, +1); put(S, K, N, u, su, +1)
            if not improved:
                break
        return cur

    results = []
    for k in range(max(1, trials)):
        rng = random.Random(seed * 100003 + k)
        keys = [(rarity[t] * (1.0 if k == 0 else 1.0 + 0.6 * rng.random()), rng.random(), t)
                for t in range(T)]
        order = [t for _, _, t in sorted(keys)]
        S = np.zeros((3, C)); K = np.zeros((3, C)); N = np.zeros(3)
        a = np.full(T, -1, dtype=int)
        for t in order:                      # greedy placement, rarest tiles first
            bj, bs = None, None
            for sp in range(3):
                put(S, K, N, t, sp, +1)
                j = J(S, K, N)
                put(S, K, N, t, sp, -1)
                if bj is None or j < bj - 1e-15:
                    bj, bs = j, sp
            put(S, K, N, t, bs, +1)
            a[t] = bs
        results.append((moves(a, S, K, N, rng), k, a.copy()))

    # swap refinement on the best few trials only (swaps are the expensive move)
    best = None
    for _, k, a in sorted(results, key=lambda x: x[0])[:3]:
        a = a.copy()
        rng = random.Random(seed * 7919 + k)
        S, K, N = state_from(a)
        swaps(a, S, K, N)
        moves(a, S, K, N, rng)
        final = J(S, K, N)
        if best is None or final < best[0] - 1e-15:
            best = (final, a.copy(), S.copy(), K.copy(), N.copy())
    final, a, S, K, N = best
    assign = {tiles[t]: SPLITS[a[t]] for t in range(T)}

    warnings = []
    for c in range(1, C):
        if not active[c]:
            continue
        for sp, name in ((1, "val"), (2, "test")):
            if K[sp, c] == 0:
                warnings.append(f"{CLASS_NAMES[c]}: 0 tiles in {name} — cannot be measured there "
                                f"(class lives in {int(tot_tiles[c])} tile(s) in total)")
            elif K[sp, c] < need[sp, c]:
                warnings.append(f"{CLASS_NAMES[c]}: only {int(K[sp, c])} tile(s) in {name} "
                                f"(wanted {int(need[sp, c])}; class lives in {int(tot_tiles[c])} tile(s))")
        if K[0, c] == 0:
            warnings.append(f"{CLASS_NAMES[c]}: 0 tiles in TRAIN — the model cannot learn it")
    for c in range(1, C):
        if tot_px[c] > 0 and tot_tiles[c] == 0:
            warnings.append(f"{CLASS_NAMES[c]}: has pixels but no tile reaches --min-class-px "
                            f"({min_class_px}); it is not constrained by the split")

    info = {"objective": float(final), "trials": int(trials), "S": S, "K": K, "N": N,
            "tot_px": tot_px, "tot_tiles": tot_tiles, "active": active, "need": need,
            "warnings": warnings}
    if verbose:
        print(f"  best objective {final:.4f} over {max(1, trials)} trial(s)")
    return assign, info


# ---------------------------------------------------------------------------
# density cap (patch mode)
# ---------------------------------------------------------------------------
def apply_density_cap(patches, cap, seed):
    if not cap:
        return {p.filename for p in patches}
    freq = collections.Counter()
    for p in patches:
        for m in p.classes:
            freq[m] += 1

    def score(p):
        fg = [m for m in p.classes if m != 0]
        return max(1.0 / freq[m] for m in fg) if fg else 0.0

    by_tile = collections.defaultdict(list)
    for p in patches:
        by_tile[p.tile].append(p)
    keep, rng = set(), random.Random(seed)
    for tile, plist in by_tile.items():
        if len(plist) <= cap:
            keep.update(p.filename for p in plist)
            continue
        ordered = sorted(plist, key=lambda p: (score(p), rng.random()), reverse=True)
        keep.update(p.filename for p in ordered[:cap])
    return keep


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def print_report(info, ratios, patch_counts=None):
    S, K, N = info["S"], info["K"], info["N"]
    tot_px, tot_tiles, active = info["tot_px"], info["tot_tiles"], info["active"]
    T = int(N.sum())
    print("\n" + "=" * 100)
    print("SMART SPLIT — RESULT")
    print("=" * 100)
    for i, s in enumerate(SPLITS):
        print(f"  {s:5s}: {int(N[i]):4d} tiles ({100 * N[i] / T:4.1f}%)"
              + (f" | {patch_counts[s]:6,} patches" if patch_counts else ""))
    print(f"  TOTAL: {T} tiles   (targets {ratios[0]:.0%}/{ratios[1]:.0%}/{ratios[2]:.0%})\n")

    hdr = (f"{'mdl':>3} {'class':<24} {'role':<7} | {'tiles':>5} | tiles tr/va/te | "
           f"% of class pixels tr/va/te")
    print(hdr)
    print("-" * len(hdr))
    order = sorted(range(1, NUM_CLASSES), key=lambda c: tot_tiles[c])
    for c in order:
        role = "KEY" if c in KEY_CLASS_IDS else ""
        if not active[c]:
            print(f"{c:>3} {CLASS_NAMES[c]:<24} {role:<7} | {int(tot_tiles[c]):>5} |   NO DATA in this "
                  f"inventory — class is untrainable until annotated")
            continue
        sh = [100 * S[i, c] / tot_px[c] for i in range(3)]
        flag = ""
        if K[1, c] == 0 or K[2, c] == 0:
            flag = "  <-- MISSING in val/test"
        elif K[0, c] == 0:
            flag = "  <-- MISSING in train"
        elif not (3 <= sh[1] <= 20 and 3 <= sh[2] <= 20):
            flag = "  <-- val/test pixel share outside 3-20%"
        print(f"{c:>3} {CLASS_NAMES[c]:<24} {role:<7} | {int(tot_tiles[c]):>5} | "
              f"  {int(K[0, c]):>3}/{int(K[1, c]):>2}/{int(K[2, c]):>2}    | "
              f"  {sh[0]:5.1f} / {sh[1]:5.1f} / {sh[2]:5.1f}{flag}")
    if info["warnings"]:
        print("\nCONSTRAINT WARNINGS:")
        for w in info["warnings"]:
            print("  - " + w)


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------
def write_manifest(out_dir: Path, assign, settings, patches=None, kept=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "smart_split_manifest.json", "w") as f:
        json.dump({"settings": settings, "tile_assignment": assign}, f, indent=2)
    if patches is not None:
        with open(out_dir / "smart_split_assignment.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["new_split", "filename", "source_tile", "kept"])
            for p in patches:
                w.writerow([assign[p.tile], p.filename, p.tile, int(p.filename in kept)])
    else:
        with open(out_dir / "smart_split_assignment.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["new_split", "tile"])
            for t, s in sorted(assign.items()):
                w.writerow([s, t])


def write_split(patches, assign, kept, split_paths, out_dir, move, dry_run):
    if dry_run:
        print(f"\n[dry-run] Wrote manifest + assignment CSV to {out_dir}. No files copied.")
        return
    src_img = {s: resolve_subdir(split_paths[s], "images") for s in SPLITS}
    src_msk = {s: resolve_subdir(split_paths[s], "masks") for s in SPLITS}
    for s in SPLITS:
        (out_dir / s / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / s / "masks").mkdir(parents=True, exist_ok=True)
    op = shutil.move if move else shutil.copyfile
    n = 0
    for p in patches:
        if p.filename not in kept:
            continue
        dst = assign[p.tile]
        img_src = find_image(src_img[p.old_split], p.filename)
        msk_src = src_msk[p.old_split] / p.filename
        if img_src is None or not msk_src.exists():
            print(f"  WARNING: missing source for {p.filename} "
                  f"(img={img_src is not None} msk={msk_src.exists()})")
            continue
        op(str(img_src), str(out_dir / dst / "images" / img_src.name))
        op(str(msk_src), str(out_dir / dst / "masks" / p.filename))
        n += 1
        if n % 5000 == 0:
            print(f"  ...{'moved' if move else 'copied'} {n:,} pairs")
    print(f"\nDone: {'moved' if move else 'copied'} {n:,} image+mask pairs into {out_dir}")


# ---------------------------------------------------------------------------
# API used by prepare_patches_v2.py
# ---------------------------------------------------------------------------
def smart_split_from_masks(mask_dir: Path, ratios=(0.8, 0.1, 0.1), seed=42, stems=None, **kw):
    """Tile-level smart split from raw integer masks. Returns ({tile_stem: split}, info).
    `stems` limits the tiles considered (e.g. only masks that have an image)."""
    tiles, PX = tile_matrix_from_masks(Path(mask_dir), stems)
    assign, info = assign_tiles(tiles, PX, ratios=ratios, seed=seed, **kw)
    print_report(info, ratios)
    return assign, info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Tile-grouped, class- and pixel-balanced smart split.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--masks-dir", type=Path, default=None,
                   help="Raw integer masks of the source tiles (pre-patching). Writes the plan only.")
    p.add_argument("--train", type=Path)
    p.add_argument("--val", type=Path)
    p.add_argument("--test", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--inventory-csv", type=Path, default=None,
                   help="Reuse split_report_v2.py's per_image_class_inventory.csv (patch mode).")
    p.add_argument("--ratios", type=float, nargs=3, default=[0.8, 0.1, 0.1],
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--min-tiles", type=int, default=4)
    p.add_argument("--min-tiles-overrides", type=str, default="",
                   help="'Class name:n,Class name:n' or 'modelid:n'.")
    p.add_argument("--min-class-px", type=int, default=500)
    p.add_argument("--key-weight", type=float, default=2.0)
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--max-patches-per-tile", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--move", action="store_true", help="Move instead of copy (patch mode).")
    p.add_argument("--dry-run", action="store_true", help="Plan + report only; copy nothing.")
    return p.parse_args()


def main():
    a = parse_args()
    overrides = parse_overrides(a.min_tiles_overrides)
    settings = {"ratios": a.ratios, "min_tiles": a.min_tiles, "overrides": overrides,
                "min_class_px": a.min_class_px, "key_weight": a.key_weight,
                "trials": a.trials, "seed": a.seed, "num_classes": NUM_CLASSES}
    kw = dict(ratios=tuple(a.ratios), min_tiles=a.min_tiles, min_overrides=overrides,
              min_class_px=a.min_class_px, key_weight=a.key_weight, trials=a.trials, seed=a.seed)

    if a.masks_dir:
        if not a.masks_dir.exists():
            print(f"ERROR: --masks-dir missing: {a.masks_dir}"); sys.exit(1)
        print("Reading tile masks...")
        tiles, PX = tile_matrix_from_masks(a.masks_dir)
        print(f"  {len(tiles)} tiles")
        assign, info = assign_tiles(tiles, PX, **kw)
        print_report(info, a.ratios)
        write_manifest(a.output, assign, settings)
        print(f"\nWrote plan to {a.output}  (use it with prepare_patches_v2.py --split-strategy smart)")
        return

    if not (a.train and a.val and a.test):
        print("ERROR: give --masks-dir, or all of --train --val --test."); sys.exit(1)
    split_paths = {"train": a.train, "val": a.val, "test": a.test}
    for s, pth in split_paths.items():
        if not pth.exists():
            print(f"ERROR: {s} path missing: {pth}"); sys.exit(1)

    print("Building inventory...")
    if a.inventory_csv:
        if not a.inventory_csv.exists():
            print(f"ERROR: inventory CSV not found: {a.inventory_csv}"); sys.exit(1)
        patches = inventory_from_csv(a.inventory_csv)
        print(f"  loaded {len(patches):,} patches from {a.inventory_csv}")
    else:
        patches = inventory_from_scan(split_paths)
        print(f"  scanned {len(patches):,} patches")

    tiles, PX = tile_matrix_from_patches(patches)
    print(f"  {len(tiles)} source tiles")
    assign, info = assign_tiles(tiles, PX, **kw)

    kept = apply_density_cap(patches, a.max_patches_per_tile, a.seed)
    if a.max_patches_per_tile:
        print(f"\nDensity cap {a.max_patches_per_tile}/tile: keeping {len(kept):,} of {len(patches):,} patches")
    patch_counts = collections.Counter(assign[p.tile] for p in patches if p.filename in kept)
    print_report(info, a.ratios, patch_counts)

    a.output.mkdir(parents=True, exist_ok=True)
    write_manifest(a.output, assign, settings, patches, kept)
    write_split(patches, assign, kept, split_paths, a.output, a.move, a.dry_run)


if __name__ == "__main__":
    main()
