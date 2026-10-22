"""
prepare_patches_v2.py  v2
======================
One generic patch-preparation pipeline for ANY CVAT segmentation export.
Same CLI and datasets.json format as v1; improved for the 24-label LULC set.

Stages
------
1. Parse labelmap.txt -> colour -> raw id (by NAME, via lulc_class_scheme_v2) and
   PRINT the mapping table (name, colour, raw id, model channel). Unknown
   spellings are flagged loudly (--strict-labels aborts on them).
   Use --check-labels to run only this stage on a new export.
2. Convert RGB masks -> raw integer masks (vectorised; ~50x faster than v1's
   one-pass-per-colour loop) and write raw_pixel_counts.json.
3. Split SOURCE TILES into train/val/test:
       smart  (default) class- AND pixel-balanced, key classes protected —
                        see smart_split_v2.py; writes smart_split_manifest.json
       tile   random, whole tiles
       site   random, by site id (<site>_<index>)
4. Patch (optionally resize first), filter, save.
5. Write dataset_info.json and per_image_class_inventory.csv (the exact format
   split_report_v2.py / smart_split_v2.py read, so nothing has to be rescanned).

Patch keep rule (all counted on the patch, raw ids)
---------------------------------------------------
  keep if  target-class pixels >= --min-target-px
      or   RARE-class pixels   >= --rare-min-px        (rare = class present in
                                                        < --rare-tiles source tiles;
                                                        never dropped by the next rule)
      or   random < --bg-keep-frac                      (optional hard negatives)
  but drop (non-rare only) if unlabelled (255) share > --max-ignore-frac.
Edge strips are covered: the last row/column of patches is shifted inward
instead of being silently dropped (v1 lost up to patch_size-1 px per side).

Target classes default to every mapped raw id that has a MODEL channel
(model_class_scheme_v2.RAW_TO_MODEL), so excluded classes (EXCLUDE_RAW) never
cause patches to be kept.

Usage
-----
    # validate the labelmap of a new export (fast, writes nothing):
    python prepare_patches_v2.py --name lulc_v2 --annotations D:/LULC_300/annotations --check-labels

    # full run:
    python prepare_patches_v2.py --name lulc_v2 \
        --src-images "D:/LULC_300/images" \
        --annotations "D:/LULC_300/annotations" \
        --output "D:/LULC_300/patches" \
        --split-strategy smart --patch-size 512 --overlap 64 --min-target-px 500

    # or from the config file:
    python prepare_patches_v2.py --config datasets.json --dataset lulc_v2

    # reuse already-converted masks (no labelmap needed):
    python prepare_patches_v2.py --name lulc_v2_fullres --int-masks D:/LULC_300/lulc_v2_int_masks \
        --skip-mask-conversion --src-images D:/LULC_300/images --output D:/LULC_300/patches_full

Then:   python split_report_v2.py --train OUT/train --val OUT/val --test OUT/test
"""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(2 ** 40))
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

try:
    from tqdm import tqdm
except ImportError:                      # tqdm is optional
    def tqdm(it, **kw):
        return it

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lulc_class_scheme_v2 import NAME_TO_RAW, RAW_NAMES, REMOVED_NAMES, _norm

try:
    from model_class_scheme_v2 import RAW_TO_MODEL, CLASS_NAMES, NUM_CLASSES, RAW_CLASSES
    HAVE_MODEL = True
except Exception as _e:                  # patching still works without the model scheme
    HAVE_MODEL = False
    RAW_TO_MODEL, CLASS_NAMES, NUM_CLASSES, RAW_CLASSES = {}, {}, 0, []
    print(f"WARNING: model_class_scheme_v2.py not importable ({_e}). Continuing without it: "
          f"target classes = all mapped raw ids; no per_image_class_inventory.csv.")

IMG_EXTS = (".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG", ".tif", ".tiff", ".TIF", ".TIFF")

# raw pixel value -> model id (-1 = ignore)
RAW_LUT = np.full(256, -1, dtype=np.int16)
for _raw, _mid in RAW_TO_MODEL.items():
    if 0 <= _raw < 256:
        RAW_LUT[_raw] = _mid


# ==============================================================================
# STAGE 1: labelmap -> colour LUT (+ visible mapping table)
# ==============================================================================

def parse_labelmap(labelmap_path: Path, strict: bool = False):
    """Returns color_to_raw {(r,g,b): raw}. Prints the mapping table."""
    color_to_raw, rows, unknown = {}, [], []
    with open(labelmap_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, rest = line.split(":", 1)
            name = name.strip()
            r, g, b = [int(v) for v in rest.split(":")[0].split(",")]
            raw = NAME_TO_RAW.get(_norm(name))
            if raw is None:
                if _norm(name) in REMOVED_NAMES:
                    status = "removed -> ignore"
                else:
                    status = "UNMAPPED -> ignore  <-- check spelling in lulc_class_scheme_v2.NAME_TO_RAW"
                    unknown.append(name)
            else:
                if (r, g, b) in color_to_raw and color_to_raw[(r, g, b)] != raw:
                    print(f"  WARNING: colour {(r, g, b)} is used by '{name}' AND by raw "
                          f"{color_to_raw[(r, g, b)]} — the later one wins.")
                color_to_raw[(r, g, b)] = raw
                status = ""
            rows.append((name, (r, g, b), raw, status))

    print(f"\n[Stage 1] labelmap: {labelmap_path}")
    print(f"  {'labelmap name':<28} {'colour':<16} {'raw':>4}  {'raw name':<24} {'model':>5}")
    for name, col, raw, status in rows:
        if raw is None:
            print(f"  {name:<28} {str(col):<16} {'-':>4}  {status}")
            continue
        mid = RAW_TO_MODEL.get(raw)
        mtxt = "ign" if (HAVE_MODEL and mid is None) else (str(mid) if mid is not None else "?")
        print(f"  {name:<28} {str(col):<16} {raw:>4}  {RAW_NAMES.get(raw, '?'):<24} {mtxt:>5}")
    if unknown:
        msg = f"{len(unknown)} labelmap name(s) have no mapping: {unknown}"
        if strict:
            print(f"ERROR: {msg}")
            sys.exit(1)
        print(f"  WARNING: {msg}  (their pixels become ignore=255)")
    return color_to_raw


# ==============================================================================
# STAGE 2: RGB masks -> raw integer masks (vectorised)
# ==============================================================================

def build_palette(color_to_raw: dict):
    keys = np.array([(r << 16) | (g << 8) | b for (r, g, b) in color_to_raw], dtype=np.uint32)
    raws = np.array(list(color_to_raw.values()), dtype=np.uint8)
    order = np.argsort(keys)
    return keys[order], raws[order]


def rgb_to_raw_mask(rgb: np.ndarray, palette, chunk_rows: int = 1024) -> np.ndarray:
    """Map each RGB pixel -> raw class id. Colours not in the palette -> 255 (ignore)."""
    keys, raws = palette
    h, w = rgb.shape[:2]
    out = np.full((h, w), 255, dtype=np.uint8)
    if len(keys) == 0:
        return out
    for y0 in range(0, h, chunk_rows):
        blk = rgb[y0:y0 + chunk_rows].astype(np.uint32)
        k = (blk[..., 0] << 16) | (blk[..., 1] << 8) | blk[..., 2]
        idx = np.searchsorted(keys, k)
        idx[idx >= len(keys)] = len(keys) - 1
        hit = keys[idx] == k
        out[y0:y0 + chunk_rows] = np.where(hit, raws[idx], 255)
    return out


def convert_all_masks(color_to_raw: dict, seg_class_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    palette = build_palette(color_to_raw)
    mask_files = sorted(seg_class_dir.glob("*.png"))
    if not mask_files:
        print(f"ERROR: no .png masks in {seg_class_dir}")
        sys.exit(1)
    counts = {}
    for mask_path in tqdm(mask_files, desc="[Stage 2] Converting RGB->raw masks"):
        rgb = np.array(Image.open(mask_path).convert("RGB"))
        raw_mask = rgb_to_raw_mask(rgb, palette)
        counts[mask_path.stem] = np.bincount(raw_mask.ravel(), minlength=256)[:256]
        Image.fromarray(raw_mask).save(out_dir / mask_path.name)
    save_raw_counts(out_dir, counts)
    print(f"[Stage 2] Saved {len(mask_files)} raw-index masks to {out_dir}")
    return counts


def save_raw_counts(int_masks_dir: Path, counts: dict):
    data = {stem: {str(i): int(c) for i, c in enumerate(arr) if c > 0} for stem, arr in counts.items()}
    with open(int_masks_dir / "raw_pixel_counts.json", "w") as f:
        json.dump(data, f)


def load_raw_counts(int_masks_dir: Path):
    """Per-tile raw pixel counts; reads the cached JSON, rescans if missing/incomplete."""
    stems = sorted(p.stem for p in int_masks_dir.glob("*.png"))
    cache = int_masks_dir / "raw_pixel_counts.json"
    counts = {}
    if cache.exists():
        with open(cache) as f:
            data = json.load(f)
        for stem, d in data.items():
            arr = np.zeros(256, dtype=np.int64)
            for k, v in d.items():
                arr[int(k)] = v
            counts[stem] = arr
    missing = [s for s in stems if s not in counts]
    if missing:
        for s in tqdm(missing, desc="[Stage 2] Scanning masks for class counts"):
            counts[s] = np.bincount(np.array(Image.open(int_masks_dir / f"{s}.png")).ravel(),
                                    minlength=256)[:256]
        save_raw_counts(int_masks_dir, counts)
    return {s: counts[s] for s in stems}


def report_class_counts(counts: dict, mapped_raws, min_px: int):
    """Class-by-class pixel and tile counts of the converted masks; flags NO DATA."""
    tot = np.zeros(256, dtype=np.int64)
    ntiles = np.zeros(256, dtype=np.int64)
    for arr in counts.values():
        tot += arr
        ntiles += (arr >= min_px)
    print(f"\n[Stage 2] Class inventory over {len(counts)} tiles "
          f"(tiles = tiles with >= {min_px} px of the class)")
    print(f"  {'raw':>4} {'class':<26} {'model':>5} {'pixels':>14} {'tiles':>6}")
    no_data = []
    for raw in sorted(set(mapped_raws) | {r for r in range(255) if tot[r] > 0}):
        mid = RAW_TO_MODEL.get(raw)
        mtxt = "ign" if (HAVE_MODEL and mid is None) else (str(mid) if mid is not None else "?")
        flag = ""
        if raw != 0 and tot[raw] == 0:
            flag = "  <-- NO DATA"
            if mid is not None:
                no_data.append(raw)
        print(f"  {raw:>4} {RAW_NAMES.get(raw, f'unknown_{raw}'):<26} {mtxt:>5} "
              f"{int(tot[raw]):>14,} {int(ntiles[raw]):>6}{flag}")
    ign = int(tot[255])
    print(f"  ignore (255): {ign:,} px")
    if no_data:
        print("  NOTE: model classes with NO DATA: "
              + ", ".join(f"{r} {RAW_NAMES.get(r)}" for r in no_data)
              + "\n        Either annotate them, or list them in model_class_scheme_v2.EXCLUDE_RAW "
                "so the head does not carry an untrainable channel.")
    return tot, ntiles


# ==============================================================================
# STAGE 3: splits
# ==============================================================================

def site_split(stems, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Split by site id (filename convention: <site_id>_<index>)."""
    site_ids = sorted(set("_".join(s.split("_")[:-1]) for s in stems))
    if len(site_ids) < 3:
        print(f"  WARNING: only {len(site_ids)} site id(s) found from file names "
              f"(<site>_<index>) — falling back to a random tile split.")
        return tile_split(stems, train_ratio, val_ratio, seed)
    rng = random.Random(seed)
    rng.shuffle(site_ids)
    n = len(site_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_sites = set(site_ids[:n_train])
    val_sites = set(site_ids[n_train:n_train + n_val])
    if not val_sites and len(train_sites) > 1:
        moved = sorted(train_sites)[-1]
        train_sites.discard(moved)
        val_sites.add(moved)
        print(f"  INFO: moved site '{moved}' train->val to avoid empty val split")
    split_map = {}
    for stem in stems:
        site = "_".join(stem.split("_")[:-1])
        split_map[stem] = "train" if site in train_sites else ("val" if site in val_sites else "test")
    return split_map


def tile_split(stems, train_ratio=0.8, val_ratio=0.1, seed=42):
    stems = list(stems)
    random.Random(seed).shuffle(stems)
    n = len(stems)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {s: ("train" if i < n_train else ("val" if i < n_train + n_val else "test"))
            for i, s in enumerate(stems)}


def make_split(strategy, int_masks_dir, stems, s, output):
    if strategy == "smart":
        import smart_split_v2
        ratios = (s["train_ratio"], s["val_ratio"], max(0.0, 1.0 - s["train_ratio"] - s["val_ratio"]))
        overrides = smart_split_v2.parse_overrides(s["min_tiles_overrides"])
        assign, _ = smart_split_v2.smart_split_from_masks(
            int_masks_dir, ratios=ratios, seed=s["seed"], stems=stems,
            min_tiles=s["min_tiles"], min_overrides=overrides,
            min_class_px=s["min_class_px"], key_weight=s["key_weight"], trials=s["trials"])
        output.mkdir(parents=True, exist_ok=True)
        with open(output / "smart_split_manifest.json", "w") as f:
            json.dump({"settings": {k: s[k] for k in ("train_ratio", "val_ratio", "seed", "min_tiles",
                                                      "min_class_px", "key_weight", "trials")},
                       "tile_assignment": assign}, f, indent=2)
        return assign
    fn = site_split if strategy == "site" else tile_split
    split_map = fn(stems, s["train_ratio"], s["val_ratio"], s["seed"])
    c = {sp: sum(1 for v in split_map.values() if v == sp) for sp in ("train", "val", "test")}
    print(f"[Stage 3] {strategy} split: train={c['train']} | val={c['val']} | test={c['test']} tiles")
    return split_map


# ==============================================================================
# STAGE 4: resize -> patch -> filter -> save
# ==============================================================================

def window_starts(n: int, size: int, stride: int):
    """Patch start offsets covering [0, n); the last window is shifted inward so the edge is covered."""
    if n < size:
        return []
    starts = list(range(0, n - size + 1, stride))
    if starts[-1] + size < n:
        starts.append(n - size)
    return starts


def model_counts(counts256: np.ndarray):
    out = np.zeros(NUM_CLASSES, dtype=np.int64)
    valid = RAW_LUT >= 0
    np.add.at(out, RAW_LUT[valid], counts256[valid])
    return out


def find_image(src_images_dir: Path, stem: str):
    for ext in IMG_EXTS:
        c = src_images_dir / f"{stem}{ext}"
        if c.exists():
            return c
    return None


def read_image_bgr(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:                                   # cv2 cannot open some TIFFs / huge files
        img = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    return img


def create_patches(split_map, src_images_dir, int_masks_dir, output_dir, resize_to, patch_size,
                   overlap, target_classes, min_target_px, rare_raws, rare_min_px, max_ignore_frac,
                   bg_keep_frac, img_format, seed, name_prefix="", dataset_name=""):
    stride = patch_size - overlap
    if stride <= 0:
        print("ERROR: overlap must be smaller than patch_size")
        sys.exit(1)
    rng = random.Random(seed)
    tgt = np.array(sorted(set(target_classes)), dtype=int)
    rare = np.array(sorted(set(rare_raws)), dtype=int)
    stats = {s: {"total": 0, "kept": 0, "skipped": 0, "rare_rescued": 0, "ignore_dropped": 0}
             for s in ("train", "val", "test")}
    inventory = []                                    # one row per kept patch
    img_ext = ".jpg" if img_format == "jpg" else ".png"
    png_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    jpg_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    skipped_tiles = []

    for split in ("train", "val", "test"):
        out_img_dir = output_dir / split / "images"
        out_mask_dir = output_dir / split / "masks"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_mask_dir.mkdir(parents=True, exist_ok=True)

        for stem in tqdm([k for k, v in split_map.items() if v == split], desc=f"[Stage 4] Patching {split}"):
            img_path = find_image(src_images_dir, stem)
            mask_path = int_masks_dir / f"{stem}.png"
            if img_path is None or not mask_path.exists():
                skipped_tiles.append(stem)
                continue
            img = read_image_bgr(img_path)
            int_mask = np.array(Image.open(mask_path))

            if resize_to is not None:
                img = cv2.resize(img, (resize_to, resize_to), interpolation=cv2.INTER_AREA)
                int_mask = cv2.resize(int_mask, (resize_to, resize_to), interpolation=cv2.INTER_NEAREST)
            elif img.shape[:2] != int_mask.shape[:2]:
                print(f"  WARNING: {stem}: image {img.shape[:2]} != mask {int_mask.shape[:2]} "
                      f"— skipped (use --resize-to, or fix the export)")
                skipped_tiles.append(stem)
                continue

            h, w = int_mask.shape[:2]
            ys, xs = window_starts(h, patch_size, stride), window_starts(w, patch_size, stride)
            patch_id = 0
            area = float(patch_size * patch_size)
            for y in ys:
                for x in xs:
                    stats[split]["total"] += 1
                    mp = int_mask[y:y + patch_size, x:x + patch_size]
                    cnt = np.bincount(mp.ravel(), minlength=256)[:256]
                    target_px = int(cnt[tgt].sum()) if tgt.size else 0
                    rare_px = int(cnt[rare].sum()) if rare.size else 0
                    ign_frac = cnt[255] / area

                    keep, rescued = False, False
                    if target_px >= min_target_px:
                        keep = True
                    if rare_px >= rare_min_px:
                        keep, rescued = True, (target_px < min_target_px)
                    if not keep and bg_keep_frac > 0 and rng.random() < bg_keep_frac:
                        keep = True
                    if keep and rare_px < rare_min_px and ign_frac > max_ignore_frac:
                        stats[split]["ignore_dropped"] += 1
                        keep = False
                    if not keep:
                        stats[split]["skipped"] += 1
                        continue

                    name = f"{name_prefix}{stem}_p{patch_id:04d}"
                    ip = img[y:y + patch_size, x:x + patch_size]
                    cv2.imwrite(str(out_img_dir / f"{name}{img_ext}"), ip,
                                jpg_params if img_format == "jpg" else png_params)
                    cv2.imwrite(str(out_mask_dir / f"{name}.png"), mp, png_params)
                    stats[split]["kept"] += 1
                    stats[split]["rare_rescued"] += int(rescued)
                    patch_id += 1

                    if HAVE_MODEL:
                        mc = model_counts(cnt)
                        inventory.append([split, f"{name}.png", dataset_name or name_prefix.rstrip("_"),
                                          f"{name_prefix}{stem}", int(mc[1:].sum() > 0),
                                          int(cnt[RAW_LUT < 0].sum())] + mc.tolist())

    for split, st in stats.items():
        pct = 100 * st["kept"] / st["total"] if st["total"] else 0
        print(f"[Stage 4] {split:5s}: windows={st['total']:7d} | kept={st['kept']:6d} ({pct:.0f}%) | "
              f"skipped={st['skipped']:6d} | rare-rescued={st['rare_rescued']:5d} | "
              f"dropped for >{max_ignore_frac:.0%} unlabelled={st['ignore_dropped']:5d}")
    if skipped_tiles:
        print(f"  WARNING: {len(skipped_tiles)} tile(s) had no image / mask pair: {skipped_tiles[:10]}"
              f"{' ...' if len(skipped_tiles) > 10 else ''}")
    return stats, inventory


# ==============================================================================
# STAGE 5: provenance + inventory
# ==============================================================================

def save_inventory(output_dir: Path, inventory, filename):
    if not HAVE_MODEL or not inventory:
        return None
    cols = [f"m{mid}_{CLASS_NAMES.get(mid, mid)}".replace(" ", "_") for mid in range(NUM_CLASSES)]
    path = output_dir / filename
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "filename", "dataset", "source_tile", "has_foreground", "ignore_px"] + cols)
        w.writerows(inventory)
    return path


def print_class_table(inventory):
    """Patches containing each model class, per split — quick eyeball before split_report_v2."""
    if not HAVE_MODEL or not inventory:
        return
    pres = {sp: np.zeros(NUM_CLASSES, dtype=int) for sp in ("train", "val", "test")}
    for row in inventory:
        pres[row[0]] += (np.array(row[6:], dtype=np.int64) > 0)
    print("\n[Stage 4] Patches containing each model class")
    print(f"  {'mdl':>3} {'class':<26} {'train':>7} {'val':>6} {'test':>6}")
    for mid in range(1, NUM_CLASSES):
        t, v, te = pres["train"][mid], pres["val"][mid], pres["test"][mid]
        flag = "  <-- none in val/test" if (v == 0 or te == 0) else ""
        print(f"  {mid:>3} {CLASS_NAMES[mid]:<26} {t:>7} {v:>6} {te:>6}{flag}")


def save_dataset_info(output_dir, name, settings, stats, raw_totals, filename="dataset_info.json"):
    info = {
        "dataset_name": name,
        "description": f"{name} — generated by prepare_patches_v2.py v2",
        "settings": {k: settings[k] for k in (
            "split_strategy", "train_ratio", "val_ratio", "seed", "resize_to", "patch_size", "overlap",
            "min_target_px", "target_classes", "rare_tiles", "rare_min_px", "max_ignore_frac",
            "bg_keep_frac", "img_format", "min_tiles", "min_class_px", "key_weight", "trials")},
        "class_names": {str(k): v for k, v in RAW_NAMES.items()},
        "model_scheme": {"raw_classes": RAW_CLASSES,
                         "class_names": {str(k): v for k, v in CLASS_NAMES.items()}} if HAVE_MODEL else None,
        "pixel_counts_raw": {str(r): int(c) for r, c in enumerate(raw_totals) if c > 0},
        "patch_stats": stats,
    }
    with open(output_dir / filename, "w") as f:
        json.dump(info, f, indent=2)
    print(f"[Stage 5] Dataset info saved -> {output_dir / filename}")


# ==============================================================================
# CORE PIPELINE
# ==============================================================================

SETTINGS_DEFAULTS = {
    "annotations": None, "int_masks": None,
    "split_strategy": "smart", "train_ratio": 0.8, "val_ratio": 0.1, "seed": 42,
    "resize_to": None, "patch_size": 512, "overlap": 0,
    "min_target_px": 500, "target_classes": None,
    "rare_tiles": 15, "rare_min_px": 100, "max_ignore_frac": 0.7, "bg_keep_frac": 0.0,
    "img_format": "png",
    "min_tiles": 4, "min_tiles_overrides": "", "min_class_px": 500, "key_weight": 2.0, "trials": 100,
    "skip_mask_conversion": False, "masks_only": False, "force": False,
    "strict_labels": False, "check_labels": False,
}


def run_pipeline(name: str, settings: dict, shared_output: Path = None):
    s = {**SETTINGS_DEFAULTS, **settings}
    print("=" * 70)
    print(f"PATCH PIPELINE v2: {name}")
    print("=" * 70)

    if s["check_labels"]:
        if not s["annotations"]:
            print("ERROR: --check-labels needs --annotations")
            return False
        parse_labelmap(Path(s["annotations"]) / "labelmap.txt", strict=s["strict_labels"])
        print("\n[check-labels] nothing written.")
        return True

    src_images = Path(s["src_images"])
    output = Path(shared_output) if shared_output else Path(s["output"])
    int_masks_dir = Path(s["int_masks"]) if s["int_masks"] else \
        (Path(s["output"]).parent / f"{name}_int_masks")
    name_prefix = f"{name}__" if shared_output else ""

    if s["skip_mask_conversion"]:
        if not int_masks_dir.exists() or not any(int_masks_dir.glob("*.png")):
            print(f"ERROR [{name}]: --skip-mask-conversion set but no masks in {int_masks_dir}")
            return False
        print(f"[Stage 1+2] Reusing {len(list(int_masks_dir.glob('*.png')))} converted masks in {int_masks_dir}")
        mapped_raws = set(NAME_TO_RAW.values())
        counts = load_raw_counts(int_masks_dir)
        present = {r for arr in counts.values() for r in np.nonzero(arr)[0].tolist() if r != 255}
        mapped_raws = present
    else:
        if not s["annotations"]:
            print(f"ERROR [{name}]: annotations is required unless skip_mask_conversion is set.")
            return False
        annotations = Path(s["annotations"])
        color_to_raw = parse_labelmap(annotations / "labelmap.txt", strict=s["strict_labels"])
        mapped_raws = set(color_to_raw.values())
        if s["target_classes"]:
            missing = [c for c in s["target_classes"] if c not in mapped_raws]
            if missing and not s["force"]:
                print(f"ERROR [{name}]: no labelmap colour maps to raw {missing} "
                      f"({[RAW_NAMES.get(c) for c in missing]}). Check lulc_class_scheme_v2.NAME_TO_RAW "
                      f"against labelmap.txt, or use --force.")
                return False
            if missing:
                print(f"  WARNING (force): target classes {missing} have no labelmap colour")
        counts = convert_all_masks(color_to_raw, annotations / "SegmentationClass", int_masks_dir)

    raw_totals, raw_tiles = report_class_counts(counts, mapped_raws, s["min_target_px"])

    # target classes: explicit, else every present raw id that has a model channel
    if s["target_classes"]:
        target_classes = list(s["target_classes"])
    else:
        target_classes = sorted(r for r in mapped_raws if r != 0 and (r in RAW_TO_MODEL or not HAVE_MODEL))
        print(f"\n  target classes (auto): {target_classes}")
    rare_raws = [r for r in target_classes if 0 < raw_tiles[r] < s["rare_tiles"]]
    if rare_raws:
        print(f"  rare classes (< {s['rare_tiles']} tiles; kept at >= {s['rare_min_px']} px, "
              f"never dropped for unlabelled share): "
              + ", ".join(f"{RAW_NAMES.get(r)}({int(raw_tiles[r])})" for r in rare_raws))

    if s["masks_only"]:
        print(f"\n[masks_only] Stopping after mask conversion. Masks: {int_masks_dir}")
        return True

    # image/mask pairing report; only paired tiles enter the split
    mask_stems = sorted(p.stem for p in int_masks_dir.glob("*.png"))
    paired = [m for m in mask_stems if find_image(src_images, m) is not None]
    img_stems = {p.stem for p in src_images.iterdir() if p.suffix in IMG_EXTS} if src_images.exists() else set()
    print(f"\n[Pairing] {len(paired)} tiles have image+mask | masks without image: "
          f"{len(mask_stems) - len(paired)} | images without mask: {len(img_stems - set(mask_stems))}")
    if not paired:
        print(f"ERROR [{name}]: no image/mask pairs. Check --src-images ({src_images}).")
        return False

    output.mkdir(parents=True, exist_ok=True)
    split_map = make_split(s["split_strategy"], int_masks_dir, paired, s, output)

    stats, inventory = create_patches(
        split_map, src_images, int_masks_dir, output, s["resize_to"], s["patch_size"], s["overlap"],
        target_classes, s["min_target_px"], rare_raws, s["rare_min_px"], s["max_ignore_frac"],
        s["bg_keep_frac"], s["img_format"], s["seed"], name_prefix, name)

    s["target_classes"] = target_classes
    suffix = f"_{name}" if shared_output else ""
    inv_path = save_inventory(output, inventory, f"per_image_class_inventory{suffix}.csv")
    print_class_table(inventory)
    save_dataset_info(output, name, s, stats, raw_totals,
                      f"dataset_info_{name}.json" if shared_output else "dataset_info.json")

    total_kept = sum(v["kept"] for v in stats.values())
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE [{name}] — {total_kept:,} patches ready in {output}")
    if inv_path:
        print(f"Inventory (reusable by smart_split_v2.py --inventory-csv): {inv_path}")
    print("Next: python split_report_v2.py --train OUT/train --val OUT/val --test OUT/test")
    print("=" * 70)
    return True


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Generic patch-preparation pipeline (v2).",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--config", type=Path, default=None, help="datasets.json; use with --dataset or --all")
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--shared-output", type=Path, default=None,
                   help="Write ALL datasets into ONE folder (names prefixed with the dataset name).")
    p.add_argument("--name")
    p.add_argument("--src-images", type=Path)
    p.add_argument("--annotations", type=Path, help="Folder with labelmap.txt and SegmentationClass/")
    p.add_argument("--int-masks", type=Path, default=None)
    p.add_argument("--output", type=Path)
    p.add_argument("--split-strategy", choices=["smart", "tile", "site"], default="smart")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resize-to", type=int, default=None)
    p.add_argument("--patch-size", type=int, default=512)
    p.add_argument("--overlap", type=int, default=0)
    p.add_argument("--min-target-px", type=int, default=500)
    p.add_argument("--target-classes", type=str, default=None,
                   help="Comma-separated raw ids. Default: every present raw id that has a model channel.")
    p.add_argument("--rare-tiles", type=int, default=15,
                   help="A class in fewer source tiles than this is 'rare' (default 15).")
    p.add_argument("--rare-min-px", type=int, default=100,
                   help="A patch with >= this many rare-class px is always kept (default 100).")
    p.add_argument("--max-ignore-frac", type=float, default=0.7,
                   help="Drop non-rare patches with more than this share of unlabelled px (default 0.7).")
    p.add_argument("--bg-keep-frac", type=float, default=0.0,
                   help="Keep this random fraction of target-free patches as hard negatives (default 0).")
    p.add_argument("--img-format", choices=["png", "jpg"], default="png",
                   help="Patch image format; masks are always PNG (default png).")
    p.add_argument("--min-tiles", type=int, default=4, help="smart: wanted tiles/class in val and test")
    p.add_argument("--min-tiles-overrides", type=str, default="", help="smart: 'Class name:n,...'")
    p.add_argument("--min-class-px", type=int, default=500, help="smart: px for a tile to 'have' a class")
    p.add_argument("--key-weight", type=float, default=2.0, help="smart: extra weight for key classes")
    p.add_argument("--trials", type=int, default=100, help="smart: random restarts")
    p.add_argument("--skip-mask-conversion", action="store_true")
    p.add_argument("--masks-only", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--strict-labels", action="store_true", help="Abort if a labelmap name has no mapping.")
    p.add_argument("--check-labels", action="store_true", help="Only print the labelmap mapping table.")
    return p.parse_args()


def main():
    a = parse_args()
    if a.config:
        with open(a.config) as f:
            registry = json.load(f)
        if a.all:
            names = list(registry.keys())
        elif a.dataset:
            if a.dataset not in registry:
                print(f"ERROR: '{a.dataset}' not in {a.config}. Available: {list(registry.keys())}")
                sys.exit(1)
            names = [a.dataset]
        else:
            print("ERROR: --config requires --dataset <name> or --all.")
            sys.exit(1)
        results = {n: run_pipeline(n, registry[n], shared_output=a.shared_output) for n in names}
        if len(names) > 1:
            print("\nBATCH SUMMARY")
            for n, ok in results.items():
                print(f"  {'OK  ' if ok else 'FAIL'} — {n}")
        if not all(results.values()):
            sys.exit(1)
        return

    if not a.name or (not a.check_labels and not (a.src_images and a.output)):
        print("ERROR: --name, --src-images and --output are required (or --config, or --check-labels).")
        sys.exit(1)
    settings = {
        "src_images": a.src_images, "output": a.output, "annotations": a.annotations,
        "int_masks": a.int_masks, "split_strategy": a.split_strategy,
        "train_ratio": a.train_ratio, "val_ratio": a.val_ratio, "seed": a.seed,
        "resize_to": a.resize_to, "patch_size": a.patch_size, "overlap": a.overlap,
        "min_target_px": a.min_target_px,
        "target_classes": [int(c) for c in a.target_classes.split(",")] if a.target_classes else None,
        "rare_tiles": a.rare_tiles, "rare_min_px": a.rare_min_px,
        "max_ignore_frac": a.max_ignore_frac, "bg_keep_frac": a.bg_keep_frac,
        "img_format": a.img_format, "min_tiles": a.min_tiles,
        "min_tiles_overrides": a.min_tiles_overrides, "min_class_px": a.min_class_px,
        "key_weight": a.key_weight, "trials": a.trials,
        "skip_mask_conversion": a.skip_mask_conversion, "masks_only": a.masks_only,
        "force": a.force, "strict_labels": a.strict_labels, "check_labels": a.check_labels,
    }
    if not run_pipeline(a.name, settings, shared_output=a.shared_output):
        sys.exit(1)


if __name__ == "__main__":
    main()
