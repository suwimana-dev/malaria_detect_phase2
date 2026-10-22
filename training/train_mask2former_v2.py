"""
train_mask2former_v2.py — LULC segmentation with Mask2Former (Swin-T backbone)
======================================================================
Stand-alone training script for ONE architecture (the other two live in
train_deeplab_v2.py / train_segformer_v2.py / train_mask2former_v2.py).
Same splits, class mapping (model_class_scheme_v2.py), ignore handling (255),
class stats dump, metrics (mIoU + mIoU_key), spot-VM checkpoint/resume and
bucket sync as the others, so the three can be compared fairly.

Usage
-----
    python train_mask2former_v2.py --data-root D:/lulc_v2_patches --out-dir runs/stats --stats-only
    python train_mask2former_v2.py --data-root ... --out-dir runs/mask2former_t --resume --gcs-dir gs://BUCKET/runs/x
    python train_mask2former_v2.py --data-root ... --out-dir runs/x --eval-only --eval-test

Encoders: swin-tiny.  Defaults: see ARCH_DEFAULTS below.
Effective batch = --batch-size x --accum-steps (16 by default).
Dependencies: torch, numpy, pillow, transformers, wandb (optional).
"""

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# T4/RTX perf flags (free, do not affect accuracy)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))           # this file's own folder
sys.path.insert(0, str(_here.parent))    # its parent, in case the scheme files live there
try:
    from model_class_scheme_v2 import (RAW_TO_MODEL, RAW_CLASSES, CLASS_NAMES,
                                       NUM_CLASSES, KEY_CLASS_IDS)
except ModuleNotFoundError as _e:
    print("ERROR: cannot import model_class_scheme_v2 / lulc_class_scheme_v2. Put them in the same "
          f"folder as this file (or its parent). Underlying error: {_e}")
    sys.exit(1)

IGNORE = 255
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

# Per-architecture defaults (effective batch = batch_size x accum_steps = 16 everywhere,
# as in the MobileNet run; small batches + accumulation keep a 16 GB T4 happy).
ARCH = 'mask2former'
ARCH_DEFAULTS = {
    "mask2former":   dict(encoder="swin-tiny",       lr=1e-4, wd=5e-2, bs=2, accum=8, warmup=2),
}
HF_IDS = {
    "swin-tiny": "facebook/mask2former-swin-tiny-ade-semantic",
}

STOP = {"flag": False}


def _on_sigterm(signum, frame):            # spot VMs send SIGTERM ~30 s before preemption
    STOP["flag"] = True
    print("\n[signal] SIGTERM received — will checkpoint and exit at the next step.", flush=True)


# ==============================================================================
# label lookup + one-time dataset scan
# ==============================================================================
def build_lut(background):
    lut = np.full(256, IGNORE, np.uint8)
    for raw, mid in RAW_TO_MODEL.items():
        if 0 <= raw < 256:
            lut[raw] = mid
    if background == "ignore":
        lut[0] = IGNORE
    lut[255] = IGNORE
    return lut


def find_images(img_dir: Path, mask_files):
    """mask name -> image path (same stem, any image extension)."""
    idx = {}
    by_stem = {}
    for p in img_dir.iterdir():
        if p.suffix in IMG_EXTS:
            by_stem[p.stem] = p
    missing = []
    for m in mask_files:
        p = by_stem.get(Path(m).stem)
        if p is None:
            missing.append(m)
        else:
            idx[m] = p
    if missing:
        raise SystemExit(f"{len(missing)} mask(s) have no image in {img_dir}, e.g. {missing[:3]}")
    return idx


def scan_split(mask_dir: Path, lut, cache: Path, tag: str):
    """pixels per class, patches per class, and the class set of every patch (cached)."""
    files = sorted(p.name for p in mask_dir.glob("*.png"))
    sig = hashlib.md5(json.dumps([RAW_CLASSES, int(lut[0]), len(files),
                                  sum((mask_dir / f).stat().st_size for f in files)]).encode()).hexdigest()
    if cache.exists():
        d = json.loads(cache.read_text())
        if d.get("sig") == sig:
            return (np.array(d["pixels"], np.int64), [set(s) for s in d["present"]], d["files"])
    pixels = np.zeros(NUM_CLASSES, np.int64)
    present = []
    print(f"[scan] {tag}: {len(files)} masks...", flush=True)
    for name in files:
        m = lut[np.array(Image.open(mask_dir / name))]
        cnt = np.bincount(m.ravel(), minlength=256)[:256]
        pixels += cnt[:NUM_CLASSES]
        present.append({int(c) for c in np.nonzero(cnt[:NUM_CLASSES])[0]})
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"sig": sig, "pixels": pixels.tolist(),
                                 "present": [sorted(s) for s in present], "files": files}))
    return pixels, present, files


def dump_class_stats(stats, out_dir: Path, background):
    """Print + write class_stats.csv: pixels and patches per class and split."""
    rows = []
    print("\n" + "=" * 100)
    print("CLASS STATS (model classes; pixels after the ignore LUT)")
    print("=" * 100)
    print(f"  {'mdl':>3} {'class':<26} {'train px':>13} {'train patches':>13} {'val px':>12} "
          f"{'val patches':>11} {'test patches':>12}")
    problems = []
    for m in range(NUM_CLASSES):
        row = {"model_id": m, "class": CLASS_NAMES[m], "key": int(m in KEY_CLASS_IDS)}
        cells = []
        for sp in ("train", "val", "test"):
            if sp not in stats:
                row[f"{sp}_pixels"], row[f"{sp}_patches"] = "", ""
                continue
            px, present, _ = stats[sp]
            row[f"{sp}_pixels"] = int(px[m])
            row[f"{sp}_patches"] = sum(1 for s in present if m in s)
        rows.append(row)
        if m == 0 and background == "ignore":
            continue
        tp = row["train_patches"]
        flag = ""
        if tp == 0:
            flag = "  <-- NO TRAIN DATA"
            problems.append(f"{CLASS_NAMES[m]}: no training pixels")
        elif "val" in stats and row["val_patches"] == 0:
            flag = "  <-- not measurable (0 val patches)"
        print(f"  {m:>3} {CLASS_NAMES[m]:<26} {row['train_pixels']:>13,} {row['train_patches']:>13,} "
              f"{row.get('val_pixels', ''):>12,} {row.get('val_patches', ''):>11,} "
              f"{row.get('test_patches', ''):>12}{flag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "class_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  wrote {out_dir / 'class_stats.csv'}")
    if problems:
        print("  WARNING — these classes cannot be learned: " + "; ".join(problems)
              + "\n  List them in model_class_scheme_v2.EXCLUDE_RAW or annotate more examples.")


def class_weights(pixels, cap, background):
    """Inverse pixel frequency, normalised to mean 1, capped (same recipe as before)."""
    fg = pixels.astype(np.float64)
    if background == "ignore":
        fg[0] = 0
    nz = fg > 0
    w = np.zeros(NUM_CLASSES)
    w[nz] = 1.0 / fg[nz]
    w[nz] /= w[nz].mean()
    w = np.clip(w, 0, cap)
    if background == "ignore":
        w[0] = 0.0
    return torch.tensor(w, dtype=torch.float32)


def sampler_weights(present, cap):
    """Per-patch weight = boost of the rarest class it holds (sqrt inverse frequency, capped)."""
    n = len(present)
    cnt = np.zeros(NUM_CLASSES)
    for s in present:
        for m in s:
            cnt[m] += 1
    boost = np.ones(NUM_CLASSES)
    for m in range(1, NUM_CLASSES):
        if cnt[m] > 0:
            boost[m] = min((n / cnt[m]) ** 0.5, cap)
    w = [max((boost[m] for m in s if m != 0), default=1.0) for s in present]
    return torch.tensor(w, dtype=torch.double)


def detect_rare(present, rare_frac, names):
    """Rare = present in < rare_frac of train patches (or named by --rare-classes)."""
    n = max(len(present), 1)
    if names:
        by_name = {v.strip().lower(): k for k, v in CLASS_NAMES.items()}
        out = set()
        for nm in [x.strip().lower() for x in names.split(",") if x.strip()]:
            if nm not in by_name:
                raise SystemExit(f"--rare-classes: unknown class '{nm}'")
            out.add(by_name[nm])
        return out
    cnt = np.zeros(NUM_CLASSES)
    for s in present:
        for m in s:
            cnt[m] += 1
    return {m for m in range(1, NUM_CLASSES) if 0 < cnt[m] / n < rare_frac}


# ==============================================================================
# dataset
# ==============================================================================
class SegDataset(Dataset):
    def __init__(self, img_paths, mask_dir, files, train, mean, std, lut, aug_p=0.5,
                 present=None, rare=None, rare_aug_p=0.85, rare_jitter=0.35):
        self.img_paths, self.mask_dir, self.files = img_paths, Path(mask_dir), files
        self.train, self.aug_p, self.mean, self.std, self.lut = train, aug_p, mean, std, lut
        self.present, self.rare = present, rare or set()
        self.rare_aug_p, self.rare_jitter = rare_aug_p, rare_jitter

    def __len__(self):
        return len(self.files)

    def _geometric(self, img, mask):
        if random.random() < 0.5:
            img, mask = img[:, ::-1], mask[:, ::-1]
        if random.random() < 0.5:
            img, mask = img[::-1], mask[::-1]
        k = random.randint(0, 3)
        if k:
            img, mask = np.rot90(img, k), np.rot90(mask, k)
        return np.ascontiguousarray(img), np.ascontiguousarray(mask)

    def _photometric(self, img, jr):
        img = img * (1.0 + random.uniform(-jr, jr))                       # brightness
        m = img.mean()
        img = (img - m) * (1.0 + random.uniform(-jr, jr)) + m             # contrast
        return np.clip(img, 0.0, 1.0)

    def __getitem__(self, i):
        name = self.files[i]
        img = np.array(Image.open(self.img_paths[name]).convert("RGB"), np.float32) / 255.0
        mask = self.lut[np.array(Image.open(self.mask_dir / name))]
        if self.train:
            rare = bool(self.present and (self.present[i] & self.rare))
            p = self.rare_aug_p if rare else self.aug_p
            jr = self.rare_jitter if rare else 0.2
            if random.random() < p:
                img, mask = self._geometric(img, mask)
                if random.random() < 0.5:
                    img = self._photometric(img, jr)
        img = (img - self.mean) / self.std
        return (torch.from_numpy(img.transpose(2, 0, 1)).float(),
                torch.from_numpy(mask.astype(np.int64)))


# ==============================================================================
# model wrappers: one interface for the three architectures
#   train_loss(img, mask, loss_fn) -> scalar tensor
#   eval_step(img, mask, loss_fn)  -> (loss, logits[B, C, H, W])
# ==============================================================================
# ==============================================================================
# model wrappers: one interface for the three architectures
#   train_loss(img, mask, loss_fn) -> scalar tensor
#   eval_step(img, mask, loss_fn)  -> (loss, logits[B, C, H, W])
# ==============================================================================
class Mask2FormerModel(nn.Module):
    """Semantic Mask2Former. Loss is the model's own (Hungarian matching on per-class masks);
    class weights do not apply to it, balance comes from the sampler. Pixels labelled 255 belong
    to no mask, so they are treated as 'not any class' by the mask loss."""

    def __init__(self, encoder, pretrained, hf_id=None):
        super().__init__()
        from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig
        hf_id = hf_id or HF_IDS.get(encoder, encoder)
        if pretrained:
            self.net = Mask2FormerForUniversalSegmentation.from_pretrained(
                hf_id, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)
        else:
            self.net = Mask2FormerForUniversalSegmentation(Mask2FormerConfig(num_labels=NUM_CLASSES))
        self.mean, self.std = IMAGENET_MEAN, IMAGENET_STD

    @staticmethod
    def _targets(mask):
        mask_labels, class_labels = [], []
        for b in range(mask.shape[0]):
            cls = [int(c) for c in torch.unique(mask[b]).tolist() if c != IGNORE]
            if cls:
                mask_labels.append(torch.stack([(mask[b] == c) for c in cls]).float())
                class_labels.append(torch.tensor(cls, device=mask.device, dtype=torch.long))
            else:
                mask_labels.append(torch.zeros((0,) + tuple(mask.shape[-2:]), device=mask.device))
                class_labels.append(torch.zeros(0, device=mask.device, dtype=torch.long))
        return mask_labels, class_labels

    @staticmethod
    def _semantic(out, size):
        cls = out.class_queries_logits.softmax(-1)[..., :-1]              # [B, Q, C]
        msk = out.masks_queries_logits.sigmoid()                          # [B, Q, h, w]
        seg = torch.einsum("bqc,bqhw->bchw", cls.float(), msk.float())
        return F.interpolate(seg, size=size, mode="bilinear", align_corners=False)

    def train_loss(self, img, mask, loss_fn):
        ml, cl = self._targets(mask)
        return self.net(pixel_values=img, mask_labels=ml, class_labels=cl).loss

    def eval_step(self, img, mask, loss_fn):
        ml, cl = self._targets(mask)
        out = self.net(pixel_values=img, mask_labels=ml, class_labels=cl)
        return out.loss, self._semantic(out, img.shape[-2:])

    def param_groups(self, lr, wd):
        bb = [p for n, p in self.named_parameters() if "pixel_level_module.encoder" in n and p.requires_grad]
        rest = [p for n, p in self.named_parameters() if "pixel_level_module.encoder" not in n and p.requires_grad]
        return [{"params": rest, "lr": lr, "weight_decay": wd},
                {"params": bb, "lr": lr * 0.1, "weight_decay": wd}]       # pretrained backbone moves slowly


def build_model(a):
    return Mask2FormerModel(a.encoder, not a.no_pretrained, a.hf_id)


# ==============================================================================
# metrics
# ==============================================================================
def add_confusion(cm, logits, target, suppress_bg):
    if suppress_bg:
        logits = logits.clone()
        logits[:, 0] = -float("inf")
    v = target != IGNORE
    t, p = target[v], logits.argmax(1)[v]
    cm += torch.bincount(t * NUM_CLASSES + p, minlength=NUM_CLASSES ** 2).reshape(cm.shape)


def per_class_iou(cm):
    cm = cm.double()
    tp = cm.diag()
    denom = cm.sum(0) + cm.sum(1) - tp
    return torch.where(denom > 0, tp / denom, torch.full_like(tp, float("nan")))


def summarize(iou, background):
    lo = 1 if background == "ignore" else 0
    miou = torch.nanmean(iou[lo:]).item()
    key = [k for k in KEY_CLASS_IDS if not torch.isnan(iou[k])]
    miou_key = torch.stack([iou[k] for k in key]).mean().item() if key else float("nan")
    return miou, miou_key


@torch.no_grad()
def validate(model, loader, loss_fn, device, use_amp, background, max_batches=None):
    model.eval()
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    tot, nb = 0.0, 0
    for bi, (img, mask) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        img, mask = img.to(device), mask.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            loss, logits = model.eval_step(img, mask, loss_fn)
        tot += float(loss.item())
        nb += 1
        add_confusion(cm, logits.float(), mask, background == "ignore")
    return tot / max(nb, 1), per_class_iou(cm.cpu()), cm.cpu()


# ==============================================================================
# checkpoints + bucket sync (spot-VM friendly)
# ==============================================================================
def atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _gs_cmd():
    if shutil.which("gsutil"):
        return ["gsutil", "-q", "cp"]
    if shutil.which("gcloud"):
        return ["gcloud", "storage", "cp", "--quiet"]
    return None


def gcs_push(paths, dest):
    cmd = _gs_cmd()
    if not dest or cmd is None:
        return
    for p in paths:
        if Path(p).exists():
            r = subprocess.run(cmd + [str(p), dest.rstrip("/") + "/"], capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  [gcs] push failed for {Path(p).name}: {r.stderr.strip()[:200]}")


def gcs_pull(name, dest, local_dir):
    cmd = _gs_cmd()
    if not dest or cmd is None:
        return False
    r = subprocess.run(cmd + [dest.rstrip("/") + "/" + name, str(local_dir) + "/"],
                       capture_output=True, text=True)
    return r.returncode == 0


def scheme_meta(a, mean, std):
    return {"arch": a.arch, "encoder": a.encoder, "raw_classes": list(RAW_CLASSES),
            "class_names": {str(k): v for k, v in CLASS_NAMES.items()}, "num_classes": NUM_CLASSES,
            "background": a.background, "mean": [float(x) for x in mean], "std": [float(x) for x in std]}


# ==============================================================================
# main
# ==============================================================================
def main():
    ap = argparse.ArgumentParser(description="LULC segmentation: DeepLabv3+ / SegFormer / Mask2Former.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--data-root", type=Path, required=True, help="folder with train/ val/ [test/], each images/ + masks/")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--encoder", default=None, help="deeplabv3plus: efficientnet-b3|efficientnet-b4|mobilenet_v2 | "
                    "segformer: mit-b1|mit-b2 | mask2former: swin-tiny")
    ap.add_argument("--hf-id", default=None, help="override the Hugging Face model id (segformer / mask2former)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--accum-steps", type=int, default=None, help="gradient accumulation (effective batch = bs x accum)")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--warmup-epochs", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--weight-cap", type=float, default=3.0, help="class-weight cap (CE archs)")
    ap.add_argument("--sampler-cap", type=float, default=3.0, help="oversampling cap; raise to 6-12 for very thin classes")
    ap.add_argument("--patience", type=int, default=15, help="early-stopping patience in epochs")
    ap.add_argument("--background", choices=["ignore", "train"], default="ignore",
                    help="ignore: Background is not trained (as in train_mobilvnet.py)")
    ap.add_argument("--rare-frac", type=float, default=0.05, help="class in < this share of train patches = rare")
    ap.add_argument("--rare-classes", default=None, help="comma-separated names; overrides auto-detection")
    ap.add_argument("--select", choices=["mIoU", "mIoU_key"], default="mIoU", help="metric that picks best.pth")
    ap.add_argument("--no-amp", action="store_true", help="disable fp16 autocast (use if the loss turns NaN)")
    ap.add_argument("--no-pretrained", action="store_true", help="random init (offline / smoke tests)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stats-only", action="store_true", help="dump class stats and exit")
    ap.add_argument("--eval-only", action="store_true", help="skip training; evaluate best.pth")
    ap.add_argument("--eval-test", action="store_true", help="score best.pth on the test split at the end")
    ap.add_argument("--gcs-dir", default=None, help="gs://bucket/path — checkpoints are pushed here")
    ap.add_argument("--sync-every", type=int, default=1, help="push to the bucket every N epochs")
    ap.add_argument("--wandb-project", default="malaria-ai-lulc-v2")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps-per-epoch", type=int, default=None, help="debug: cap optimizer batches per epoch")
    ap.add_argument("--max-val-batches", type=int, default=None, help="debug: cap validation batches")
    a = ap.parse_args()
    a.arch = ARCH

    d = ARCH_DEFAULTS[ARCH]
    a.encoder = a.encoder or d["encoder"]
    a.batch_size = a.batch_size or d["bs"]
    a.accum_steps = a.accum_steps or d["accum"]
    a.lr = a.lr or d["lr"]
    a.weight_decay = a.weight_decay if a.weight_decay is not None else d["wd"]
    a.warmup_epochs = a.warmup_epochs if a.warmup_epochs is not None else d["warmup"]

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    signal.signal(signal.SIGTERM, _on_sigterm)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and not a.no_amp
    lut = build_lut(a.background)

    # ---- data --------------------------------------------------------------
    splits = {sp: (a.data_root / sp / "images", a.data_root / sp / "masks")
              for sp in ("train", "val", "test")}
    for sp in ("train", "val"):
        if not splits[sp][1].is_dir():
            raise SystemExit(f"missing {splits[sp][1]}")
    stats = {sp: scan_split(m, lut, a.out_dir / f"{sp}_stats.json", sp)
             for sp, (i, m) in splits.items() if m.is_dir() and any(m.glob("*.png"))}
    print(f"\nScheme: {NUM_CLASSES} output channels, key classes: "
          + ", ".join(CLASS_NAMES[k] for k in KEY_CLASS_IDS))
    dump_class_stats(stats, a.out_dir, a.background)
    if a.stats_only:
        return

    pixels, present, tr_files = stats["train"]
    va_pixels, va_present, va_files = stats["val"]
    cw = class_weights(pixels, a.weight_cap, a.background)
    sw = sampler_weights(present, a.sampler_cap)
    rare = detect_rare(present, a.rare_frac, a.rare_classes)
    print("\nRare classes (stronger augmentation): " + (", ".join(CLASS_NAMES[m] for m in sorted(rare)) or "none"))
    if a.arch != "mask2former":
        print("Class weights (capped at %.1f): " % a.weight_cap
              + ", ".join(f"{CLASS_NAMES[m]}={cw[m]:.2f}" for m in range(1, NUM_CLASSES) if cw[m] > 0))

    # ---- model -------------------------------------------------------------
    model = build_model(a).to(device)
    mean, std = model.mean, model.std
    meta = scheme_meta(a, mean, std)

    tr_imgs = find_images(splits["train"][0], tr_files)
    va_imgs = find_images(splits["val"][0], va_files)
    train_ds = SegDataset(tr_imgs, splits["train"][1], tr_files, True, mean, std, lut,
                          present=present, rare=rare)
    val_ds = SegDataset(va_imgs, splits["val"][1], va_files, False, mean, std, lut)
    sampler = WeightedRandomSampler(sw, len(train_ds), replacement=True)
    train_ld = DataLoader(train_ds, a.batch_size, sampler=sampler, num_workers=a.num_workers,
                          pin_memory=device.type == "cuda", drop_last=True)
    val_ld = DataLoader(val_ds, a.batch_size, shuffle=False, num_workers=a.num_workers,
                        pin_memory=device.type == "cuda")
    if len(train_ld) == 0:
        raise SystemExit(f"fewer train patches ({len(train_ds)}) than --batch-size {a.batch_size}")

    loss_fn = nn.CrossEntropyLoss(weight=cw.to(device), ignore_index=IGNORE)
    opt = torch.optim.AdamW(model.param_groups(a.lr, a.weight_decay), lr=a.lr)
    if a.warmup_epochs > 0:
        warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=a.warmup_epochs)
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, a.epochs - a.warmup_epochs))
        sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[a.warmup_epochs])
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- resume (also from the bucket after a preemption) -------------------
    ckpt, best_path = a.out_dir / "last.pth", a.out_dir / "best.pth"
    start, best, bad = 0, -1.0, 0
    if (a.resume or a.eval_only) and not ckpt.exists() and a.gcs_dir:
        for name in ("last.pth", "best.pth", "metrics.csv", "wandb_id.txt"):
            gcs_pull(name, a.gcs_dir, a.out_dir)
    if a.resume and ckpt.exists() and not a.eval_only:
        c = torch.load(ckpt, map_location=device, weights_only=False)
        cm_ = c.get("meta", {})
        if cm_.get("raw_classes") != meta["raw_classes"] or cm_.get("arch") != a.arch \
                or cm_.get("encoder") != a.encoder or cm_.get("background") != a.background:
            raise SystemExit("last.pth was trained with a different arch/encoder/class scheme/background "
                             f"({cm_.get('arch')}/{cm_.get('encoder')}/{cm_.get('raw_classes')}). "
                             "Use a new --out-dir, do not resume across scheme changes.")
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"]); scaler.load_state_dict(c["scaler"])
        start, best, bad = c["epoch"] + 1, c.get("best", -1.0), c.get("bad", 0)
        print(f"[resume] epoch {start} (best {best:.4f}, {bad} epoch(s) since last improvement)")

    # ---- W&B ---------------------------------------------------------------
    use_wandb = not a.no_wandb and not a.eval_only
    if use_wandb:
        import wandb
        idf = a.out_dir / "wandb_id.txt"
        rid = idf.read_text().strip() if (a.resume and idf.exists()) else wandb.util.generate_id()
        idf.write_text(rid)
        wandb.init(project=a.wandb_project, entity=a.wandb_entity, id=rid, resume="allow",
                   name=a.run_name or f"{a.arch}-{a.encoder}", group=a.arch,
                   config=json.loads(json.dumps(vars(a), default=str)))

    # ---- training ----------------------------------------------------------
    def save_last(epoch, best_, bad_):
        atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                     "scaler": scaler.state_dict(), "epoch": epoch, "best": best_, "bad": bad_,
                     "meta": meta}, ckpt)

    metrics_csv = a.out_dir / "metrics.csv"
    nonfinite = 0
    if not a.eval_only:
        for epoch in range(start, a.epochs):
            t0 = time.time()
            model.train()
            opt.zero_grad(set_to_none=True)
            run, nb = 0.0, 0
            n_steps = len(train_ld) if not a.max_steps_per_epoch else min(len(train_ld), a.max_steps_per_epoch)
            for step, (img, mask) in enumerate(train_ld):
                if step >= n_steps:
                    break
                img, mask = img.to(device), mask.to(device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    loss = model.train_loss(img, mask, loss_fn) / a.accum_steps
                if not torch.isfinite(loss):
                    nonfinite += 1
                    opt.zero_grad(set_to_none=True)
                    print(f"  [warn] non-finite loss at epoch {epoch + 1} step {step} ({nonfinite} so far)")
                    if nonfinite >= 20:
                        raise SystemExit("20 non-finite losses — rerun with --no-amp or a lower --lr")
                    continue
                scaler.scale(loss).backward()
                if (step + 1) % a.accum_steps == 0 or step + 1 == n_steps:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt); scaler.update()
                    opt.zero_grad(set_to_none=True)
                run += loss.item() * a.accum_steps; nb += 1
                if STOP["flag"]:
                    save_last(epoch - 1, best, bad)         # epoch not finished: resume repeats it
                    gcs_push([ckpt, best_path, metrics_csv], a.gcs_dir)
                    print("[signal] checkpoint saved and pushed; exiting.")
                    sys.exit(143)
            sched.step()
            tr_loss = run / max(nb, 1)

            va_loss, iou, _ = validate(model, val_ld, loss_fn, device, use_amp, a.background, a.max_val_batches)
            miou, miou_key = summarize(iou, a.background)
            score = miou if a.select == "mIoU" else miou_key
            print(f"\nEpoch {epoch + 1}/{a.epochs}  train={tr_loss:.4f}  val={va_loss:.4f}  "
                  f"mIoU={miou:.4f}  mIoU_key={miou_key:.4f}  ({time.time() - t0:.0f}s)")
            log = {"epoch": epoch + 1, "train_loss": tr_loss, "val_loss": va_loss, "mIoU": miou,
                   "mIoU_key": miou_key, "lr": opt.param_groups[0]["lr"]}
            lo = 1 if a.background == "ignore" else 0
            for m in range(lo, NUM_CLASSES):
                log[f"IoU/{CLASS_NAMES[m]}"] = iou[m].item()
                print(f"    {('*' if m in KEY_CLASS_IDS else ' ')} {CLASS_NAMES[m]:<24} {iou[m].item():.4f}")
            new_file = not metrics_csv.exists()
            with open(metrics_csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(log.keys()))
                if new_file:
                    w.writeheader()
                w.writerow(log)
            if use_wandb:
                import wandb; wandb.log(log)

            if not np.isnan(score) and score > best:
                best, bad = score, 0
                atomic_save({"model": model.state_dict(), "score": best, "epoch": epoch,
                             "select": a.select, "meta": meta}, best_path)
                print(f"    * new best {a.select} {best:.4f} -> best.pth")
            else:
                bad += 1
            save_last(epoch, best, bad)              # after best/bad are updated for this epoch
            if a.gcs_dir and (epoch + 1) % a.sync_every == 0:
                gcs_push([ckpt, best_path, metrics_csv, a.out_dir / "class_stats.csv",
                          a.out_dir / "wandb_id.txt"], a.gcs_dir)
            if bad >= a.patience:
                print(f"\nEarly stop: no val improvement for {a.patience} epochs.")
                break
        print(f"\nDone. Best {a.select} = {best:.4f}  ({a.out_dir})")

    # ---- final reports -----------------------------------------------------
    def report(split, ld, name):
        c = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(c["model"])
        loss, iou, cm = validate(model, ld, loss_fn, device, use_amp, a.background, a.max_val_batches)
        miou, miou_key = summarize(iou, a.background)
        rep = {"split": split, "arch": a.arch, "encoder": a.encoder, "best_epoch": c.get("epoch", -1) + 1,
               "loss": loss, "mIoU": miou, "mIoU_key": miou_key,
               "per_class_iou": {CLASS_NAMES[m]: (None if torch.isnan(iou[m]) else iou[m].item())
                                 for m in range(NUM_CLASSES)},
               "confusion_matrix": cm.tolist(), "class_names": [CLASS_NAMES[m] for m in range(NUM_CLASSES)]}
        (a.out_dir / name).write_text(json.dumps(rep, indent=2))
        print(f"\n[{split}] best.pth  mIoU={miou:.4f}  mIoU_key={miou_key:.4f}  -> {a.out_dir / name}")

    if best_path.exists():
        report("val", val_ld, "val_report.json")
        if a.eval_test:
            te_m = splits["test"][1]
            if not te_m.is_dir():
                print("[test] no test split found — skipped.")
            else:
                te_pixels, te_present, te_files = stats["test"]
                te_ds = SegDataset(find_images(splits["test"][0], te_files), te_m, te_files, False, mean, std, lut)
                te_ld = DataLoader(te_ds, a.batch_size, shuffle=False, num_workers=a.num_workers)
                report("test", te_ld, "test_report.json")
        if a.gcs_dir:
            gcs_push([a.out_dir / "val_report.json", a.out_dir / "test_report.json"], a.gcs_dir)


if __name__ == "__main__":
    main()
