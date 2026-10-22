"""
model_class_scheme_v2.py   (v2 — pending-retrain scheme for the 300-image LULC set)
==================================================================================
Single source of truth for the CURRENT model's contiguous output class space.
Training, split_report_v2.py, smart_split_v2.py, prepare_patches_v2.py and every
inference-side script import CLASS_NAMES / RAW_TO_MODEL from here.

Position in RAW_CLASSES IS the model's output channel index.

HOW THIS ONE IS ORDERED
-----------------------
Active classes are grouped by project role (water habitats first, then wet
vegetated habitats, then context). Classes that are in the annotation scheme
but have NO training examples yet sit in PENDING_RAW and are APPENDED at the
end *only when* they are removed from EXCLUDE_RAW — so switching one on never
renumbers any other class (only the head width changes, so retrain, don't
resume).

  EXCLUDE_RAW = [17]       Water Storage Containers: 0 examples as of the 16 Oct
                           2026 status update, so it is IGNORE (255) in training.
                           It is a key habitat — once it has examples, set
                           EXCLUDE_RAW = [] and retrain.
                           Sugarcane (raw 10) is KEPT although under-represented:
                           it is channel 22 (appended after the grouped classes).

Run `python model_class_scheme_v2.py` to print the live table.

NOT COMPATIBLE with the deployed 16/17-channel model (unified_v9b): different
count and order. Keep this file in the v2 pipeline folder; only copy it over
the deployed copy when the new model is trained and the backend is updated.
The old scheme stays available as LEGACY_SCHEMES["v9b"].
"""

from lulc_class_scheme_v2 import RAW_NAMES, CLASS_ROLE, KEY_ROLES, SURVEY_HABITAT

# ---- edit here ---------------------------------------------------------------
ACTIVE_RAW = [
    0,                               # Background
    # water habitats ------------------------------------------------ KEY
    2, 3, 7, 8, 9, 4, 18, 15,        # Fish Ponds, Irrigation Channels, Irrigation Ditches,
                                     # Intercrop Water Drains, Rainwater Puddles, Flooded Areas,
                                     # River Proximities, Brick Making Pits
    # wet vegetated habitats ---------------------------------------- KEY
    25, 1, 26,                       # Rice Fields (early), Rice Fields (late), Marshes
    # vegetation context
    5, 22, 21, 24,                   # Other_crop, Arable Land, Natural Vegetation, Shrubs
    # built / bare context
    11, 23, 14, 13, 16, 20,          # Human settlement, Impervious Surface, Road, Foot_Path,
                                     # Brick Site, Bare Land
]
PENDING_RAW = [10, 17]               # Sugarcane, Water Storage Containers (appended in this order when enabled)
EXCLUDE_RAW = [17]                   # no examples yet -> ignore; remove from the list once annotated
# ------------------------------------------------------------------------------

RAW_CLASSES = ACTIVE_RAW + [r for r in PENDING_RAW if r not in EXCLUDE_RAW]

# Legacy raw ids whose existing pixels count toward a real channel.
# Applied only when the merge target actually has a channel.
_ALIASES_ALL = {
    6:  17,   # Domestic Pit  -> Water Storage Containers
    12: 5,    # Inter_Crops   -> Other_crop
}
RAW_MERGE_ALIASES = {a: t for a, t in _ALIASES_ALL.items() if t in RAW_CLASSES}

RAW_TO_MODEL = {raw: i for i, raw in enumerate(RAW_CLASSES)}
for _a, _t in RAW_MERGE_ALIASES.items():
    RAW_TO_MODEL[_a] = RAW_TO_MODEL[_t]

CLASS_NAMES = {i: RAW_NAMES[raw] for i, raw in enumerate(RAW_CLASSES)}
NUM_CLASSES = len(RAW_CLASSES)

# role + priority per MODEL id (used by smart_split_v2 / split_report_v2)
CLASS_ROLES = {i: CLASS_ROLE.get(raw, "context") for i, raw in enumerate(RAW_CLASSES)}
KEY_CLASS_IDS = sorted(i for i, role in CLASS_ROLES.items() if role in KEY_ROLES)
SURVEY_HABITATS = {i: SURVEY_HABITAT.get(raw) for i, raw in enumerate(RAW_CLASSES)}

# --- sanity checks -----------------------------------------------------------
_unknown = [r for r in ACTIVE_RAW + PENDING_RAW if r not in RAW_NAMES]
if _unknown:
    raise ValueError(f"raw id(s) {_unknown} are not in lulc_class_scheme_v2.RAW_NAMES")
if len(set(RAW_CLASSES)) != len(RAW_CLASSES):
    raise ValueError("RAW_CLASSES contains duplicates")
_dup = set(ACTIVE_RAW) & set(PENDING_RAW)
if _dup:
    raise ValueError(f"raw id(s) {sorted(_dup)} appear in both ACTIVE_RAW and PENDING_RAW")
_both = [a for a in RAW_MERGE_ALIASES if a in RAW_CLASSES]
if _both:
    raise ValueError(f"raw id(s) {_both} are in RAW_CLASSES and also merged — pick one")


# --- legacy model versions ---------------------------------------------------
# Old prediction masks do NOT share this index-to-name mapping.
_LEGACY_NAME = {1: "Rice Fields"}   # before the early/late split


def _legacy_names(raws):
    return {i: _LEGACY_NAME.get(r, RAW_NAMES[r]) for i, r in enumerate(raws)}


V6_RAW_CLASSES = [0, 1, 3, 4, 5, 6, 8, 11, 12, 13, 14, 15, 16]
V9B_RAW_CLASSES = [0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 13, 14, 15, 16, 17, 18, 10]  # deployed 17-channel model

LEGACY_SCHEMES = {
    "v6":  {"raw_classes": V6_RAW_CLASSES,  "class_names": _legacy_names(V6_RAW_CLASSES)},
    "v7a": {"raw_classes": V6_RAW_CLASSES,  "class_names": _legacy_names(V6_RAW_CLASSES)},
    "v9b": {"raw_classes": V9B_RAW_CLASSES, "class_names": _legacy_names(V9B_RAW_CLASSES)},
    "current": {"raw_classes": RAW_CLASSES, "class_names": CLASS_NAMES},
}


def get_scheme(version: str = "current"):
    """Returns (raw_classes, class_names) for the given model version tag."""
    if version not in LEGACY_SCHEMES:
        raise ValueError(f"Unknown model version '{version}'. Known: {list(LEGACY_SCHEMES)}")
    s = LEGACY_SCHEMES[version]
    return s["raw_classes"], s["class_names"]


if __name__ == "__main__":
    print(f"{NUM_CLASSES} output channels (incl. Background)\n")
    print(f"{'mdl':>3} {'raw':>3}  {'class':<26} {'role':<24} {'key':<4} survey habitat")
    for i, raw in enumerate(RAW_CLASSES):
        print(f"{i:>3} {raw:>3}  {CLASS_NAMES[i]:<26} {CLASS_ROLES[i]:<24} "
              f"{'KEY' if i in KEY_CLASS_IDS else '':<4} {SURVEY_HABITATS[i] or '-'}")
    if EXCLUDE_RAW:
        print("\nExcluded (ignore=255 in training):",
              ", ".join(f"{r} {RAW_NAMES[r]}" for r in EXCLUDE_RAW))
    if RAW_MERGE_ALIASES:
        print("Merged legacy ids:", RAW_MERGE_ALIASES)
