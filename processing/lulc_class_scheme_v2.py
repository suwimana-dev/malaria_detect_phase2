"""
lulc_class_scheme_v2.py   (v2 — 24-label CVAT scheme, Oct 2026)
==============================================================
Single source of truth for the RAW annotation-time class scheme, shared by
every patch-preparation pipeline. Import NAME_TO_RAW / RAW_NAMES from here
instead of re-typing a copy in each pipeline file.

Two schemes are tracked separately because they change at different rates:

  RAW scheme (this file)  — every class ever annotated, in ANY campaign.
    Patch-prep writes mask pixel VALUES using these raw ids. Adding a newly
    annotated class only ever means adding one line here — it never
    invalidates older patches, which simply never contain the new value.

  MODEL scheme (model_class_scheme_v2.py) — the CURRENT model's contiguous
    0..N-1 output space: a subset + remapping of the raw scheme.

WHAT CHANGED IN v2 (new CVAT label set)
---------------------------------------
  * Existing raw ids are UNCHANGED, so every older mask stays valid.
  * New raw ids 20-26 (appended, nothing renumbered):
        20 Bare Land           21 Natural Vegetation   22 Arable Land
        23 Impervious Surface  24 Shrubs               25 Rice Fields (early)
        26 Marshes
  * "Rice Fields(late)" -> raw 1 (renamed "Rice Fields (late)"); the plain
    legacy label "Rice Fields" (older campaigns, stage not recorded) also
    routes to raw 1.   "Rice Fields (early)" -> new raw 25.
  * "Marshes" is its OWN class now (raw 26). The July 2026 report merged it
    into Other_crop (raw 5); the survey scoring treats marshes as a separate
    habitat, so it is split back out. Masks already converted under the old
    rule still contain Marshes as raw 5 — reconvert if you need them.
  * _norm() now also drops parentheses and collapses spaces, so CVAT's
    "Rice Fields(late)" and "Rice Fields (early)" both resolve.

CLASS ROLES (new)
-----------------
CLASS_ROLE tells every downstream script what a class is FOR in this project:
  water_habitat          holds standing / flowing water the vectors breed in
  wet_vegetation_habitat rice (early/late) and marshes — wet, vegetated habitat
  vegetation_context     other vegetation / crops, relevant for the 2 m edge
                         vegetation band and for LULC mapping
  built_context          settlement, roads, paths, impervious, brick site
  bare_context           bare land
KEY_ROLES = water_habitat + wet_vegetation_habitat. smart_split_v2 gives these
classes extra weight so they are guaranteed to be measurable in val and test.

SURVEY_HABITAT maps a raw id to the habitat key used by risk.py's
HABITAT_LABELS (None = no habitat key there yet).
"""

import re


def _norm(name: str) -> str:
    s = name.strip().lower().replace("_", " ").replace("-", " ")
    s = re.sub(r"[()]", " ", s)
    return " ".join(s.split())


# --- Canonical RAW class scheme -------------------------------------------
RAW_NAMES = {
    0:  "Background",
    1:  "Rice Fields (late)",
    2:  "Fish Ponds",
    3:  "Irrigation Channels",
    4:  "Flooded Areas",
    5:  "Other_crop",
    6:  "Domestic Pit",            # RETIRED -> merged into raw 17
    7:  "Irrigation Ditches",
    8:  "Intercrop Water Drains",
    9:  "Rainwater Puddles",
    10: "Sugarcane",
    11: "Human settlement",
    12: "Inter_Crops",             # RETIRED -> merged into raw 5
    13: "Foot_Path",
    14: "Road",
    15: "Brick Making Pits",
    16: "Brick Site",
    17: "Water Storage Containers",
    18: "River Proximities",
    19: "Construction",
    # ---- NEW in v2 ----
    20: "Bare Land",
    21: "Natural Vegetation",
    22: "Arable Land",
    23: "Impervious Surface",
    24: "Shrubs",
    25: "Rice Fields (early)",
    26: "Marshes",
}

RETIRED_RAW_IDS = {
    6:  "merged into Water Storage Containers (raw 17)",
    12: "merged into Other_crop (raw 5)",
}


def raw_display_name(raw_id: int) -> str:
    name = RAW_NAMES.get(raw_id, f"unknown_{raw_id}")
    if raw_id in RETIRED_RAW_IDS:
        return f"{name} (RETIRED — {RETIRED_RAW_IDS[raw_id]})"
    return name


UNIFIED_CLASSES = sorted(RAW_NAMES.keys())
NUM_RAW_CLASSES = len(UNIFIED_CLASSES)

# --- Roles ------------------------------------------------------------------
CLASS_ROLE = {
    0:  "background",
    # water habitats
    2:  "water_habitat",    # Fish Ponds
    3:  "water_habitat",    # Irrigation Channels
    4:  "water_habitat",    # Flooded Areas
    7:  "water_habitat",    # Irrigation Ditches
    8:  "water_habitat",    # Intercrop Water Drains
    9:  "water_habitat",    # Rainwater Puddles
    15: "water_habitat",    # Brick Making Pits (hold water)
    17: "water_habitat",    # Water Storage Containers
    18: "water_habitat",    # River Proximities
    # wet vegetated habitats
    1:  "wet_vegetation_habitat",    # Rice Fields (late)
    25: "wet_vegetation_habitat",    # Rice Fields (early) — flooded / just planted
    26: "wet_vegetation_habitat",    # Marshes
    # context
    5:  "vegetation_context",        # Other_crop
    22: "vegetation_context",        # Arable Land
    10: "vegetation_context",        # Sugarcane
    21: "vegetation_context",        # Natural Vegetation
    24: "vegetation_context",        # Shrubs
    11: "built_context",             # Human settlement
    23: "built_context",             # Impervious Surface
    14: "built_context",             # Road
    13: "built_context",             # Foot_Path
    16: "built_context",             # Brick Site
    19: "built_context",             # Construction
    20: "bare_context",              # Bare Land
}
KEY_ROLES = {"water_habitat", "wet_vegetation_habitat"}

# raw id -> risk.py HABITAT_LABELS key (None = no habitat key there yet)
SURVEY_HABITAT = {
    1: "rice_fields", 25: "rice_fields", 26: "marshes",
    2: "fish_ponds", 3: "irrigation_channels", 7: "irrigation_ditches",
    8: "intercrop_water_drains", 9: "rainwater_puddles", 15: "pits",
    18: "streams", 5: "other_crop", 10: "sugarcane",
    11: "human_settlement", 13: "foot_path",
    4: None, 17: None, 14: None, 16: None, 19: None,
    20: None, 21: None, 22: None, 23: None, 24: None,
}

# Every labelmap.txt name variant seen across campaigns -> canonical raw id.
NAME_TO_RAW = {
    _norm("background"):                           0,
    # rice — stage-aware in v2
    _norm("Rice Fields"):                          1,    # legacy label, stage unknown -> late
    _norm("Rice Fields(late)"):                    1,
    _norm("Rice Fields (early)"):                  25,
    _norm("rice field boundaries"):                1,
    _norm("Fish Ponds"):                           2,
    _norm("Pools"):                                2,
    _norm("Irrigation Channels"):                  3,
    _norm("Flooded Areas"):                        4,
    _norm("River Proximities"):                    18,
    _norm("Other_crop"):                           5,
    _norm("Other_crop2"):                          5,
    _norm("Vegetable-based Farming"):              5,
    _norm("Inter_Crops"):                          5,    # raw 12 RETIRED
    _norm("Marshes"):                              26,   # v2: own class (was 5)
    _norm("Irrigation Ditches"):                   7,
    _norm("Small Streams"):                        7,
    _norm("Seepage Springs"):                      7,
    _norm("Intercrop Water Drains"):               8,
    _norm("Rainwater Puddles"):                    9,
    _norm("Wheel Prints"):                         9,
    _norm("Foot Prints"):                          9,
    _norm("Hoof Prints"):                          9,
    _norm("Sugarcane"):                            10,
    _norm("Sugarcane fields"):                     10,
    _norm("Human settlement"):                     11,
    _norm("Foot_Path"):                            13,
    _norm("Road"):                                 14,
    _norm("Main-Road"):                            14,
    _norm("Brick Making Pits"):                    15,
    _norm("Mining Pits"):                          15,
    _norm("Sand Mining Pits"):                     15,
    _norm("Stone Mining Pits"):                    15,
    _norm("sand stock pile"):                      15,
    _norm("brick_site"):                           16,
    _norm("Water Storage Containers"):             17,
    _norm("Water Storage Containers in marshland"): 17,
    _norm("Water Storage Containers in Hillside"): 17,
    _norm("Domestic Pit"):                         17,   # raw 6 RETIRED
    _norm("Domestic Pit for Construction Project"): 17,
    _norm("Construction"):                         19,
    # ---- NEW in v2 ----
    _norm("Bare Land"):                            20,
    _norm("Natural Vegetation"):                   21,
    _norm("Arable land"):                          22,
    _norm("Impervious Surface"):                   23,
    _norm("Shrubs"):                               24,
}

# Names deliberately excluded -> resolve to ignore (255).
REMOVED_NAMES = {
    _norm("Edges of Dykes"):
        "Narrow linear feature, difficult to delineate; overlaps with Irrigation Channels",
    _norm("Hill Trenches"):
        "Not observed in current study sites",
    _norm("Hill Drainage Gullies"):
        "Not present in current study sites",
}


# --- import-time sanity checks ---------------------------------------------
_bad = [r for r in NAME_TO_RAW.values() if r not in RAW_NAMES]
if _bad:
    raise ValueError(f"NAME_TO_RAW routes to raw id(s) {sorted(set(_bad))} missing from RAW_NAMES")
_norole = [r for r in RAW_NAMES if r not in CLASS_ROLE and r not in RETIRED_RAW_IDS]
if _norole:
    raise ValueError(f"raw id(s) {_norole} have no CLASS_ROLE entry")
