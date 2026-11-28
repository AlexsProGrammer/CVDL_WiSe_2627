# Analyser EDA script
# Plan convert each input format into the exact same table structure for later Stages



# import Classes from inference.py for single source of truth



# ---

# --- CLI Flags ---

# -- data-dir (required): path to images

# --labels (optional): path to labels

# --format(required): csv, folder, prefix

# --out-dir (optional): path for output


# --check-images (optional): opens each img check for corruption, log resolution stats, render thumbnail grid


# ---

# --- Stage 1: Loader Adapter (csv | folder | prefix)
# Outputs uniform rows: { "path": Path, "raw_label": str }





# ---

# --- Stage 2: Label Normalizer & Target Mapper ---
# Cleans strings -> matches against TARGET_CLASSES -> adds "mapped_id"





# ---

# --- Stage 3: Pass 1 — Fast Tabular Audit (< 1 sec) ---
# Prints class counts, unmatched labels, and imbalance ratio





# ---

# --- Stage 4: Pass 2 — Pixel, Duplicate & Visual Audit ---
# Opens images safely -> checks corrupt files, (W, H), MD5 -> saves contact sheet
# Only runs if --check-images is passed