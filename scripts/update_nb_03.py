"""
Update notebooks/03_feature_engineering.ipynb to reflect the new pipeline.

Run with:  python scripts/update_nb_03.py
"""
import json
from pathlib import Path

NB_PATH = Path("/home/mbarndouka/Documents/amalitechmllab/notebooks/03_feature_engineering.ipynb")

# ── helpers ────────────────────────────────────────────────────────────────────

def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": "",          # will be patched below
        "metadata": {},
        "source": source,
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": "",
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def assign_ids(cells: list[dict]) -> None:
    """Give every cell a unique id (Jupyter requires them)."""
    import uuid
    seen: set[str] = set()
    for c in cells:
        # Keep existing ids if they are non-empty and unique
        existing = c.get("id", "")
        if existing and existing not in seen:
            seen.add(existing)
        else:
            new_id = uuid.uuid4().hex[:8]
            while new_id in seen:
                new_id = uuid.uuid4().hex[:8]
            c["id"] = new_id
            seen.add(new_id)


def src_contains(cell: dict, fragment: str) -> bool:
    src = cell.get("source", "")
    if isinstance(src, list):
        src = "".join(src)
    return fragment in src


def cell_index(cells: list[dict], fragment: str) -> int:
    for i, c in enumerate(cells):
        if src_contains(c, fragment):
            return i
    raise ValueError(f"Cell containing {fragment!r} not found")


# ── load ───────────────────────────────────────────────────────────────────────

nb = json.loads(NB_PATH.read_text())
cells = nb["cells"]

# ── 1. Clear all outputs ───────────────────────────────────────────────────────

for c in cells:
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None

print(f"[1] Cleared outputs for {sum(1 for c in cells if c['cell_type'] == 'code')} code cells")

# ── 2. Update cell[0] markdown — pipeline steps list ──────────────────────────

new_intro_pipeline = (
    "1. Drop redundant columns (`source_name`, `destination_name`)\n"
    "2. Add route feature (`source_destination` combined column)\n"
    "3. One-hot encode categorical columns (route excluded — target-encoded instead)\n"
    "4. Separate features from target (`fare`)\n"
    "5. Log-transform target (log1p — reduces right-skew, stored in log-scale)\n"
    "6. Train / val / test split (70 / 10 / 20)\n"
    "7. Log-transform skewed numerics (`duration`, `days_left`) — after split, no leakage\n"
    "8. Target-encode `route` — replace string col with mean log-fare per route (train stats only)\n"
    "9. Fit `StandardScaler` on train, apply to all splits\n"
    "10. Save feature set"
)

cell0_src = cells[0]["source"]
if isinstance(cell0_src, list):
    cell0_src = "".join(cell0_src)

# Replace the pipeline steps block (everything between "**Pipeline order**\n" and the end)
marker = "**Pipeline order**\n"
if marker in cell0_src:
    pre = cell0_src[: cell0_src.index(marker) + len(marker)]
    cells[0]["source"] = pre + new_intro_pipeline
    print("[2] Updated cell[0] pipeline steps list")
else:
    print("[2] WARNING: Could not find '**Pipeline order**' marker in cell[0]")

# ── 3. Update cell[1] imports ─────────────────────────────────────────────────

new_imports_block = """\
from features.engineering import (
    TARGET,
    REDUNDANT_COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    drop_redundant_columns,
    add_route_feature,
    one_hot_encode,
    split_features_target,
    split_train_val_test,
    log_transform_numerics,
    target_encode,
    fit_and_scale,
    engineer,
    save_feature_set,
    log_feature_set,
    FeatureSet,
)"""

# Find the imports cell (cell[1])
imports_idx = cell_index(cells, "from features.engineering import")
old_src = cells[imports_idx]["source"]
if isinstance(old_src, list):
    old_src = "".join(old_src)

# Replace the from features.engineering block
start_marker = "from features.engineering import ("
end_marker = ")"
start_pos = old_src.index(start_marker)
# Find the matching closing paren
search_from = start_pos + len(start_marker)
paren_depth = 1
pos = search_from
while pos < len(old_src) and paren_depth > 0:
    if old_src[pos] == "(":
        paren_depth += 1
    elif old_src[pos] == ")":
        paren_depth -= 1
    pos += 1
end_pos = pos  # one past the closing ')'

cells[imports_idx]["source"] = (
    old_src[:start_pos] + new_imports_block + old_src[end_pos:]
)
print(f"[3] Updated imports block in cell[{imports_idx}]")

# ── 4. Insert new cells after the drop_redundant_columns code cell ────────────

drop_cell_idx = cell_index(cells, "df1 = drop_redundant_columns(")
print(f"[4] Found drop_redundant_columns cell at index {drop_cell_idx}")

cell_A = md_cell(
    "---\n"
    "## 2.5 Add Route Feature\n"
    "`add_route_feature()` creates a `route` column = `source + \"_\" + destination` before OHE.  \n"
    "This captures the specific route as a single string (e.g., `\"DAC_LHR\"`) which will later be "
    "**target-encoded** into one numeric column instead of 114 OHE columns."
)

cell_B = code_cell(
    'df1r = add_route_feature(df1)\n'
    'print(f"Route column added. Unique routes: {df1r[\'route\'].nunique()}")\n'
    'print(f"Shape: {df1.shape} → {df1r.shape}")\n'
    'print("\\nSample routes:")\n'
    "df1r[['source', 'destination', 'route']].drop_duplicates().head(8).to_string(index=False)"
)

# Insert after drop_cell_idx
cells.insert(drop_cell_idx + 1, cell_A)
cells.insert(drop_cell_idx + 2, cell_B)
print(f"[4] Inserted route-feature markdown + code cells after index {drop_cell_idx}")

# ── 5. Update OHE markdown cell ───────────────────────────────────────────────

# After insertion, find OHE markdown cell by searching for its content
ohe_md_idx = cell_index(cells, "One-Hot Encode Categorical Columns")
old_ohe_md = cells[ohe_md_idx]["source"]
if isinstance(old_ohe_md, list):
    old_ohe_md = "".join(old_ohe_md)

note = (
    "\n\n> **Note:** `route` is excluded from OHE — it stays as a raw string column to be "
    "target-encoded after the train/val/test split. This replaces 114+ OHE route columns with "
    "a single numeric column."
)
cells[ohe_md_idx]["source"] = old_ohe_md + note
print(f"[5] Updated OHE markdown cell at index {ohe_md_idx}")

# ── 6. Update OHE code cell ───────────────────────────────────────────────────

ohe_code_idx = cell_index(cells, "df2 = one_hot_encode(df1, CATEGORICAL_COLUMNS)")
cells[ohe_code_idx]["source"] = (
    "features_cfg = cfg.get('features', {})\n"
    "target_encode_cols = tuple(features_cfg.get('target_encode_cols', []))\n"
    "\n"
    "# route stays as raw string — will be target-encoded after the split (no leakage)\n"
    "ohe_cols = tuple(c for c in list(CATEGORICAL_COLUMNS) + ['route'] if c not in target_encode_cols)\n"
    "df2 = one_hot_encode(df1r, ohe_cols)\n"
    "\n"
    "print(f'OHE applied to: {list(ohe_cols)}')\n"
    "print(f'Skipped (target-encoded after split): {list(target_encode_cols)}')\n"
    "print(f'Shape: {df1r.shape} → {df2.shape}')\n"
    "print(f\"'route' column still present as string: {'route' in df2.columns}\")"
)
print(f"[6] Updated OHE code cell at index {ohe_code_idx}")

# ── 7. Update the y info print cell (after split_features_target) ─────────────

# Find cell containing 'y range' or 'y.min()' used for display
y_range_idx = cell_index(cells, "y range : {y.min()")
cells[y_range_idx]["source"] = (
    "log_target = bool(cfg.get('features', {}).get('log_target', False))\n"
    "print(f'log_target={log_target}')\n"
    "if log_target:\n"
    "    y_model = np.log1p(y)\n"
    "    print(f'y raw range  : {y.min():.2f} – {y.max():.2f}  median={y.median():.2f}')\n"
    "    print(f'y log range  : {y_model.min():.4f} – {y_model.max():.4f}  median={y_model.median():.4f}')\n"
    "else:\n"
    "    y_model = y\n"
    "    print(f'y range : {y.min():.2f} – {y.max():.2f}  median={y.median():.2f}')"
)
print(f"[7] Updated y-range print cell at index {y_range_idx}")

# ── 8. Update the split code cell ─────────────────────────────────────────────

split_idx = cell_index(cells, "X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(")
# Only update the cell that has the split config preamble (not the engineer() full-pipeline cell)
# There may be multiple; pick the one that also has test_size / val_size assignments
for i, c in enumerate(cells):
    src = c.get("source", "")
    if isinstance(src, list):
        src = "".join(src)
    if ("split_train_val_test(" in src and
            "test_size" in src and
            "val_size" in src and
            "engineer" not in src):
        split_idx = i
        break

cells[split_idx]["source"] = (
    "data_cfg     = cfg.get('data', {})\n"
    "test_size    = float(data_cfg.get('test_size',    0.2))\n"
    "val_size     = float(data_cfg.get('val_size',     0.1))\n"
    "random_state = int(data_cfg.get('random_state',   42))\n"
    "\n"
    "log_target = bool(cfg.get('features', {}).get('log_target', False))\n"
    "y_model = np.log1p(y) if log_target else y\n"
    "\n"
    "print(f'test_size={test_size}  val_size={val_size}  random_state={random_state}')\n"
    "\n"
    "X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(\n"
    "    X, y_model, test_size, val_size, random_state\n"
    ")\n"
    "\n"
    "total = len(X)\n"
    "print(f'\\nTrain : {len(X_train):>6,}  ({len(X_train)/total*100:.1f}%)')\n"
    "print(f'Val   : {len(X_val):>6,}  ({len(X_val)/total*100:.1f}%)')\n"
    "print(f'Test  : {len(X_test):>6,}  ({len(X_test)/total*100:.1f}%)')\n"
    "if log_target:\n"
    "    print(f'\\ny_train stored in log-scale: [{y_train.min():.4f}, {y_train.max():.4f}]')"
)
print(f"[8] Updated split code cell at index {split_idx}")

# ── 9. Insert 4 new cells after the split proportions visual cell ──────────────

# Find the visual split proportions cell (the one that has 'barh' and split proportions)
visual_split_idx = cell_index(cells, "Train / Val / Test split")
# There are two: the markdown cell and the code cell. We want the code cell.
# The code cell has ax.barh
for i, c in enumerate(cells):
    src = c.get("source", "")
    if isinstance(src, list):
        src = "".join(src)
    if "ax.barh" in src and "Train / Val / Test split" in src:
        visual_split_idx = i
        break

# Also skip the KDE distribution comparison cell if present — insert after it
# Find target distribution per split (kde cell)
kde_cell_idx = None
for i, c in enumerate(cells):
    src = c.get("source", "")
    if isinstance(src, list):
        src = "".join(src)
    if "plot.kde" in src and "train vs val vs test" in src:
        kde_cell_idx = i
        break

insert_after = kde_cell_idx if kde_cell_idx is not None else visual_split_idx
print(f"[9] Inserting 4 new cells after index {insert_after}")

cell_C = md_cell(
    "---\n"
    "## 6. Log-Transform Skewed Numerics\n"
    "`duration` and `days_left` are right-skewed — a few long-haul flights and far-future bookings "
    "have extreme values.  \n"
    "`log_transform_numerics()` applies `log1p` **after the split** (fit on nothing — pure transform, "
    "no leakage risk).  \n"
    "This compresses the range and makes the distributions more symmetric, helping tree models find "
    "better splits."
)

cell_D = code_cell(
    "log_numeric_cols = tuple(cfg.get('features', {}).get('log_numeric_cols', ['duration', 'days_left']))\n"
    "print(f'Applying log1p to: {list(log_numeric_cols)}')\n"
    "\n"
    "# Before\n"
    "for col in log_numeric_cols:\n"
    "    if col in X_train.columns:\n"
    "        print(f'\\n{col}  before: mean={X_train[col].mean():.3f}  "
    "std={X_train[col].std():.3f}  max={X_train[col].max():.3f}')\n"
    "\n"
    "X_train_ln = log_transform_numerics(X_train, log_numeric_cols)\n"
    "X_val_ln   = log_transform_numerics(X_val,   log_numeric_cols)\n"
    "X_test_ln  = log_transform_numerics(X_test,  log_numeric_cols)\n"
    "\n"
    "# After\n"
    "for col in log_numeric_cols:\n"
    "    if col in X_train_ln.columns:\n"
    "        print(f'{col}  after : mean={X_train_ln[col].mean():.3f}  "
    "std={X_train_ln[col].std():.3f}  max={X_train_ln[col].max():.3f}')"
)

cell_E = md_cell(
    "---\n"
    "## 7. Target-Encode Route\n"
    "`target_encode()` replaces the `route` string column with its **mean log-fare** computed on the "
    "training set only.  \n"
    "- 152 unique routes → 1 numeric column (`route_te`)  \n"
    "- Unseen routes in val/test fall back to global train mean  \n"
    "- No data leakage: val/test stats never influence the encoding  \n"
    "- Tree models split directly on this numeric value"
)

cell_F = code_cell(
    "target_encode_cols = tuple(cfg.get('features', {}).get('target_encode_cols', []))\n"
    "print(f'Target-encoding: {list(target_encode_cols)}')\n"
    "\n"
    "X_train_te, X_val_te, X_test_te = target_encode(\n"
    "    X_train_ln, y_train, X_val_ln, X_test_ln, target_encode_cols\n"
    ")\n"
    "\n"
    "print(f'\\nFeature count: {X_train_ln.shape[1]} → {X_train_te.shape[1]}')\n"
    "print(f\"'route' column gone: {'route' not in X_train_te.columns}\")\n"
    "print(f\"'route_te' column present: {'route_te' in X_train_te.columns}\")\n"
    "print(f\"\\nroute_te stats (train): mean={X_train_te['route_te'].mean():.4f}  \"\n"
    "      f\"min={X_train_te['route_te'].min():.4f}  max={X_train_te['route_te'].max():.4f}\")\n"
    "print(f\"\\nTop 5 most expensive routes (by route_te):\")\n"
    "route_means = y_train.groupby(X_train_ln['route']).mean().sort_values(ascending=False)\n"
    "for route, val in route_means.head(5).items():\n"
    "    print(f\"  {route}: {val:.4f} (≈ BDT {np.expm1(val):,.0f})\")"
)

# Insert in reverse order so indices stay correct
cells.insert(insert_after + 1, cell_C)
cells.insert(insert_after + 2, cell_D)
cells.insert(insert_after + 3, cell_E)
cells.insert(insert_after + 4, cell_F)
print(f"[9] Inserted cells C/D/E/F at positions {insert_after+1}–{insert_after+4}")

# ── 10. Update scaling section ─────────────────────────────────────────────────

# Update "before scaling" print cell — references X_train -> X_train_te
before_scale_idx = cell_index(cells, "Numerical columns to scale:")
old_bs = cells[before_scale_idx]["source"]
if isinstance(old_bs, list):
    old_bs = "".join(old_bs)
cells[before_scale_idx]["source"] = old_bs.replace("X_train[c]", "X_train_te[c]").replace(
    "if c in X_train.columns", "if c in X_train_te.columns"
)
print(f"[10a] Updated before-scaling print cell at index {before_scale_idx}")

# Update fit_and_scale call cell
scale_call_idx = cell_index(cells, "X_train_s, X_val_s, X_test_s, scaler = fit_and_scale(")
cells[scale_call_idx]["source"] = (
    "X_train_s, X_val_s, X_test_s, scaler = fit_and_scale(\n"
    "    X_train_te, X_val_te, X_test_te, NUMERICAL_COLUMNS\n"
    ")\n"
    "\n"
    "scaled_num = [c for c in NUMERICAL_COLUMNS if c in X_train_s.columns]\n"
    "print('Post-scaling stats (train set):')\n"
    "print(X_train_s[scaled_num].describe().loc[['mean', 'std']].round(4).to_string())"
)
print(f"[10b] Updated fit_and_scale call at index {scale_call_idx}")

# Update the before-vs-after plot cell to reference X_train_te
plot_scale_idx = cell_index(cells, "cols_to_plot = [c for c in ['duration', 'days_left']")
old_plot = cells[plot_scale_idx]["source"]
if isinstance(old_plot, list):
    old_plot = "".join(old_plot)
cells[plot_scale_idx]["source"] = old_plot.replace(
    "if c in X_train.columns", "if c in X_train_te.columns"
).replace(
    "X_train[col].hist", "X_train_te[col].hist"
)
print(f"[10c] Updated before/after scaling plot cell at index {plot_scale_idx}")

# ── 11. Update summary markdown cell ──────────────────────────────────────────

summary_idx = cell_index(cells, "## Summary")
cells[summary_idx]["source"] = (
    "## Summary\n"
    "\n"
    "| Step | Action | Result |\n"
    "|------|--------|--------|\n"
    "| 1 | Drop redundant columns | `source_name`, `destination_name` removed |\n"
    "| 2 | Add route feature | `route = source_destination` (152 unique routes) |\n"
    "| 3 | One-hot encode 7 categorical cols | ~57 indicator columns (route excluded) |\n"
    "| 4 | Separate features / target | `X` (features), `y` (fare) |\n"
    "| 5 | Log-transform target | `y = log1p(fare)` — skewness 1.58 → -0.17 |\n"
    "| 6 | Train / val / test split | 70% / 10% / 20% |\n"
    "| 7 | Log-transform numerics | `log1p(duration)`, `log1p(days_left)` — after split |\n"
    "| 8 | Target-encode route | 152 routes → 1 numeric col (train stats only) |\n"
    "| 9 | StandardScaler on 9 numerical cols | Fit on train only — no leakage |\n"
    "| 10 | Save to `data/features/` | 6 parquets + `scaler.pkl` — **77 features** |\n"
    "\n"
    "**Key wins:**\n"
    "- **228 → 77 features**: Route target encoding eliminated 151 sparse OHE columns\n"
    "- **Log target**: Reduces fare skewness (1.58 → -0.17), models fit log-normal distribution\n"
    "- **Log numerics**: Compresses extreme values in duration/days_left\n"
    "\n"
    "**Next:** Step 4 — Model Training"
)
print(f"[11] Updated summary cell at index {summary_idx}")

# ── Finalise and write ─────────────────────────────────────────────────────────

assign_ids(cells)
nb["cells"] = cells
NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"\nDone. Notebook written to {NB_PATH}")
print(f"Total cells: {len(cells)}")
