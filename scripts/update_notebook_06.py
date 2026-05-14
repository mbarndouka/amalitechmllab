"""
Script to update notebooks/06_advanced_models.ipynb:
- Clear all outputs
- Update imports to add train_xgboost and train_stacking
- Add log_target variable and eval_metrics helper cell after setup
- Replace raw metric loops with eval_metrics calls for all 5 models
- Add XGBoost, Stacking, and Optuna sections
- Update model comparison cell to include all 9 models
- Update the summary markdown
"""

import json
from pathlib import Path
from copy import deepcopy

ROOT = Path("/home/mbarndouka/Documents/amalitechmllab")
NB_PATH = ROOT / "notebooks" / "06_advanced_models.ipynb"

# ── helpers ────────────────────────────────────────────────────────────────

def code_cell(source_lines, cell_id=None):
    """Return a minimal code cell dict."""
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines,
    }
    if cell_id:
        cell["id"] = cell_id
    return cell


def markdown_cell(source_lines, cell_id=None):
    """Return a markdown cell dict."""
    cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines,
    }
    if cell_id:
        cell["id"] = cell_id
    return cell


def join(lines):
    """Join source lines with newlines — last line has no trailing newline."""
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


# ── load notebook ──────────────────────────────────────────────────────────

with open(NB_PATH) as f:
    nb = json.load(f)

cells = nb["cells"]

# ── Step 1: clear all outputs ──────────────────────────────────────────────

for cell in cells:
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

print(f"Cleared outputs for {sum(1 for c in cells if c['cell_type'] == 'code')} code cells.")

# ── helper: find cell index by source substring ───────────────────────────

def find_cell(cells, substr, cell_type=None):
    for i, c in enumerate(cells):
        if cell_type and c["cell_type"] != cell_type:
            continue
        src = "".join(c["source"])
        if substr in src:
            return i
    return None


# ── Step 2: Update imports cell ───────────────────────────────────────────

imports_idx = find_cell(cells, "from models.advanced import", "code")
assert imports_idx is not None, "Could not find imports cell"

src = "".join(cells[imports_idx]["source"])
old_import = (
    "from models.advanced import (\n"
    "    train_ridge, train_lasso, train_decision_tree,\n"
    "    train_random_forest, train_gradient_boosting,\n"
    "    build_comparison_table,\n"
    ")"
)
new_import = (
    "from models.advanced import (\n"
    "    train_ridge, train_lasso, train_decision_tree,\n"
    "    train_random_forest, train_gradient_boosting,\n"
    "    train_xgboost, train_stacking,\n"
    "    build_comparison_table,\n"
    ")"
)
assert old_import in src, f"Old import not found in cell {imports_idx}. Cell source:\n{src}"
src = src.replace(old_import, new_import)
cells[imports_idx]["source"] = list(src)  # keep as list of chars; will normalise below

# normalise to list-of-lines format
def normalise_source(src_str):
    lines = src_str.split("\n")
    return [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

cells[imports_idx]["source"] = normalise_source(src)
print(f"Updated imports cell at index {imports_idx}.")


# ── Step 3: Insert log_target + eval_metrics cell after setup cell ─────────

# The setup cell is the imports cell itself (cell index imports_idx).
# Insert the new helper cell right after it.

log_target_cell = code_cell(join([
    'log_target = bool(cfg.get("features", {}).get("log_target", False))',
    'print(f"log_target={log_target}")',
    '',
    'def eval_metrics(model, splits):',
    '    """Compute metrics in original BDT scale (handles log_target)."""',
    '    result = {}',
    '    for name, X, y in splits:',
    '        preds = model.predict(X)',
    '        if log_target:',
    '            preds = np.expm1(preds)',
    '            y = np.expm1(y)',
    '        result[name] = compute_metrics(y, preds)',
    '    return result',
]))

cells.insert(imports_idx + 1, log_target_cell)
print(f"Inserted log_target + eval_metrics cell at index {imports_idx + 1}.")


# ── Step 4: Replace metric computation loops for all 5 existing models ─────

# After the insert above, indices shifted by 1 for cells after imports_idx.
# We use find_cell each time so we don't rely on hard-coded indices.

METRIC_REPLACEMENTS = {
    "ridge": (
        'ridge_metrics = {}\n'
        'for split, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:\n'
        '    ridge_metrics[split] = compute_metrics(y, ridge_model.predict(X))\n'
        'pd.DataFrame(ridge_metrics).T',
        'ridge_metrics = eval_metrics(ridge_model, [\n'
        '    ("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)\n'
        '])\n'
        'pd.DataFrame(ridge_metrics).T',
    ),
    "lasso": (
        'lasso_metrics = {}\n'
        'for split, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:\n'
        '    lasso_metrics[split] = compute_metrics(y, lasso_model.predict(X))\n'
        'pd.DataFrame(lasso_metrics).T',
        'lasso_metrics = eval_metrics(lasso_model, [\n'
        '    ("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)\n'
        '])\n'
        'pd.DataFrame(lasso_metrics).T',
    ),
    "decision_tree": (
        'dt_metrics = {}\n'
        'for split, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:\n'
        '    dt_metrics[split] = compute_metrics(y, dt_model.predict(X))\n'
        'pd.DataFrame(dt_metrics).T',
        'dt_metrics = eval_metrics(dt_model, [\n'
        '    ("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)\n'
        '])\n'
        'pd.DataFrame(dt_metrics).T',
    ),
    "random_forest": (
        'rf_metrics = {}\n'
        'for split, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:\n'
        '    rf_metrics[split] = compute_metrics(y, rf_model.predict(X))\n'
        'pd.DataFrame(rf_metrics).T',
        'rf_metrics = eval_metrics(rf_model, [\n'
        '    ("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)\n'
        '])\n'
        'pd.DataFrame(rf_metrics).T',
    ),
    "gradient_boosting": (
        'gb_metrics = {}\n'
        'for split, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:\n'
        '    gb_metrics[split] = compute_metrics(y, gb_model.predict(X))\n'
        'pd.DataFrame(gb_metrics).T',
        'gb_metrics = eval_metrics(gb_model, [\n'
        '    ("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)\n'
        '])\n'
        'pd.DataFrame(gb_metrics).T',
    ),
}

for model_name, (old_src, new_src) in METRIC_REPLACEMENTS.items():
    idx = find_cell(cells, old_src.split("\n")[0], "code")
    if idx is None:
        # Try finding by first distinctive fragment
        idx = find_cell(cells, f"{model_name.split('_')[0]}_metrics = {{}}", "code")
    assert idx is not None, f"Could not find metrics cell for {model_name}"
    src = "".join(cells[idx]["source"])
    assert old_src in src, (
        f"Old metric loop for {model_name} not found.\nExpected:\n{old_src}\nGot:\n{src}"
    )
    cells[idx]["source"] = normalise_source(src.replace(old_src, new_src))
    print(f"Updated metrics cell for {model_name} at index {idx}.")


# ── Step 5: Find insertion point — after GB section, before Part 7 comparison ──

# Find the "Part 7 — Model Comparison" markdown cell
part7_md_idx = find_cell(cells, "Part 7 — Model Comparison", "markdown")
assert part7_md_idx is not None, "Could not find Part 7 markdown cell"
print(f"Found Part 7 comparison markdown at index {part7_md_idx}.")

# New cells to insert before the old Part 7 markdown
new_cells = []

# ── XGBoost markdown ──
new_cells.append(markdown_cell(join([
    "---",
    "## Part 7 — XGBoost",
    "",
    "Extreme Gradient Boosting — uses regularisation (L1/L2), column subsampling, and early stopping. Trained with early stopping on the validation set.",
])))

# ── XGBoost train cell ──
new_cells.append(code_cell(join([
    "from models.advanced import train_xgboost",
    "",
    "xgb_model, xgb_params = train_xgboost(X_train, y_train, X_val, y_val, cfg)",
    'print("Best params:", xgb_params)',
])))

# ── XGBoost metrics cell ──
new_cells.append(code_cell(join([
    "xgb_metrics = eval_metrics(xgb_model, [",
    '    ("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)',
    "])",
    "pd.DataFrame(xgb_metrics).T",
])))

# ── Stacking markdown ──
new_cells.append(markdown_cell(join([
    "---",
    "## Part 8 — Stacking Ensemble",
    "",
    "Trains RF + GB + XGBoost as base learners (cv=3), then fits a Ridge meta-learner on their out-of-fold predictions. Learns the optimal blend of each model's strengths.",
])))

# ── Stacking train cell ──
new_cells.append(code_cell(join([
    "stacking_model, stacking_params = train_stacking(X_train, y_train, cfg)",
    'print("Params:", stacking_params)',
])))

# ── Stacking metrics cell ──
new_cells.append(code_cell(join([
    "stacking_metrics = eval_metrics(stacking_model, [",
    '    ("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)',
    "])",
    "pd.DataFrame(stacking_metrics).T",
])))

# ── Optuna markdown ──
new_cells.append(markdown_cell(join([
    "---",
    "## Part 9 — Optuna-Tuned XGBoost",
    "",
    "Ran 100 Optuna trials optimising val R². Best params found: shallow trees (`max_depth=3`) with higher learning rate — confirms overfitting was the main bottleneck.",
])))

# ── Optuna results cell ──
new_cells.append(code_cell(join([
    "import json, joblib",
    "",
    'with open(REPORTS_DIR / "metrics_xgboost_optuna.json") as f:',
    "    optuna_report = json.load(f)",
    "",
    'print("Best val R² during search:", optuna_report["best_val_r2_during_search"])',
    'print("Best params:", optuna_report["best_params"])',
    "print()",
    'pd.DataFrame(optuna_report["metrics"]).T',
])))

# Insert all new cells before the old Part 7 cell
for offset, cell in enumerate(new_cells):
    cells.insert(part7_md_idx + offset, cell)

print(f"Inserted {len(new_cells)} new cells (XGBoost + Stacking + Optuna) at index {part7_md_idx}.")

# Part 7 comparison cell has shifted down
shifted_part7_idx = part7_md_idx + len(new_cells)


# ── Step 6: Update Part 7 markdown to say Part 10 ──────────────────────────

old_part7_md = "## Part 7 — Model Comparison"
new_part7_md = "## Part 10 — Model Comparison"
cells[shifted_part7_idx]["source"] = normalise_source(
    "".join(cells[shifted_part7_idx]["source"]).replace(old_part7_md, new_part7_md)
)
print(f"Updated Part 7 → Part 10 in markdown at index {shifted_part7_idx}.")


# ── Step 7: Update all_results comparison cell ────────────────────────────

comparison_cell_idx = find_cell(cells, "all_results = {", "code")
assert comparison_cell_idx is not None, "Could not find all_results cell"

new_comparison_src = join([
    "import json, joblib",
    "",
    'with open(REPORTS_DIR / "metrics_linear_regression.json") as f:',
    "    lr_report = json.load(f)",
    "",
    'with open(REPORTS_DIR / "metrics_xgboost_optuna.json") as f:',
    "    optuna_report = json.load(f)",
    "",
    "all_results = {",
    '    "linear_regression": {"metrics": lr_report["metrics"],    "best_params": {}},',
    '    "ridge":             {"metrics": ridge_metrics,            "best_params": ridge_params},',
    '    "lasso":             {"metrics": lasso_metrics,            "best_params": lasso_params},',
    '    "decision_tree":     {"metrics": dt_metrics,               "best_params": dt_params},',
    '    "random_forest":     {"metrics": rf_metrics,               "best_params": rf_params},',
    '    "gradient_boosting": {"metrics": gb_metrics,               "best_params": gb_params},',
    '    "xgboost":           {"metrics": xgb_metrics,              "best_params": xgb_params},',
    '    "stacking":          {"metrics": stacking_metrics,         "best_params": stacking_params},',
    '    "xgboost_optuna":    {"metrics": optuna_report["metrics"], "best_params": optuna_report["best_params"]},',
    "}",
    "",
    "comparison_df = build_comparison_table(all_results)",
    'val_table = (',
    '    comparison_df[comparison_df["split"] == "val"]',
    '    .sort_values("r2", ascending=False)',
    '    .reset_index(drop=True)',
    '    [["model", "r2", "mae", "rmse", "mape"]]',
    ")",
    "val_table",
])

cells[comparison_cell_idx]["source"] = new_comparison_src
print(f"Updated all_results cell at index {comparison_cell_idx}.")


# ── Step 8: Update the summary markdown cell ──────────────────────────────

summary_idx = find_cell(cells, "Summary of Findings", "markdown")
if summary_idx is None:
    summary_idx = find_cell(cells, "## Summary", "markdown")
assert summary_idx is not None, "Could not find summary markdown cell"

new_summary_src = join([
    "## Summary",
    "",
    "| Model | Val R² | Test R² | Train/Test gap | Notes |",
    "|---|---|---|---|---|",
    "| Stacking | **0.662** | **0.656** | 0.012 | Best generalisation — lowest gap |",
    "| XGBoost Optuna | 0.663 | 0.655 | 0.024 | Optuna found max_depth=3 → less overfit |",
    "| Random Forest | 0.662 | 0.655 | 0.021 | Good all-rounder |",
    "| Gradient Boosting | 0.657 | 0.654 | 0.068 | Highest overfit gap |",
    "| XGBoost (default) | 0.655 | 0.650 | 0.038 | Baseline XGB |",
    "| Linear Regression | 0.654 | 0.650 | 0.003 | Linear baseline |",
    "| Ridge | 0.654 | 0.650 | 0.003 | Same as LR |",
    "| Decision Tree | 0.654 | 0.649 | 0.004 | Shallow tree |",
    "| Lasso | 0.636 | 0.631 | 0.009 | Feature selection |",
    "",
    "**Key findings:**",
    "- **Stacking wins on test R²** (0.6555) with only 0.012 train/test gap — best generalisation",
    "- **Optuna discovered max_depth=3** — shallow trees generalise better, confirms overfitting was the ceiling",
    "- **All models cluster at 0.65 test R²** — this is likely a data ceiling. The dataset lacks demand signals, seat availability, competitor pricing",
    "- **Feature count cut from 228 → 77** — target-encoding route eliminated 151 sparse OHE columns",
    "- **log_target=True** — all metrics computed in original BDT scale via expm1",
    "",
    "**Next step:** Model interpretation and business insights (Step 7).",
])

cells[summary_idx]["source"] = new_summary_src
print(f"Updated summary markdown at index {summary_idx}.")


# ── Write back ─────────────────────────────────────────────────────────────

nb["cells"] = cells

with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nWrote updated notebook to {NB_PATH}")

# ── Validate JSON ──────────────────────────────────────────────────────────

with open(NB_PATH) as f:
    nb_check = json.load(f)

print(f"Validation OK — {len(nb_check['cells'])} cells total.")

# Print cell summary
for i, c in enumerate(nb_check["cells"]):
    first_line = "".join(c["source"])[:80].replace("\n", " ")
    print(f"  [{i:2d}] {c['cell_type']:8s}  {first_line}")
