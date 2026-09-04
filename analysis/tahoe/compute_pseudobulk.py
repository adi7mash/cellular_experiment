"""
Stream Tahoe-100M from HuggingFace and compute pseudobulk mean expression
per (drug, cell_line) combination. Processes one shard at a time to avoid
downloading all 429GB.

Output: pseudobulk_means.npz with:
  - means: [N_conditions, N_genes] dense matrix
  - drug_names: condition drug names
  - cell_line_ids: condition cell line IDs
  - gene_indices: mapping position -> gene index in Tahoe vocabulary
  - cell_counts: number of cells per condition
  - control_means: [N_cell_lines, N_genes] DMSO control means per cell line
  - control_cell_lines: cell line IDs for controls
"""
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from collections import defaultdict
import json
import time
import sys

CACHE_DIR = "/opt/dlami/nvme/tahoe/hf_cache"
OUTPUT = "/opt/dlami/nvme/tahoe/pseudobulk_means.npz"
N_SHARDS = 3388
MAX_SHARDS = 300  # cap to avoid multi-hour download; 300 shards ~ 40GB, ~1.8M cells from target CLs
MAX_GENE_IDX = 62713

# MAP uses 8 cell lines for their main evaluation
MAP_CELL_LINES = {'CVCL_0023', 'CVCL_0069', 'CVCL_0480', 'CVCL_0131',
                  'CVCL_0218', 'CVCL_0504', 'CVCL_1056', 'CVCL_1098'}

def p(*a, **k):
    print(*a, **k, flush=True)

p(f"Starting pseudobulk computation for Tahoe-100M ({N_SHARDS} shards)")
p(f"Filtering to MAP cell lines: {sorted(MAP_CELL_LINES)}")

# Accumulators: (drug, cell_line) -> { sum: sparse dict, count: int }
condition_sums = defaultdict(lambda: defaultdict(float))
condition_counts = defaultdict(int)

# Control accumulators: cell_line -> { sum: sparse dict, count: int }
control_sums = defaultdict(lambda: defaultdict(float))
control_counts = defaultdict(int)

all_gene_indices = set()
t0 = time.time()
total_cells = 0
total_kept = 0

# Sample shards evenly across the full range for better coverage
shard_step = max(1, N_SHARDS // MAX_SHARDS)
shard_indices = list(range(0, N_SHARDS, shard_step))[:MAX_SHARDS]
p(f"Sampling {len(shard_indices)} shards (step={shard_step}) for coverage")

for si, shard_idx in enumerate(shard_indices):
    shard_name = f"data/train-{shard_idx:05d}-of-{N_SHARDS:05d}.parquet"

    try:
        path = hf_hub_download(
            repo_id="tahoebio/Tahoe-100M",
            filename=shard_name,
            repo_type="dataset",
            cache_dir=CACHE_DIR
        )
    except Exception as e:
        p(f"  SKIP shard {shard_idx}: {e}")
        continue

    df = pq.read_table(path).to_pandas()
    total_cells += len(df)

    # Filter to MAP cell lines only
    mask = df['cell_line_id'].isin(MAP_CELL_LINES)
    df_filtered = df[mask]
    total_kept += len(df_filtered)

    if len(df_filtered) == 0:
        if shard_idx % 100 == 0:
            elapsed = time.time() - t0
            p(f"  Shard {shard_idx}/{N_SHARDS} - {total_cells:,} cells processed, "
              f"{total_kept:,} kept - {elapsed:.0f}s")
        continue

    for idx in range(len(df_filtered)):
        row = df_filtered.iloc[idx]
        drug = row['drug']
        cell_line = row['cell_line_id']
        genes = row['genes']
        expressions = row['expressions']

        is_control = ('DMSO' in str(drug) or 'dmso' in str(drug).lower() or
                     drug == 'Vehicle')

        if is_control:
            acc = control_sums[cell_line]
            control_counts[cell_line] += 1
        else:
            key = (drug, cell_line)
            acc = condition_sums[key]
            condition_counts[key] += 1

        # First entry is a special token (gene=1, expr=-2.0) — skip it
        start = 1 if (len(expressions) > 0 and expressions[0] < 0) else 0
        for gi, expr in zip(genes[start:], expressions[start:]):
            acc[gi] += expr
            all_gene_indices.add(gi)

    if si % 20 == 0:
        elapsed = time.time() - t0
        n_conds = len(condition_counts)
        n_ctrl = len(control_counts)
        p(f"  [{si}/{len(shard_indices)}] shard {shard_idx} - {total_cells:,} cells, "
          f"{total_kept:,} kept, {n_conds} conditions, {n_ctrl} controls - "
          f"{elapsed:.0f}s ({(si+1)/(elapsed+1e-9):.2f} shards/s)")

elapsed = time.time() - t0
p(f"\nDone streaming. {total_cells:,} total cells, {total_kept:,} kept.")
p(f"  {len(condition_counts)} drug conditions, {len(control_counts)} control groups")
p(f"  {len(all_gene_indices)} unique genes observed")
p(f"  Elapsed: {elapsed:.0f}s")

# Convert to dense matrices
gene_indices = sorted(all_gene_indices)
gene_to_col = {g: i for i, g in enumerate(gene_indices)}
n_genes = len(gene_indices)

# Conditions
conditions = sorted(condition_counts.keys())
n_conditions = len(conditions)
p(f"\nBuilding dense matrices: {n_conditions} conditions × {n_genes} genes")

means = np.zeros((n_conditions, n_genes), dtype=np.float32)
drug_names = []
cell_line_ids = []
cell_counts = []

for i, (drug, cl) in enumerate(conditions):
    count = condition_counts[(drug, cl)]
    for gi, total in condition_sums[(drug, cl)].items():
        if gi in gene_to_col:
            means[i, gene_to_col[gi]] = total / count
    drug_names.append(drug)
    cell_line_ids.append(cl)
    cell_counts.append(count)

# Controls
control_cls = sorted(control_counts.keys())
control_means = np.zeros((len(control_cls), n_genes), dtype=np.float32)
control_cell_counts = []

for i, cl in enumerate(control_cls):
    count = control_counts[cl]
    for gi, total in control_sums[cl].items():
        if gi in gene_to_col:
            control_means[i, gene_to_col[gi]] = total / count
    control_cell_counts.append(count)

p(f"Conditions matrix: {means.shape}")
p(f"Controls matrix: {control_means.shape}")

np.savez_compressed(OUTPUT,
    means=means,
    drug_names=np.array(drug_names),
    cell_line_ids=np.array(cell_line_ids),
    cell_counts=np.array(cell_counts),
    gene_indices=np.array(gene_indices),
    control_means=control_means,
    control_cell_lines=np.array(control_cls),
    control_cell_counts=np.array(control_cell_counts))

p(f"\nSaved to {OUTPUT}")
p(f"  {n_conditions} conditions (drug × cell_line)")
p(f"  {len(control_cls)} control cell lines")
p(f"  {n_genes} genes")

# Summary stats
p(f"\nCells per condition: min={min(cell_counts)}, median={np.median(cell_counts):.0f}, max={max(cell_counts)}")
p(f"Control cells: {dict(zip(control_cls, control_cell_counts))}")
