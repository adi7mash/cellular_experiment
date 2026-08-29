"""L2 normalization diagnostic for all embedding types (Phase 4b)."""
import json
import glob
import os
import sys
import torch
import numpy as np

def norm_diagnostic(embeddings: torch.Tensor, name: str) -> dict:
    norms = torch.norm(embeddings, dim=-1)
    return {
        "name": name,
        "shape": list(embeddings.shape),
        "norm_mean": norms.mean().item(),
        "norm_std": norms.std().item(),
        "norm_min": norms.min().item(),
        "norm_max": norms.max().item(),
        "norm_median": norms.median().item(),
        "is_unit_normed": bool(torch.allclose(norms, torch.ones_like(norms), atol=1e-4)),
        "fraction_near_unit": (torch.abs(norms - 1.0) < 1e-3).float().mean().item(),
        "norm_cv": (norms.std() / norms.mean()).item() if norms.mean() > 0 else float("inf"),
    }


def load_pt_dir(directory: str) -> tuple[list[str], torch.Tensor]:
    files = sorted(glob.glob(os.path.join(directory, "*.pt")))
    if not files:
        return [], torch.tensor([])

    tensors = []
    names = []
    for f in files:
        t = torch.load(f, map_location="cpu", weights_only=True).float()
        if t.dim() == 1:
            tensors.append(t.unsqueeze(0))
        else:
            tensors.append(t.mean(dim=0, keepdim=True) if t.dim() > 1 else t.unsqueeze(0))
        names.append(os.path.basename(f).replace(".pt", ""))

    stacked = torch.cat(tensors, dim=0)
    return names, stacked


def find_embedding_dirs(checkpoint_dir: str) -> dict:
    dirs = {}
    for subdir in sorted(os.listdir(checkpoint_dir)):
        full = os.path.join(checkpoint_dir, subdir)
        if not os.path.isdir(full):
            continue

        if "molecule_codebook" in subdir:
            for sub2 in ["tokens", "pooled_attention"]:
                path = os.path.join(full, sub2)
                if os.path.isdir(path):
                    dirs[f"molecule_codebook_{sub2}"] = path
        elif "protein_codebook" in subdir:
            for sub2 in ["tokens", "pooled_attention"]:
                path = os.path.join(full, sub2)
                if os.path.isdir(path):
                    dirs[f"protein_codebook_{sub2}"] = path
        elif "molecule_molecule" in subdir:
            dirs["molecule_projection"] = full
        elif "protein_protein" in subdir:
            dirs["protein_projection"] = full

    return dirs


def main(checkpoint_dir: str, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)

    emb_dirs = find_embedding_dirs(checkpoint_dir)
    print(f"Found embedding directories: {list(emb_dirs.keys())}")

    reports = {}
    for name, path in emb_dirs.items():
        print(f"\nAnalyzing {name} from {path}...")
        entity_names, matrix = load_pt_dir(path)
        if matrix.numel() == 0:
            print(f"  WARNING: No tensors found in {path}")
            continue

        report = norm_diagnostic(matrix, name)
        reports[name] = report

        cv = report["norm_cv"]
        print(f"  Shape: {report['shape']}")
        print(f"  Norm: mean={report['norm_mean']:.6f}, std={report['norm_std']:.6f}, "
              f"min={report['norm_min']:.6f}, max={report['norm_max']:.6f}")
        print(f"  Unit-normed: {report['is_unit_normed']}, fraction near unit: {report['fraction_near_unit']:.4f}")
        print(f"  CV (std/mean): {cv:.6f}")

        if report["is_unit_normed"]:
            print(f"  -> ALREADY UNIT-NORMED. Use raw dot product as cosine similarity.")
        elif cv < 0.05:
            print(f"  -> LOW VARIANCE norms. L2-normalizing is safe, compare both.")
        else:
            print(f"  -> MEANINGFUL VARIANCE in norms. Run all 3 conditions (raw dot, cosine, norm-as-feature).")

    out_path = os.path.join(results_dir, "norm_diagnostics.json")
    with open(out_path, "w") as f:
        json.dump(reports, f, indent=2)
    print(f"\nSaved diagnostics to {out_path}")
    return reports


if __name__ == "__main__":
    checkpoint_dir = sys.argv[1]
    results_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(checkpoint_dir, "results")
    main(checkpoint_dir, results_dir)
