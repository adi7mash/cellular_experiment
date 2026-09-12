#!/usr/bin/env python3
"""
Generate improved dimensionality reduction visualizations of Bonbon CG codebook embeddings.

Outputs saved to: /home/adimash/cellular_experiment/results/presentation_visuals/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──
CG_PATH = "/opt/dlami/nvme/rxrx3_phenomics/results/dark-snowball-245/interaction_prints/cg_codebook_tokens.npz"
GENE_ANN_PATH = "/opt/dlami/nvme/rxrx3_phenomics/ground_truth/gene_annotations.tsv"
TAHOE_META_PATH = "/opt/dlami/nvme/tahoe/drug_metadata.csv"
OUT_DIR = "/home/adimash/cellular_experiment/results/presentation_visuals"

# ── Style constants ──
FONT_FAMILY = "DejaVu Sans"
BG_COLOR = "#FFFFFF"
TEXT_COLOR = "#2D2D2D"
GRID_ALPHA = 0.0

# MoA color palette — hand-picked for max distinguishability
MOA_COLORS = {
    "DNA synthesis/repair inhibitor": "#E63946",
    "Cyclooxygenase inhibitor": "#457B9D",
    "Multi-TK inhibitor": "#2A9D8F",
    "HDAC inhibitor": "#E9C46A",
    "MTOR inhibitor": "#F4A261",
    "EGFR/ERBB inhibitor": "#264653",
    "JAK/STAT inhibitor": "#A8DADC",
    "MEK inhibitor": "#6D6875",
    "RAF inhibitor": "#B5838D",
    "Retinoic receptor agonist": "#606C38",
    "Other TK inhibitor": "#118AB2",
    "CDK inhibitor": "#9B2226",
    "Proteasome inhibitor": "#BB3E03",
    "PI3K/AKT inhibitor": "#005F73",
    "DNA methyltransferase inhibitor": "#94D2BD",
    "RAS inhibitor": "#CA6702",
    "Adrenoceptor agonist": "#AE2012",
    "Microtubule inhibitor": "#0A9396",
}

# Cluster colors for unlabeled compounds
CLUSTER_COLORS = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2",
    "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7",
    "#9C755F", "#BAB0AC",
]

PROTEIN_FAMILY_COLORS = {
    "Kinase": "#2166AC",
    "Protease": "#D6604D",
    "Other": "#D0D0D0",
}


def load_data():
    """Load all datasets."""
    cg = np.load(CG_PATH, allow_pickle=True)
    scores = cg["score_cosine"]  # (1673, 735)
    compound_ids = cg["compound_ids"]
    protein_ids = cg["protein_ids"]

    gene_ann = pd.read_csv(GENE_ANN_PATH, sep="\t")
    tahoe_meta = pd.read_csv(TAHOE_META_PATH)

    return scores, compound_ids, protein_ids, gene_ann, tahoe_meta


def style_ax(ax, title, xlabel="UMAP 1", ylabel="UMAP 2"):
    """Apply consistent styling to axes."""
    ax.set_title(title, fontsize=16, fontweight="bold", color=TEXT_COLOR, pad=15,
                 fontfamily=FONT_FAMILY)
    ax.set_xlabel(xlabel, fontsize=12, color=TEXT_COLOR, fontfamily=FONT_FAMILY)
    ax.set_ylabel(ylabel, fontsize=12, color=TEXT_COLOR, fontfamily=FONT_FAMILY)
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.grid(False)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Compound UMAP with density contours
# ═══════════════════════════════════════════════════════════════════════════
def fig_compound_umap(scores, compound_ids, tahoe_meta):
    print("  [1/4] Computing compound UMAP ...")
    scaler = StandardScaler()
    X = scaler.fit_transform(scores)

    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, n_components=2,
                        metric="cosine", random_state=42)
    emb = reducer.fit_transform(X)

    # Build MoA labels for overlapping compounds
    tahoe_moa = tahoe_meta[tahoe_meta["moa-fine"] != "unclear"].set_index("drug")["moa-fine"]
    compound_moa = {}
    for i, cid in enumerate(compound_ids):
        if cid in tahoe_moa.index:
            compound_moa[i] = tahoe_moa[cid]

    # K-means clustering for all compounds
    km = KMeans(n_clusters=10, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(emb)

    fig, ax = plt.subplots(figsize=(12, 10), facecolor=BG_COLOR)

    # KDE density contours
    xy = emb.T
    kde = gaussian_kde(xy, bw_method=0.15)
    xmin, xmax = emb[:, 0].min() - 1.5, emb[:, 0].max() + 1.5
    ymin, ymax = emb[:, 1].min() - 1.5, emb[:, 1].max() + 1.5
    xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    zz = kde(positions).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=12, cmap="Blues", alpha=0.25)
    ax.contour(xx, yy, zz, levels=8, colors="#4393C3", alpha=0.3, linewidths=0.5)

    # Plot unlabeled compounds by cluster
    labeled_idx = set(compound_moa.keys())
    unlabeled_mask = np.array([i not in labeled_idx for i in range(len(compound_ids))])
    for c in range(10):
        mask = (cluster_labels == c) & unlabeled_mask
        if mask.sum() > 0:
            ax.scatter(emb[mask, 0], emb[mask, 1], c=CLUSTER_COLORS[c],
                       s=8, alpha=0.25, edgecolors="none", rasterized=True)

    # Plot MoA-labeled compounds on top
    moa_classes = sorted(set(compound_moa.values()))
    for moa in moa_classes:
        idx = [i for i, m in compound_moa.items() if m == moa]
        color = MOA_COLORS.get(moa, "#333333")
        ax.scatter(emb[idx, 0], emb[idx, 1], c=color, s=40, alpha=0.85,
                   edgecolors="white", linewidths=0.4, label=moa, zorder=5)

    # Annotate cluster centers
    for c in range(10):
        mask = cluster_labels == c
        cx, cy = emb[mask, 0].mean(), emb[mask, 1].mean()
        n = mask.sum()
        ax.annotate(f"C{c}\n(n={n})", (cx, cy), fontsize=7, fontweight="bold",
                    color="#555555", ha="center", va="center",
                    fontfamily=FONT_FAMILY,
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    # Legend — only MoA classes
    leg = ax.legend(loc="upper left", fontsize=7.5, frameon=True, framealpha=0.9,
                    edgecolor="#CCCCCC", title="Mechanism of Action", title_fontsize=9,
                    ncol=1, markerscale=1.5, borderpad=0.8)
    leg.get_frame().set_facecolor("#FAFAFA")

    style_ax(ax, "Compound CG Codebook Embeddings (UMAP)\n"
             "1,673 compounds | 735 protein CG scores | density contours")
    ax.text(0.99, 0.01, "n_neighbors=30, min_dist=0.3, metric=cosine",
            transform=ax.transAxes, fontsize=7, color="#999999", ha="right",
            fontfamily=FONT_FAMILY)

    plt.tight_layout()
    path = f"{OUT_DIR}/fig_compound_umap_improved.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"    Saved: {path}")
    return emb


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Protein UMAP with annotations
# ═══════════════════════════════════════════════════════════════════════════
def fig_protein_umap(scores, protein_ids, gene_ann):
    print("  [2/4] Computing protein UMAP ...")
    # Transpose: proteins x compounds
    X_prot = StandardScaler().fit_transform(scores.T)

    reducer = umap.UMAP(n_neighbors=25, min_dist=0.3, n_components=2,
                        metric="cosine", random_state=42)
    emb = reducer.fit_transform(X_prot)

    # Map protein families
    family_map = dict(zip(gene_ann["gene_symbol"], gene_ann["protein_family"]))
    families = np.array([family_map.get(p, "Other") for p in protein_ids])

    # Hub proteins: highest mean absolute CG score
    hub_scores = np.mean(np.abs(scores), axis=0)
    top_hub_idx = np.argsort(hub_scores)[-12:]

    fig, ax = plt.subplots(figsize=(12, 10), facecolor=BG_COLOR)

    # Draw "Other" first, then Kinase, then Protease
    for fam, color, sz, alpha, zo in [
        ("Other", PROTEIN_FAMILY_COLORS["Other"], 15, 0.35, 1),
        ("Kinase", PROTEIN_FAMILY_COLORS["Kinase"], 35, 0.8, 3),
        ("Protease", PROTEIN_FAMILY_COLORS["Protease"], 35, 0.8, 3),
    ]:
        mask = families == fam
        ax.scatter(emb[mask, 0], emb[mask, 1], c=color, s=sz, alpha=alpha,
                   edgecolors="white" if fam != "Other" else "none",
                   linewidths=0.3, label=f"{fam} (n={mask.sum()})", zorder=zo,
                   rasterized=(fam == "Other"))

    # Convex hulls for kinases and proteases
    for fam, color in [("Kinase", PROTEIN_FAMILY_COLORS["Kinase"]),
                        ("Protease", PROTEIN_FAMILY_COLORS["Protease"])]:
        mask = families == fam
        pts = emb[mask]
        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                hull_pts = np.append(hull.vertices, hull.vertices[0])
                ax.fill(pts[hull_pts, 0], pts[hull_pts, 1],
                        color=color, alpha=0.08, zorder=0)
                ax.plot(pts[hull_pts, 0], pts[hull_pts, 1],
                        color=color, alpha=0.35, linewidth=1.2, linestyle="--", zorder=2)
            except Exception:
                pass

    # KDE contours for kinases
    kinase_mask = families == "Kinase"
    if kinase_mask.sum() > 10:
        kpts = emb[kinase_mask].T
        kde_k = gaussian_kde(kpts, bw_method=0.4)
        xmin, xmax = emb[:, 0].min() - 1, emb[:, 0].max() + 1
        ymin, ymax = emb[:, 1].min() - 1, emb[:, 1].max() + 1
        xx, yy = np.mgrid[xmin:xmax:150j, ymin:ymax:150j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        zz_k = kde_k(positions).reshape(xx.shape)
        ax.contour(xx, yy, zz_k, levels=4, colors=PROTEIN_FAMILY_COLORS["Kinase"],
                   alpha=0.25, linewidths=0.8, linestyles="solid")

    # Label hub proteins
    for idx in top_hub_idx:
        ax.annotate(
            protein_ids[idx],
            (emb[idx, 0], emb[idx, 1]),
            fontsize=7, fontweight="bold",
            color=TEXT_COLOR,
            fontfamily=FONT_FAMILY,
            xytext=(8, 6), textcoords="offset points",
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5),
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
            zorder=10,
        )
        # Highlight hub point
        fam = family_map.get(protein_ids[idx], "Other")
        fc = PROTEIN_FAMILY_COLORS.get(fam, "#333333")
        ax.scatter(emb[idx, 0], emb[idx, 1], s=70, facecolors="none",
                   edgecolors=fc, linewidths=1.5, zorder=9)

    leg = ax.legend(loc="upper right", fontsize=10, frameon=True, framealpha=0.9,
                    edgecolor="#CCCCCC", title="Protein Family", title_fontsize=11,
                    markerscale=1.5)
    leg.get_frame().set_facecolor("#FAFAFA")

    style_ax(ax, "Protein CG Profiles (UMAP)\n"
             "735 proteins profiled across 1,673 compounds | hub proteins labeled")
    ax.text(0.99, 0.01, "n_neighbors=25, min_dist=0.3, metric=cosine",
            transform=ax.transAxes, fontsize=7, color="#999999", ha="right",
            fontfamily=FONT_FAMILY)

    plt.tight_layout()
    path = f"{OUT_DIR}/fig_protein_umap_improved.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"    Saved: {path}")
    return emb


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: Protein PaCMAP
# ═══════════════════════════════════════════════════════════════════════════
def fig_protein_pacmap(scores, protein_ids, gene_ann):
    print("  [3/4] Computing protein PaCMAP ...")
    try:
        import pacmap
    except ImportError:
        print("    SKIPPED: pacmap not installed")
        return None

    X_prot = StandardScaler().fit_transform(scores.T)

    pm = pacmap.PaCMAP(n_components=2, n_neighbors=25, MN_ratio=0.5,
                       FP_ratio=2.0, random_state=42)
    emb = pm.fit_transform(X_prot)

    family_map = dict(zip(gene_ann["gene_symbol"], gene_ann["protein_family"]))
    families = np.array([family_map.get(p, "Other") for p in protein_ids])

    hub_scores = np.mean(np.abs(scores), axis=0)
    top_hub_idx = np.argsort(hub_scores)[-10:]

    fig, ax = plt.subplots(figsize=(12, 10), facecolor=BG_COLOR)

    for fam, color, sz, alpha, zo in [
        ("Other", PROTEIN_FAMILY_COLORS["Other"], 15, 0.35, 1),
        ("Kinase", PROTEIN_FAMILY_COLORS["Kinase"], 40, 0.8, 3),
        ("Protease", PROTEIN_FAMILY_COLORS["Protease"], 40, 0.8, 3),
    ]:
        mask = families == fam
        ax.scatter(emb[mask, 0], emb[mask, 1], c=color, s=sz, alpha=alpha,
                   edgecolors="white" if fam != "Other" else "none",
                   linewidths=0.3, label=f"{fam} (n={mask.sum()})", zorder=zo,
                   rasterized=(fam == "Other"))

    # Convex hulls
    for fam, color in [("Kinase", PROTEIN_FAMILY_COLORS["Kinase"]),
                        ("Protease", PROTEIN_FAMILY_COLORS["Protease"])]:
        mask = families == fam
        pts = emb[mask]
        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                hull_pts = np.append(hull.vertices, hull.vertices[0])
                ax.fill(pts[hull_pts, 0], pts[hull_pts, 1],
                        color=color, alpha=0.08, zorder=0)
                ax.plot(pts[hull_pts, 0], pts[hull_pts, 1],
                        color=color, alpha=0.35, linewidth=1.2, linestyle="--", zorder=2)
            except Exception:
                pass

    # Label hubs
    for idx in top_hub_idx:
        ax.annotate(
            protein_ids[idx],
            (emb[idx, 0], emb[idx, 1]),
            fontsize=7, fontweight="bold",
            color=TEXT_COLOR,
            fontfamily=FONT_FAMILY,
            xytext=(8, 6), textcoords="offset points",
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5),
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
            zorder=10,
        )
        fam = family_map.get(protein_ids[idx], "Other")
        fc = PROTEIN_FAMILY_COLORS.get(fam, "#333333")
        ax.scatter(emb[idx, 0], emb[idx, 1], s=70, facecolors="none",
                   edgecolors=fc, linewidths=1.5, zorder=9)

    leg = ax.legend(loc="upper right", fontsize=10, frameon=True, framealpha=0.9,
                    edgecolor="#CCCCCC", title="Protein Family", title_fontsize=11,
                    markerscale=1.5)
    leg.get_frame().set_facecolor("#FAFAFA")

    style_ax(ax, "Protein CG Profiles (PaCMAP)\n"
             "735 proteins | better global structure preservation",
             xlabel="PaCMAP 1", ylabel="PaCMAP 2")
    ax.text(0.99, 0.01, "n_neighbors=25, MN_ratio=0.5, FP_ratio=2.0",
            transform=ax.transAxes, fontsize=7, color="#999999", ha="right",
            fontfamily=FONT_FAMILY)

    plt.tight_layout()
    path = f"{OUT_DIR}/fig_protein_pacmap.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"    Saved: {path}")
    return emb


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: Bipartite/joint compound-protein visualization
# ═══════════════════════════════════════════════════════════════════════════
def fig_bipartite(scores, compound_ids, protein_ids, gene_ann, tahoe_meta):
    print("  [4/4] Computing bipartite joint embedding ...")
    n_compounds, n_proteins = scores.shape

    # Create joint feature matrix using SVD to project into shared space
    from sklearn.decomposition import TruncatedSVD

    # SVD of the CG score matrix
    svd = TruncatedSVD(n_components=50, random_state=42)
    U_comp = svd.fit_transform(scores)  # compounds in latent space
    V_prot = svd.components_.T  # proteins in latent space

    # Scale so compounds and proteins are comparable
    U_comp = U_comp / (np.linalg.norm(U_comp, axis=1, keepdims=True) + 1e-8)
    V_prot = V_prot / (np.linalg.norm(V_prot, axis=1, keepdims=True) + 1e-8)

    # Stack them for joint UMAP
    joint = np.vstack([U_comp, V_prot])
    entity_type = np.array(["compound"] * n_compounds + ["protein"] * n_proteins)

    reducer = umap.UMAP(n_neighbors=25, min_dist=0.4, n_components=2,
                        metric="cosine", random_state=42)
    emb = reducer.fit_transform(joint)

    emb_comp = emb[:n_compounds]
    emb_prot = emb[n_compounds:]

    # Labels
    family_map = dict(zip(gene_ann["gene_symbol"], gene_ann["protein_family"]))
    families = np.array([family_map.get(p, "Other") for p in protein_ids])

    tahoe_moa = tahoe_meta[tahoe_meta["moa-fine"] != "unclear"].set_index("drug")["moa-fine"]
    compound_moa = {}
    for i, cid in enumerate(compound_ids):
        if cid in tahoe_moa.index:
            compound_moa[i] = tahoe_moa[cid]

    # Cluster compounds
    km = KMeans(n_clusters=8, random_state=42, n_init=10)
    comp_clusters = km.fit_predict(emb_comp)

    fig, ax = plt.subplots(figsize=(14, 10), facecolor=BG_COLOR)

    # Compounds: unlabeled by cluster (small, transparent)
    labeled_idx = set(compound_moa.keys())
    unlabeled_mask = np.array([i not in labeled_idx for i in range(n_compounds)])
    for c in range(8):
        mask = (comp_clusters == c) & unlabeled_mask
        if mask.sum() > 0:
            ax.scatter(emb_comp[mask, 0], emb_comp[mask, 1],
                       c=CLUSTER_COLORS[c], s=6, alpha=0.15,
                       edgecolors="none", rasterized=True, marker="o")

    # Compounds: MoA-labeled
    moa_classes = sorted(set(compound_moa.values()))
    for moa in moa_classes[:10]:  # top 10 for readability
        idx = [i for i, m in compound_moa.items() if m == moa]
        color = MOA_COLORS.get(moa, "#333333")
        ax.scatter(emb_comp[idx, 0], emb_comp[idx, 1], c=color, s=35, alpha=0.8,
                   edgecolors="white", linewidths=0.3, label=f"Drug: {moa}", zorder=5,
                   marker="o")

    # Proteins by family
    for fam, color, sz, alpha, zo in [
        ("Other", "#C0C0C0", 20, 0.25, 1),
        ("Kinase", PROTEIN_FAMILY_COLORS["Kinase"], 50, 0.8, 4),
        ("Protease", PROTEIN_FAMILY_COLORS["Protease"], 50, 0.8, 4),
    ]:
        mask = families == fam
        ax.scatter(emb_prot[mask, 0], emb_prot[mask, 1],
                   c=color, s=sz, alpha=alpha,
                   edgecolors="white" if fam != "Other" else "none",
                   linewidths=0.4,
                   label=f"Protein: {fam} (n={mask.sum()})", zorder=zo,
                   marker="^")

    # Label hub proteins in joint space
    hub_scores = np.mean(np.abs(scores), axis=0)
    top_hub_idx = np.argsort(hub_scores)[-8:]
    for idx in top_hub_idx:
        fam = family_map.get(protein_ids[idx], "Other")
        if fam == "Other":
            continue
        ax.annotate(
            protein_ids[idx],
            (emb_prot[idx, 0], emb_prot[idx, 1]),
            fontsize=7, fontweight="bold",
            color=PROTEIN_FAMILY_COLORS.get(fam, TEXT_COLOR),
            fontfamily=FONT_FAMILY,
            xytext=(8, 6), textcoords="offset points",
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5),
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
            zorder=10,
        )

    # Custom legend: split into two sections
    handles, labels = ax.get_legend_handles_labels()
    # Add separator
    from matplotlib.lines import Line2D
    drug_handles = [h for h, l in zip(handles, labels) if l.startswith("Drug:")]
    drug_labels = [l.replace("Drug: ", "") for l in labels if l.startswith("Drug:")]
    prot_handles = [h for h, l in zip(handles, labels) if l.startswith("Protein:")]
    prot_labels = [l.replace("Protein: ", "") for l in labels if l.startswith("Protein:")]

    # Create compound legend
    if drug_handles:
        leg1 = ax.legend(drug_handles, drug_labels, loc="upper left",
                         fontsize=7, frameon=True, framealpha=0.9,
                         edgecolor="#CCCCCC", title="Drug MoA (circles)", title_fontsize=8,
                         borderpad=0.6)
        leg1.get_frame().set_facecolor("#FAFAFA")
        ax.add_artist(leg1)

    # Create protein legend
    if prot_handles:
        leg2 = ax.legend(prot_handles, prot_labels, loc="lower right",
                         fontsize=8, frameon=True, framealpha=0.9,
                         edgecolor="#CCCCCC", title="Protein Family (triangles)",
                         title_fontsize=9, borderpad=0.6)
        leg2.get_frame().set_facecolor("#FAFAFA")

    style_ax(ax, "Compound-Protein Interaction Landscape (Joint UMAP)\n"
             "1,673 compounds + 735 proteins via SVD of CG score matrix")
    ax.text(0.99, 0.01, "SVD(k=50) + UMAP | circles=drugs, triangles=proteins",
            transform=ax.transAxes, fontsize=7, color="#999999", ha="right",
            fontfamily=FONT_FAMILY)

    plt.tight_layout()
    path = f"{OUT_DIR}/fig_compound_protein_bipartite.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"    Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("Loading data ...")
    scores, compound_ids, protein_ids, gene_ann, tahoe_meta = load_data()
    print(f"  CG matrix: {scores.shape}")
    print(f"  Compounds: {len(compound_ids)}, Proteins: {len(protein_ids)}")
    print(f"  Gene annotations: {len(gene_ann)} genes")
    print(f"  Tahoe drugs: {len(tahoe_meta)} drugs")

    print("\nGenerating figures ...")
    fig_compound_umap(scores, compound_ids, tahoe_meta)
    fig_protein_umap(scores, protein_ids, gene_ann)
    fig_protein_pacmap(scores, protein_ids, gene_ann)
    fig_bipartite(scores, compound_ids, protein_ids, gene_ann, tahoe_meta)

    print("\nAll figures saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
