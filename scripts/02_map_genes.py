"""Map RxRx3-core gene symbols to canonical human UniProt protein sequences via search API."""
import time
import requests
import pandas as pd

DATA_DIR = "/opt/dlami/nvme/rxrx3_phenomics/data"

df = pd.read_csv(f"{DATA_DIR}/metadata_rxrx3_core.csv", low_memory=False)

crispr = df[df["perturbation_type"] == "CRISPR"]
genes = crispr["gene"].dropna().unique()
genes = [g for g in genes if g not in ("EMPTY_control", "CRISPR_control", "")]
genes = sorted(set(genes))
print(f"Unique gene symbols to map: {len(genes)}", flush=True)

BATCH_SIZE = 50
rows = []
unmapped = []
mapped_count = 0

for i in range(0, len(genes), BATCH_SIZE):
    batch = genes[i:i + BATCH_SIZE]
    gene_query = " OR ".join(f"(gene_exact:{g})" for g in batch)
    query = f"({gene_query}) AND (organism_id:9606) AND (reviewed:true)"

    all_entries = []
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {"query": query, "format": "json", "size": 500, "fields": "accession,gene_names,sequence"}

    for attempt in range(3):
        try:
            while True:
                resp = requests.get(url, params=params)
                resp.raise_for_status()
                rj = resp.json()
                all_entries.extend(rj.get("results", []))

                link_header = resp.headers.get("Link", "")
                if 'rel="next"' in link_header:
                    url = link_header.split(";")[0].strip("<>").strip()
                    params = None
                else:
                    break
            break
        except Exception as e:
            print(f"  Batch {i // BATCH_SIZE + 1} attempt {attempt + 1} failed: {e}", flush=True)
            time.sleep(5)
            url = "https://rest.uniprot.org/uniprotkb/search"
            params = {"query": query, "format": "json", "size": 500, "fields": "accession,gene_names,sequence"}
    else:
        for g in batch:
            unmapped.append(g)
        continue

    gene_to_entries = {}
    for entry in all_entries:
        accession = entry.get("primaryAccession", "")
        seq_obj = entry.get("sequence", {})
        sequence = seq_obj.get("value", "") if isinstance(seq_obj, dict) else ""

        gene_names = entry.get("genes", [])
        for ginfo in gene_names:
            gname = ginfo.get("geneName", {}).get("value", "")
            if gname in batch:
                if gname not in gene_to_entries:
                    gene_to_entries[gname] = []
                gene_to_entries[gname].append({
                    "uniprot_id": accession,
                    "sequence": sequence,
                    "length": len(sequence),
                })
            for syn in ginfo.get("synonyms", []):
                sname = syn.get("value", "")
                if sname in batch and sname not in gene_to_entries:
                    gene_to_entries[sname] = []
                    gene_to_entries[sname].append({
                        "uniprot_id": accession,
                        "sequence": sequence,
                        "length": len(sequence),
                    })

    for g in batch:
        if g in gene_to_entries and gene_to_entries[g]:
            best = max(gene_to_entries[g], key=lambda x: x["length"])
            if best["sequence"]:
                rows.append({
                    "gene_symbol": g,
                    "uniprot_id": best["uniprot_id"],
                    "sequence": best["sequence"],
                })
                mapped_count += 1
                continue
        unmapped.append(g)

    print(f"Batch {i // BATCH_SIZE + 1}/{(len(genes) + BATCH_SIZE - 1) // BATCH_SIZE}: "
          f"got {len(all_entries)} entries, mapped {mapped_count}/{i + len(batch)} total", flush=True)
    time.sleep(0.5)

print(f"\nPhase 1 (reviewed search): mapped {len(rows)}, unmapped {len(unmapped)}", flush=True)

if unmapped:
    print(f"\nTrying unmapped genes individually (including unreviewed)...", flush=True)
    still_unmapped = []
    for g in unmapped:
        query = f"(gene_exact:{g}) AND (organism_id:9606)"
        try:
            resp = requests.get(
                "https://rest.uniprot.org/uniprotkb/search",
                params={"query": query, "format": "json", "size": 10, "fields": "accession,gene_names,sequence"},
            )
            resp.raise_for_status()
            entries = resp.json().get("results", [])
            if entries:
                best = max(entries, key=lambda e: len(e.get("sequence", {}).get("value", "")))
                seq = best.get("sequence", {}).get("value", "")
                if seq:
                    rows.append({
                        "gene_symbol": g,
                        "uniprot_id": best["primaryAccession"],
                        "sequence": seq,
                    })
                    continue
            still_unmapped.append(g)
        except Exception:
            still_unmapped.append(g)
        time.sleep(0.3)

    unmapped = still_unmapped
    print(f"After individual lookups: mapped {len(rows)}, still unmapped {len(unmapped)}", flush=True)

out_df = pd.DataFrame(rows)
out_df.to_csv(f"{DATA_DIR}/gene_to_protein.tsv", sep="\t", index=False)
print(f"\nFinal: Mapped {len(rows)} genes, unmapped {len(unmapped)} genes", flush=True)

with open(f"{DATA_DIR}/unmapped_genes.txt", "w") as f:
    for g in unmapped:
        f.write(g + "\n")

if unmapped:
    print(f"Unmapped: {unmapped}", flush=True)
