"""Download the reviewed human proteome from UniProt (Swiss-Prot)."""

import csv
import time
import urllib.request
import urllib.parse

OUTPUT = "/opt/dlami/nvme/rxrx3_phenomics/data/human_proteome.tsv"
BASE_URL = "https://rest.uniprot.org/uniprotkb/search"
QUERY = "(organism_id:9606) AND (reviewed:true)"
FIELDS = "accession,gene_primary,sequence"
SIZE = 500


def fetch_page(cursor=None):
    params = {
        "query": QUERY,
        "format": "tsv",
        "fields": FIELDS,
        "size": str(SIZE),
    }
    if cursor:
        params["cursor"] = cursor

    url = f"{BASE_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Python/OutputBio")

    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read().decode("utf-8")
        link_header = resp.getheader("Link", "")

    next_cursor = None
    if 'rel="next"' in link_header:
        import re
        m = re.search(r'cursor=([^&>]+)', link_header)
        if m:
            next_cursor = m.group(1)

    return data, next_cursor


def main():
    print("Downloading human proteome from UniProt (reviewed Swiss-Prot)...")
    all_rows = []
    cursor = None
    page = 0

    while True:
        data, next_cursor = fetch_page(cursor)
        lines = data.strip().split("\n")

        if page == 0:
            header = lines[0]
            rows = lines[1:]
        else:
            rows = lines[1:] if lines[0].startswith("Entry") else lines

        all_rows.extend(rows)
        page += 1
        print(f"  Page {page}: {len(rows)} entries (total: {len(all_rows)})", flush=True)

        if not next_cursor:
            break
        cursor = next_cursor
        time.sleep(0.5)

    # Write output
    with open(OUTPUT, "w") as f:
        f.write("uniprot_id\tgene_symbol\tsequence\n")
        seen = set()
        skipped = 0
        for row in all_rows:
            parts = row.split("\t")
            if len(parts) < 3:
                skipped += 1
                continue
            uid, gene, seq = parts[0], parts[1], parts[2]
            if not seq or uid in seen:
                skipped += 1
                continue
            seen.add(uid)
            f.write(f"{uid}\t{gene}\t{seq}\n")

    print(f"\nDone: {len(seen)} unique proteins written to {OUTPUT}")
    print(f"  Skipped: {skipped}")


if __name__ == "__main__":
    main()
