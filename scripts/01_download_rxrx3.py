"""Download RxRx3-core metadata and OpenPhenom embeddings from HuggingFace."""
import shutil
from huggingface_hub import hf_hub_download

DATA_DIR = "/opt/dlami/nvme/rxrx3_phenomics/data"

metadata_path = hf_hub_download(
    "recursionpharma/rxrx3-core",
    filename="metadata_rxrx3_core.csv",
    repo_type="dataset",
)
embeddings_path = hf_hub_download(
    "recursionpharma/rxrx3-core",
    filename="OpenPhenom_rxrx3_core_embeddings.parquet",
    repo_type="dataset",
)

shutil.copy(metadata_path, f"{DATA_DIR}/metadata_rxrx3_core.csv")
shutil.copy(embeddings_path, f"{DATA_DIR}/OpenPhenom_rxrx3_core_embeddings.parquet")

print(f"Metadata: {DATA_DIR}/metadata_rxrx3_core.csv")
print(f"Embeddings: {DATA_DIR}/OpenPhenom_rxrx3_core_embeddings.parquet")
