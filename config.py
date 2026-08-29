import os

DEFAULT_CHECKPOINT = "s3://output-model-checkpoints/checkpoints/final/e_cheese_five_epoch_dark-snowball-245.ckpt"

DATA_DIR = "/opt/dlami/nvme/rxrx3_phenomics/data"
GROUND_TRUTH_DIR = "/opt/dlami/nvme/rxrx3_phenomics/ground_truth"
RESULTS_BASE = "/opt/dlami/nvme/rxrx3_phenomics/results"
EMBEDDINGS_BASE = "/opt/dlami/nvme/rxrx3_phenomics"

MILLEFEUILLE = os.path.expanduser("~/millefeuille/main.py")

CC_THRESHOLD_04 = 0.4
CC_THRESHOLD_06 = 0.6
PER_TARGET_TOP_PERCENT = 0.01
