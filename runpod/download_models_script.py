# download_models_script.py
import os
from huggingface_hub import snapshot_download
# from snac import SNAC # Only if SNAC.from_pretrained itself downloads, but we use snapshot_download

# Configuration from your original script
SNAC_MODEL_NAME = "hubertsiuzdak/snac_24khz"
MODEL_ID = "bharathkumar1922001/checkpoints_aisha_H200_good_run_v1" # Your fine-tuned Orpheus model
ORPHEUS_CACHE_PATH = "/model_cache/orpheus"
SNAC_CACHE_PATH = "/model_cache/snac"

def download_all_models():
    print(f"Creating model cache directories if they don't exist...")
    os.makedirs(ORPHEUS_CACHE_PATH, exist_ok=True)
    os.makedirs(SNAC_CACHE_PATH, exist_ok=True)

    print(f"Downloading Orpheus model ({MODEL_ID}) to {ORPHEUS_CACHE_PATH}...")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=ORPHEUS_CACHE_PATH,
        local_dir_use_symlinks=False,
        # token="YOUR_HF_TOKEN_IF_PRIVATE_MODEL" # Add if your Orpheus model is private
    )
    print("Orpheus model downloaded.")

    print(f"Downloading SNAC model ({SNAC_MODEL_NAME}) to {SNAC_CACHE_PATH}...")
    snapshot_download(
        repo_id=SNAC_MODEL_NAME,
        local_dir=SNAC_CACHE_PATH,
        local_dir_use_symlinks=False
    )
    print("SNAC model downloaded.")

if __name__ == "__main__":
    download_all_models()