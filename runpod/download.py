import os
from huggingface_hub import snapshot_download

# Paths on the Network Volume (inside the temporary Pod)
# Default RunPod mount point for network volumes is /runpod-volume/
# You can also create a custom mount point like /models if your network volume configuration allows
# For this example, assuming /runpod-volume/
BASE_VOLUME_PATH = "/workspace" 
ORPHEUS_ON_VOLUME_PATH = os.path.join(BASE_VOLUME_PATH, "orpheus_models")
SNAC_ON_VOLUME_PATH = os.path.join(BASE_VOLUME_PATH, "snac_models")

ORPHEUS_MODEL_ID = "bharathkumar1922001/checkpoints_aisha_H200_good_run_v1"
SNAC_MODEL_ID = "hubertsiuzdak/snac_24khz"

def download_to_volume():
    print(f"Ensuring target directories exist on Network Volume...")
    os.makedirs(ORPHEUS_ON_VOLUME_PATH, exist_ok=True)
    os.makedirs(SNAC_ON_VOLUME_PATH, exist_ok=True)

    # Set HF_HUB_ENABLE_HF_TRANSFER for faster downloads if not globally set
    # Best to set it as an environment variable before running the script:
    # export HF_HUB_ENABLE_HF_TRANSFER=1
    print(f"HF_HUB_ENABLE_HF_TRANSFER is: {os.getenv('HF_HUB_ENABLE_HF_TRANSFER')}")


    print(f"Downloading Orpheus model ({ORPHEUS_MODEL_ID}) to {ORPHEUS_ON_VOLUME_PATH}...")
    try:
        snapshot_download(
            repo_id=ORPHEUS_MODEL_ID,
            local_dir=ORPHEUS_ON_VOLUME_PATH,
            local_dir_use_symlinks=False,
            # token=os.getenv("HF_TOKEN") # Optional
        )
        print(f"Orpheus model downloaded to {ORPHEUS_ON_VOLUME_PATH}")
    except Exception as e:
        print(f"ERROR downloading Orpheus model: {e}")
        raise

    print(f"Downloading SNAC model ({SNAC_MODEL_ID}) to {SNAC_ON_VOLUME_PATH}...")
    try:
        snapshot_download(
            repo_id=SNAC_MODEL_ID,
            local_dir=SNAC_ON_VOLUME_PATH,
            local_dir_use_symlinks=False,
            # token=os.getenv("HF_TOKEN") # Optional
        )
        print(f"SNAC model downloaded to {SNAC_ON_VOLUME_PATH}")
    except Exception as e:
        print(f"ERROR downloading SNAC model: {e}")
        raise

if __name__ == "__main__":
    print("Starting model download to Network Volume.")
    # Ensure you have huggingface_hub and hf_transfer installed in the temp pod
    # pip install huggingface_hub hf-transfer
    download_to_volume()
    print("All models download process to Network Volume finished.")