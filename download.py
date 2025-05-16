
import os
import argparse
from huggingface_hub import snapshot_download, HfFileSystem, list_repo_files
from tqdm import tqdm
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def download_hf_dataset(repo_id, local_dir, repo_type="dataset", revision="main", hf_token=None):
    """
    Downloads all files from a Hugging Face Hub repository to a local directory.

    Args:
        repo_id (str): The ID of the repository on Hugging Face Hub (e.g., "username/dataset_name").
        local_dir (str): The local directory where the dataset will be downloaded.
        repo_type (str, optional): Type of the repository ('dataset', 'model', 'space'). Defaults to "dataset".
        revision (str, optional): The git revision (branch, tag, or commit hash) to download. Defaults to "main".
        hf_token (str, optional): Hugging Face API token for private repositories. Defaults to None.
    """
    print(f"--- Starting Dataset Download ---")
    print(f"Repository ID: {repo_id}")
    print(f"Local Directory: {local_dir}")
    print(f"Repository Type: {repo_type}")
    print(f"Revision: {revision}")

    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)
        print(f"Created local directory: {local_dir}")
    else:
        print(f"Local directory {local_dir} already exists. Files might be overwritten or download might be skipped if already present.")

    try:
        print(f"\nAttempting to download repository '{repo_id}'...")
        # snapshot_download will download all files from the repo.
        # It handles retries and resumes.
        # It also uses caching, so subsequent calls for the same repo/revision might be very fast.
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=local_dir,
            local_dir_use_symlinks=False, # Set to False to copy files instead of symlinking from cache
            revision=revision,
            token=hf_token, # Pass token if provided
            # allow_patterns=["*.wav", "*.json", "*.txt", "*.csv"], # Example: uncomment and adjust if you only want specific file types
            # ignore_patterns=["*.gitattributes", "README.md"], # Example: uncomment to ignore certain files
            resume_download=True,
            # tqdm_class=tqdm # Use custom tqdm if needed, default is good
        )
        print(f"\n--- Download Complete ---")
        print(f"Dataset '{repo_id}' successfully downloaded to: {downloaded_path}") # downloaded_path will be same as local_dir if local_dir_use_symlinks=False
        
        # Verify content (optional simple check)
        print("\nVerifying downloaded content (listing first few files/folders):")
        try:
            fs = HfFileSystem(token=hf_token)
            repo_files = fs.ls(f"{repo_type}s/{repo_id}", revision=revision, detail=False)
            
            downloaded_items = os.listdir(local_dir)
            print(f"Total items listed in repo on Hub: {len(repo_files)}")
            print(f"Total items found in local directory '{local_dir}': {len(downloaded_items)}")
            
            if downloaded_items:
                print("First few items in local directory:")
                for item in downloaded_items[:10]:
                    item_path = os.path.join(local_dir, item)
                    is_dir = os.path.isdir(item_path)
                    print(f"  - {item} {'(Directory)' if is_dir else '(File)'}")
            else:
                print(f"Warning: Local directory '{local_dir}' is empty after download attempt.")

        except Exception as e:
            print(f"Could not perform detailed verification of downloaded content: {e}")

        return True

    except Exception as e:
        print(f"\n--- Download Failed ---")
        print(f"An error occurred while downloading from '{repo_id}':")
        import traceback
        traceback.print_exc()
        print(f"\nPlease check the repository ID, your internet connection, and Hugging Face Hub status.")
        if "401 Client Error" in str(e) or "Authorization" in str(e):
            print("This might be a private repository. Ensure you have access and provide a token if necessary (--hf_token or `huggingface-cli login`).")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download a dataset from Hugging Face Hub.")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="bharathkumar1922001/hindi-tts-multi-speaker", # Default to your specified repo
        help="Hugging Face Hub repository ID (e.g., 'username/dataset_name')."
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="downloaded_dataset_raw", # Default local directory name
        help="Local directory to save the downloaded dataset."
    )
    parser.add_argument(
        "--repo_type",
        type=str,
        default="dataset",
        choices=["dataset", "model", "space"],
        help="Type of the repository on Hugging Face Hub."
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The git revision (branch, tag, or commit hash) to download."
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face API token for accessing private repositories. Can also be set via HUGGING_FACE_HUB_TOKEN environment variable or `huggingface-cli login`."
    )
    parser.add_argument(
        "--force_redownload",
        action="store_true",
        help="If set, will attempt to clear the local_dir before downloading. Use with caution."
    )

    args = parser.parse_args()

    local_dir_abs = os.path.abspath(args.local_dir)

    if args.force_redownload and os.path.exists(local_dir_abs):
        print(f"Force re-download: Removing existing directory '{local_dir_abs}'...")
        try:
            import shutil
            shutil.rmtree(local_dir_abs)
            print(f"Successfully removed '{local_dir_abs}'.")
        except Exception as e:
            print(f"Error removing directory '{local_dir_abs}': {e}. Please remove it manually if needed.")
            return


    download_hf_dataset(
        repo_id=args.repo_id,
        local_dir=local_dir_abs,
        repo_type=args.repo_type,
        revision=args.revision,
        hf_token=args.hf_token
    )

if __name__ == "__main__":
    main()