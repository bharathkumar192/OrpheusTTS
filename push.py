import os
from huggingface_hub import HfApi, upload_folder
from transformers import AutoModelForCausalLM, AutoTokenizer 

def push_checkpoint(checkpoint_path, repo_id, hf_token=None, commit_message="Pushing checkpoint to Hub"):
    if not os.path.isdir(checkpoint_path):
        print(f"Error: Checkpoint path '{checkpoint_path}' does not exist or is not a directory.")
        return

    print(f"Preparing to push checkpoint from: {checkpoint_path}")
    print(f"Target Hub repository ID: {repo_id}")

    api = HfApi(token=hf_token) # Uses env token or login if hf_token is None


    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True) # Set private as needed
        print(f"Repository '{repo_id}' ensured to exist on Hub.")
    except Exception as e:
        print(f"Could not create or ensure repository '{repo_id}': {e}")
        return
    try:
        print("Verifying/Re-saving tokenizer files for consistency...")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path) # This ensures all standard tokenizer files are there
        print("Tokenizer files verified/re-saved.")
    except Exception as e:
        print(f"Warning: Could not re-save tokenizer/config from checkpoint: {e}. Proceeding with existing files.")

    try:
        print(f"Uploading files from '{checkpoint_path}' to '{repo_id}'...")
        upload_folder(
            folder_path=checkpoint_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
            # token=hf_token, # HfApi instance already uses the token
        )
        print(f"Successfully pushed checkpoint to: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Error during upload_folder: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    checkpoint_to_push = "./orpheus/checkpoints_aisha_H200_good_run_v1" 
    target_hub_repo_id = "bharathkumar1922001/orpheus-3b-hi-aisha-20E"

    custom_commit_message = "Aisha 20E 2 * H200 16Batch * 2GAS"
    hf_api_token = None 

    if not os.path.exists(checkpoint_to_push):
        print(f"FATAL: The specified checkpoint path does not exist: {checkpoint_to_push}")
        print("Please verify the path to your checkpoint directory (e.g., a 'checkpoint-XXXX' folder).")
    else:
        push_checkpoint(
            checkpoint_path=checkpoint_to_push,
            repo_id=target_hub_repo_id,
            hf_token=hf_api_token,
            commit_message=custom_commit_message
        )
