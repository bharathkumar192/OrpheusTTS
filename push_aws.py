#!/usr/bin/env python3
"""
Script to push model checkpoints to Hugging Face Hub
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import HfApi, login, create_repo
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import json

def setup_auth():
    """Setup Hugging Face authentication"""
    try:
        # Check if already authenticated
        api = HfApi()
        user_info = api.whoami()
        print(f"✓ Already authenticated as: {user_info['name']}")
        return True
    except Exception:
        try:
            # Try to login using stored token
            login()
            print("✓ Successfully authenticated with Hugging Face")
            return True
        except Exception as e:
            print("❌ Authentication failed. Please make sure you have a valid HF token.")
            print("You can set it using: huggingface-cli login")
            print("Or set the HF_TOKEN environment variable")
            sys.exit(1)

def get_checkpoint_info(checkpoint_path):
    """Extract information from checkpoint directory"""
    config_path = os.path.join(checkpoint_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    return None

def push_folder(folder_path, repo_name, hf_username, folder_name, private=True):
    """Push a single folder to Hugging Face"""
    print(f"\n🚀 Pushing {folder_name} to {hf_username}/{repo_name}")
    
    api = HfApi()
    
    # Create repository if it doesn't exist
    full_repo_name = f"{hf_username}/{repo_name}"
    try:
        create_repo(
            repo_id=full_repo_name,
            private=private,
            exist_ok=True
        )
        print(f"✓ Repository {full_repo_name} ready")
    except Exception as e:
        print(f"⚠️  Repository creation note: {e}")
    
    # Upload all files in the folder as a subfolder
    try:
        api.upload_folder(
            folder_path=folder_path,
            repo_id=full_repo_name,
            path_in_repo=folder_name,  # Upload as subdirectory
            commit_message=f"Upload {folder_name}"
        )
        print(f"✓ Successfully uploaded {folder_name}")
        return True
    except Exception as e:
        print(f"❌ Failed to upload {folder_name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Push model checkpoints to Hugging Face Hub")
    parser.add_argument("--username", required=True, help="Your Hugging Face username")
    parser.add_argument("--repo-name", required=True, help="Repository name on Hugging Face")
    parser.add_argument("--private", action="store_true", default=True, help="Make repository private (default: True)")
    parser.add_argument("--base-path", default="/Users/bharath/Downloads/AWS_8xH100", help="Base path containing checkpoints (default: current directory)")
    parser.add_argument("--checkpoints", nargs="+", help="Specific checkpoint names to upload (e.g., checkpoint-3092 checkpoint-4638)")
    parser.add_argument("--include-extras", action="store_true", default=False, help="Include logs, generated_eval_samples, and other non-checkpoint folders")
    parser.add_argument("--extra-folders", nargs="*", default=["logs", "generated_eval_samples", "Untitled"], help="Additional folders to upload (default: logs, generated_eval_samples, Untitled)")
    
    args = parser.parse_args()
    
    # Validate repository name format
    if '/' in args.repo_name:
        print(f"❌ Error: Repository name should not contain '/'. You provided: '{args.repo_name}'")
        print(f"💡 Use just the repo name, like: '3checkpoint-10speaker-aws-H100-30per'")
        print(f"   The full repository ID will be: '{args.username}/{args.repo_name.split('/')[-1]}'")
        sys.exit(1)
    
    print("🤗 Hugging Face Model Checkpoint Uploader")
    print("=" * 50)
    
    # Setup authentication
    setup_auth()
    
    # Find all checkpoint directories
    base_path = Path(args.base_path)
    checkpoint_dirs = []
    extra_dirs = []
    
    if args.checkpoints:
        # Upload only specified checkpoints
        for checkpoint_name in args.checkpoints:
            checkpoint_path = base_path / checkpoint_name
            if checkpoint_path.exists() and checkpoint_path.is_dir():
                checkpoint_dirs.append(checkpoint_path)
            else:
                print(f"⚠️  Checkpoint '{checkpoint_name}' not found in {base_path}")
    else:
        # Upload all checkpoints (original behavior)
        for item in base_path.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                checkpoint_dirs.append(item)
    
    # Handle extra folders (logs, generated_eval_samples, etc.)
    if args.include_extras:
        for folder_name in args.extra_folders:
            folder_path = base_path / folder_name
            if folder_path.exists() and folder_path.is_dir():
                extra_dirs.append(folder_path)
            else:
                print(f"⚠️  Extra folder '{folder_name}' not found in {base_path}")
    
    all_dirs = checkpoint_dirs + extra_dirs
    
    if not all_dirs:
        if args.checkpoints or args.include_extras:
            print("❌ None of the specified directories were found!")
            if args.checkpoints:
                print("Specified checkpoints:", args.checkpoints)
            if args.include_extras:
                print("Looking for extra folders:", args.extra_folders)
        else:
            print("❌ No checkpoint directories found!")
        print("Looking in:", base_path.absolute())
        sys.exit(1)
    
    # Sort checkpoints by step number, keep extra folders at the end
    checkpoint_dirs.sort(key=lambda x: int(x.name.split("-")[1]))
    
    if checkpoint_dirs:
        print(f"📁 Found {len(checkpoint_dirs)} checkpoints:")
        for checkpoint in checkpoint_dirs:
            print(f"  - {checkpoint.name}")
    
    if extra_dirs:
        print(f"📂 Found {len(extra_dirs)} extra folders:")
        for folder in extra_dirs:
            print(f"  - {folder.name}")
    
    print(f"📊 Total directories to upload: {len(all_dirs)}")
    
    # Get model info from the first checkpoint (if any)
    if checkpoint_dirs:
        first_checkpoint = checkpoint_dirs[0]
        config_info = get_checkpoint_info(str(first_checkpoint))
        if config_info:
            print(f"\n📋 Model Info:")
            print(f"  - Architecture: {config_info.get('architectures', ['Unknown'])[0]}")
            print(f"  - Model Type: {config_info.get('model_type', 'Unknown')}")
    
    print(f"\n🎯 Target Repository: {args.username}/{args.repo_name}")
    print(f"🔒 Private: {args.private}")
    
    # Confirm before proceeding
    response = input("\nProceed with upload? (y/N): ").strip().lower()
    if response != 'y':
        print("Upload cancelled.")
        sys.exit(0)
    
    # Upload each directory
    successful_uploads = 0
    for directory in all_dirs:
        folder_name = directory.name
        if push_folder(
            str(directory), 
            args.repo_name, 
            args.username, 
            folder_name,
            args.private
        ):
            successful_uploads += 1
    
    print(f"\n📊 Upload Summary:")
    print(f"✓ Successfully uploaded: {successful_uploads}/{len(all_dirs)} directories")
    
    if successful_uploads > 0:
        print(f"\n🎉 Your files are now available at:")
        print(f"  https://huggingface.co/{args.username}/{args.repo_name}")
        print(f"\n📁 Directories uploaded:")
        for directory in all_dirs:
            folder_name = directory.name
            print(f"  - {folder_name}/")
    
    print("\n💡 Usage tips:")
    if checkpoint_dirs:
        print(f"  - Load a specific checkpoint:")
        print(f"    from transformers import AutoModel")
        print(f"    model = AutoModel.from_pretrained('{args.username}/{args.repo_name}', subfolder='{checkpoint_dirs[0].name}')")
        print(f"  - Load latest checkpoint:")
        print(f"    model = AutoModel.from_pretrained('{args.username}/{args.repo_name}', subfolder='{checkpoint_dirs[-1].name}')")
    if extra_dirs:
        print(f"  - Access extra files:")
        for folder in extra_dirs:
            print(f"    {folder.name}/ folder available in the repository")

if __name__ == "__main__":
    main()