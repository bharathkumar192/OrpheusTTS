import os
import glob
from datasets import Dataset, Audio, Features, Value
from huggingface_hub import HfApi, create_repo
import argparse
import soundfile as sf
import re

# --- Configuration (Should match infer.py and your setup) ---
# This list helps map prompt_idx back to the full text.
# Ensure this EXACTLY matches the HINDI_PROMPTS_ARG list used in infer.py
HINDI_PROMPTS_ORDERED_LIST = [
    "यार, ब्रिलियंट आइडिया! कस्टमर फीडबैक को प्रोडक्ट डेवलपमेंट साइकिल में इंटीग्रेट करना बहुत ज़रूरी है, इससे हम मार्केट रिलेवेंट बने रहेंगे, मैं तुम्हारे इस सजेशन से 100% सहमत हूँ डेफिनिटली।",
    "देख भाई, रात को हाईवे पर ड्राइव करना है, तो विज़िबिलिटी के लिए कुछ ब्राइट कलर का पहनना अच्छा आइडिया नहीं है, पर गाड़ी के अंदर के लिए एक जैकेट रख लेना यार।",
    "यार, तुम्हे क्या लगता है, क्या आर्टिफिशियल इंटेलिजेंस फ्यूचर में ह्यूमन क्रिएटिविटी को रिप्लेस कर सकता है, जैसे राइटिंग या म्यूजिक कंपोजिशन में, क्या ये पॉसिबल है?",
    "अगर तुम अपनी पब्लिक स्पीकिंग स्किल्स इम्प्रूव करना चाहते हो, तो टोस्टमास्टर्स क्लब जॉइन करना काफी फायदे मंद हो सकता है, वहां सपोर्टिव एनवायरनमेंट मिलता है प्रैक्टिस करने के लिए, भाई!",
    "वाह, क्या बात है! ये तो कमाल का सुझाव है, बॉस! मुझे पूरा यकीन है, इससे हमारे काम में बहुत सुधार आएगा।"
]
# TARGET_AUDIO_SAMPLING_RATE is used for the datasets.Audio feature.
# It should match the sampling rate of your generated files (from infer.py).
TARGET_AUDIO_SAMPLING_RATE = 24000 

def create_hf_dataset_from_info(audio_files_info):
    """
    Creates a Hugging Face Dataset from a list of audio file information.
    audio_files_info: list of dicts, each with 'file_path', 'speaker_id', 'prompt_id_str', 
                      'prompt_text', 'checkpoint_identifier', 'original_repo_id_if_base'
    """
    data = {
        "audio_filepath": [], 
        "speaker_id": [],
        "prompt_id": [],      
        "prompt_text": [],
        "checkpoint_identifier": [], # e.g., "checkpoint-6184" or "my-model_base"
        "duration_seconds": [],
        "original_sampling_rate": [],
        "original_repo_id": [], # To store the original base repo ID
    }

    for info in audio_files_info:
        try:
            audio_data, sr = sf.read(info['file_path'])
            duration = len(audio_data) / float(sr)
            
            data["audio_filepath"].append(info['file_path'])
            data["speaker_id"].append(info['speaker_id'])
            data["prompt_id"].append(info['prompt_id_str']) 
            data["prompt_text"].append(info['prompt_text'])
            data["checkpoint_identifier"].append(info['checkpoint_identifier'])
            data["duration_seconds"].append(duration)
            data["original_sampling_rate"].append(sr)
            data["original_repo_id"].append(info['original_repo_id'])


        except Exception as e:
            print(f"Skipping file {info['file_path']} due to error during metadata extraction: {e}")
            continue
            
    features = Features({
        'audio': Audio(sampling_rate=TARGET_AUDIO_SAMPLING_RATE), 
        'speaker_id': Value(dtype='string'),
        'prompt_id': Value(dtype='string'),
        'prompt_text': Value(dtype='string'),
        'checkpoint_identifier': Value(dtype='string'),
        'duration_seconds': Value(dtype='float32'),
        'original_sampling_rate': Value(dtype='int32'),
        'original_repo_id': Value(dtype='string'),
    })
    
    hf_dataset = Dataset.from_dict({"audio": data["audio_filepath"], **{k: v for k, v in data.items() if k != "audio_filepath"}}, features=features)
    return hf_dataset

def parse_filepath_metadata(filepath, base_output_dir, original_model_repo_id):
    """
    Parses checkpoint_identifier, speaker_id, prompt_id from filepath.
    Expected filepath structure from revised infer.py:
    base_output_dir/checkpoint_subfolder_or_base_model_name/checkpoint_name_safe_speakerA_prompt1_promptshort.wav

    Args:
        filepath (str): Absolute path to the .wav file.
        base_output_dir (str): The root directory specified to infer.py's --output_dir.
        original_model_repo_id (str): The --model_repo_id passed to infer.py.
    Returns:
        dict or None: Parsed metadata or None if parsing fails.
    """
    try:
        relative_path_to_base = os.path.relpath(filepath, base_output_dir)
        parts = relative_path_to_base.split(os.sep)
        
        if len(parts) < 2:
            print(f"Warning: Filepath {filepath} does not seem to be in a model/checkpoint subdirectory of {base_output_dir}. Skipping.")
            return None
            
        # The directory name IS the checkpoint_identifier (e.g., "checkpoint-6184" or "repo-name_base")
        checkpoint_identifier_from_dir = parts[0]
        filename = parts[-1] 

        # Filename format: {checkpoint_name_safe}_{speaker_id}_prompt{prompt_idx}_{filename_safe_prompt}.wav
        # Example: checkpoint-6184_aisha_prompt1_yaar_brilliant_idea.wav
        # Example: my-model_base_aisha_prompt1_yaar_brilliant_idea.wav (if base_model_repo_id was "user/my-model")

        # We need to be careful here. The `checkpoint_name_safe` part in the filename
        # is the same as `checkpoint_identifier_from_dir`.
        # So we can match from the speaker_id onwards.
        
        # Pattern to match: (anything representing checkpoint_name_safe)_(speaker_id)_prompt(digits)_(anything).wav
        # The first part (checkpoint_name_safe) can be complex, so we'll use the directory name.
        # We need to extract speaker_id and prompt_idx from the latter part of the filename.
        
        # Let's try to match from speaker ID onwards, assuming checkpoint_identifier_from_dir is correct.
        # The filename should start with checkpoint_identifier_from_dir.
        if not filename.startswith(checkpoint_identifier_from_dir):
            print(f"Warning: Filename {filename} does not start with expected checkpoint identifier '{checkpoint_identifier_from_dir}'. Skipping.")
            return None

        # Remove the checkpoint_identifier_from_dir prefix and the first underscore
        remaining_filename_part = filename[len(checkpoint_identifier_from_dir)+1:] # +1 for the underscore

        # Now match: (speaker_id)_prompt(digits)_(anything).wav
        match = re.match(r"(.+?)_prompt(\d+)_.*\.wav", remaining_filename_part)
        if not match:
            print(f"Warning: Filename part '{remaining_filename_part}' (from {filename}) does not match expected pattern (speaker_promptX_...). Skipping.")
            return None
            
        speaker_id = match.group(1)
        prompt_idx_one_based = int(match.group(2)) 
        
        if 1 <= prompt_idx_one_based <= len(HINDI_PROMPTS_ORDERED_LIST):
            prompt_text = HINDI_PROMPTS_ORDERED_LIST[prompt_idx_one_based - 1]
        else:
            prompt_text = "Unknown Prompt (index out of bounds)"
            print(f"Warning: Prompt index {prompt_idx_one_based} for file {filename} is out of bounds for known prompts.")

        prompt_id_str = f"prompt_{prompt_idx_one_based}"
        
        return {
            "speaker_id": speaker_id,
            "prompt_id_str": prompt_id_str,
            "prompt_text": prompt_text,
            "checkpoint_identifier": checkpoint_identifier_from_dir, # This is the key model version identifier
            "original_repo_id": original_model_repo_id, # Store the base repo for context
            "file_path": os.path.abspath(filepath) 
        }
    except Exception as e:
        print(f"Error parsing filepath {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Upload generated audio files from batch inference to Hugging Face Dataset Hub.")
    parser.add_argument(
        "--audio_root_dir",
        type=str,
        required=True,
        help="The root directory where generated audios are stored (e.g., the --output_dir from infer.py)."
    )
    parser.add_argument(
        "--original_model_repo_id",
        type=str,
        required=True,
        help="The --model_repo_id that was passed to the infer.py script (the base repository)."
    )
    parser.add_argument(
        "--dataset_repo_id",
        type=str,
        required=True,
        help="Hugging Face Hub repository ID for the dataset (e.g., YourUsername/YourAudioDatasetName)."
    )
    parser.add_argument(
        "--private_ds",
        action="store_true",
        help="Set the dataset repository to private on Hugging Face Hub."
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face API token (optional, otherwise uses CLI login or environment variable)."
    )

    args = parser.parse_args()

    api = HfApi(token=args.hf_token)
    try:
        create_repo(args.dataset_repo_id, repo_type="dataset", private=args.private_ds, token=args.hf_token, exist_ok=True)
        print(f"Dataset repository '{args.dataset_repo_id}' ensured to exist on Hugging Face Hub.")
    except Exception as e:
        print(f"Error creating/accessing dataset repository '{args.dataset_repo_id}': {e}")
        print("Please ensure you are logged in (`huggingface-cli login`) or have provided a valid token with write access.")
        return

    # Search for .wav files in subdirectories of audio_root_dir
    # The subdirectories are named after the checkpoint_name_safe or repo_base_name
    search_pattern = os.path.join(args.audio_root_dir, "*", "*.wav") 
    all_wav_filepaths = glob.glob(search_pattern, recursive=False) # Should not be recursive here
    
    if not all_wav_filepaths:
        print(f"No .wav files found matching pattern '{search_pattern}'. Please check --audio_root_dir.")
        return
    print(f"Found {len(all_wav_filepaths)} .wav files in subdirectories of {args.audio_root_dir}.")

    parsed_audio_metadata_list = []
    for wav_path in all_wav_filepaths:
        metadata = parse_filepath_metadata(wav_path, args.audio_root_dir, args.original_model_repo_id)
        if metadata:
            parsed_audio_metadata_list.append(metadata)
    
    if not parsed_audio_metadata_list:
        print("No audio file metadata could be parsed successfully. Exiting.")
        return

    print(f"Successfully parsed metadata for {len(parsed_audio_metadata_list)} audio files.")

    print("Creating Hugging Face Dataset object...")
    hf_dataset = create_hf_dataset_from_info(parsed_audio_metadata_list)
    
    if len(hf_dataset) == 0:
        print("Resulting dataset is empty. Exiting before push.")
        return

    print(f"Dataset created with {len(hf_dataset)} samples.")
    print("Dataset features:", hf_dataset.features)
    if len(hf_dataset) > 0:
        print("\nExample of first dataset entry:")
        print(hf_dataset[0])

    print(f"\nAttempting to push dataset to '{args.dataset_repo_id}'...")
    try:
        hf_dataset.push_to_hub(
            repo_id=args.dataset_repo_id,
            private=args.private_ds,
            token=args.hf_token,
            # commit_message="Upload batch inference audio samples" # Optional commit message
        )
        print("\nDataset pushed successfully to the Hugging Face Hub!")
        print(f"You can view your dataset at: https://huggingface.co/datasets/{args.dataset_repo_id}")
    except Exception as e:
        print(f"\nError pushing dataset to Hugging Face Hub: {e}")
        import traceback
        traceback.print_exc()
        print("Common issues: \n- Ensure `git-lfs` is installed and configured (`git lfs install`).\n- Check Hugging Face token permissions or try `huggingface-cli login` again.")

if __name__ == "__main__":
    main()