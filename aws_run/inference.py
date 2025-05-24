import os
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC 
import soundfile as sf
import numpy as np
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import gc

# --- Configuration ---
# Match these with your preprocess.py and training setup
SNAC_MODEL_NAME = "hubertsiuzdak/snac_24khz"
TARGET_AUDIO_SAMPLING_RATE = 24000

# Special Token IDs (ensure consistency with preprocess.py/train.py)
tokeniser_length = 128256
start_of_speech_token = tokeniser_length + 1
end_of_speech_token = tokeniser_length + 2
start_of_human_token = tokeniser_length + 3
end_of_human_token = tokeniser_length + 4
start_of_ai_token = tokeniser_length + 5
end_of_ai_token = tokeniser_length + 6
AUDIO_CODE_BASE_OFFSET = tokeniser_length + 10
PAD_TOKEN_ID = 128263

# SNAC codebook offsets
SNAC_OFFSETS = [
    AUDIO_CODE_BASE_OFFSET + 0 * 4096, # Offset for L0 code (index 0)
    AUDIO_CODE_BASE_OFFSET + 1 * 4096, # Offset for L1a code (index 1)
    AUDIO_CODE_BASE_OFFSET + 2 * 4096, # Offset for L2a code (index 2)
    AUDIO_CODE_BASE_OFFSET + 3 * 4096, # Offset for L2b code (index 3)
    AUDIO_CODE_BASE_OFFSET + 4 * 4096, # Offset for L1b code (index 4)
    AUDIO_CODE_BASE_OFFSET + 5 * 4096, # Offset for L2c code (index 5)
    AUDIO_CODE_BASE_OFFSET + 6 * 4096, # Offset for L2d code (index 6)
]

# --- Predefined Data ---
SPEAKERS = [
    "aisha", "anika", "arfa", "asmr", "nikita", 
    "raju", "rhea", "ruhaan", "sangeeta", "shayana"
]

HINDI_SENTENCES = [
    "ब्रिलियंट आइडिया! कस्टमर फीडबैक को प्रोडक्ट डेवलपमेंट साइकिल में इंटीग्रेट करना बहुत ज़रूरी है, इससे हम मार्केट रिलेवेंट बने रहेंगे, मैं तुम्हारे इस सजेशन से 100% सहमत हूँ डेफिनिटली।",
    "रात को हाईवे पर ड्राइव करना है, तो विज़िबिलिटी के लिए कुछ ब्राइट कलर का पहनना अच्छा आइडिया नहीं है, पर गाड़ी के अंदर के लिए एक जैकेट रख लेना।",
    "तुम्हे क्या लगता है, क्या आर्टिफिशियल इंटेलिजेंस फ्यूचर में ह्यूमन क्रिएटिविटी को रिप्लेस कर सकता है, जैसे राइटिंग या म्यूजिक कंपोजिशन में, क्या ये पॉसिबल है?",
    "अगर तुम अपनी पब्लिक स्पीकिंग स्किल्स इम्प्रूव करना चाहते हो, तो टोस्टमास्टर्स क्लब जॉइन करना काफी फायदेमंद हो सकता है, वहां सपोर्टिव एनवायरनमेंट मिलता है प्रैक्टिस करने के लिए।",
    "यार, इस प्रोजेक्ट में इतना स्ट्रेस क्यों है भाई! डेडलाइन तो अभी भी 2 दिन बाकी है, चिल मारो यार, सब कुछ हो जाएगा, टेंशन मत लो!"
]

# --- Helper Functions ---
def deinterleave_snac_codes(snac_codes_flat, device):
    """De-interleaves the flat list of 7 SNAC codes per frame back into the 3 separate code levels expected by the SNAC decoder."""
    if not snac_codes_flat or len(snac_codes_flat) % 7 != 0:
        print(f"Warning: Invalid flat SNAC code list length ({len(snac_codes_flat)}). Must be multiple of 7.")
        return None, None, None

    num_frames = len(snac_codes_flat) // 7
    codes_lvl0 = []
    codes_lvl1 = []
    codes_lvl2 = []

    for i in range(0, len(snac_codes_flat), 7):
        codes_lvl0.append(snac_codes_flat[i]   - SNAC_OFFSETS[0]) # L0
        codes_lvl1.append(snac_codes_flat[i+1] - SNAC_OFFSETS[1]) # L1a
        codes_lvl2.append(snac_codes_flat[i+2] - SNAC_OFFSETS[2]) # L2a
        codes_lvl2.append(snac_codes_flat[i+3] - SNAC_OFFSETS[3]) # L2b
        codes_lvl1.append(snac_codes_flat[i+4] - SNAC_OFFSETS[4]) # L1b
        codes_lvl2.append(snac_codes_flat[i+5] - SNAC_OFFSETS[5]) # L2c
        codes_lvl2.append(snac_codes_flat[i+6] - SNAC_OFFSETS[6]) # L2d

    try:
        codes_lvl0_tensor = torch.tensor(codes_lvl0, dtype=torch.long, device=device).unsqueeze(0)
        codes_lvl1_tensor = torch.tensor(codes_lvl1, dtype=torch.long, device=device).unsqueeze(0)
        codes_lvl2_tensor = torch.tensor(codes_lvl2, dtype=torch.long, device=device).unsqueeze(0)

        if codes_lvl0_tensor.shape[1] != num_frames: 
            raise ValueError("L0 shape mismatch")
        if codes_lvl1_tensor.shape[1] != 2 * num_frames: 
            raise ValueError("L1 shape mismatch")
        if codes_lvl2_tensor.shape[1] != 4 * num_frames: 
            raise ValueError("L2 shape mismatch")

    except Exception as e:
        print(f"Error creating tensors during de-interleaving: {e}")
        return None, None, None

    return codes_lvl0_tensor, codes_lvl1_tensor, codes_lvl2_tensor

def safe_filename(text, max_length=40):
    """Create a safe filename from text."""
    safe_text = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in text)
    return safe_text.replace(' ', '_')[:max_length]

def load_models(checkpoint_path, device):
    """Load Orpheus and SNAC models."""
    print(f"Loading models from {checkpoint_path}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        if tokenizer.pad_token_id is None:
            tokenizer.add_special_tokens({'pad_token': str(PAD_TOKEN_ID)})
            tokenizer.pad_token_id = PAD_TOKEN_ID

        model_load_kwargs = {
            "torch_dtype": torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16 if device == "cuda" else torch.float32
        }
        
        pt_version = torch.__version__
        if tuple(map(int, pt_version.split('.')[:2])) >= (2, 0) and device=="cuda" and torch.cuda.get_device_capability()[0] >= 8:
            model_load_kwargs["attn_implementation"] = "flash_attention_2"
        
        orpheus_model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_load_kwargs)
        orpheus_model.to(device)
        orpheus_model.eval()

        # Load SNAC model once and reuse
        snac_model = SNAC.from_pretrained(SNAC_MODEL_NAME)
        snac_model.to(device)
        snac_model.eval()
        
        print(f"Successfully loaded models from {checkpoint_path}")
        return tokenizer, orpheus_model, snac_model
        
    except Exception as e:
        print(f"ERROR loading models from {checkpoint_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def generate_single_audio(orpheus_model, snac_model, tokenizer, prompt_text, speaker_id, device):
    """Generate a single audio file."""
    try:
        # 1. Format Input
        full_prompt_text = f"{speaker_id}: {prompt_text}"
        input_ids_list = (
            [start_of_human_token] +
            tokenizer.encode(full_prompt_text, add_special_tokens=False) +
            [end_of_human_token, start_of_ai_token, start_of_speech_token]
        )
        input_ids = torch.tensor([input_ids_list], device=device)

        # 2. Generate SNAC Codes with Orpheus
        with torch.inference_mode():
            generated_ids = orpheus_model.generate(
                input_ids,
                max_new_tokens=2048,
                num_beams=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=[end_of_speech_token, end_of_ai_token],
            )
        
        generated_sequence = generated_ids[0][input_ids.shape[1]:].tolist()
        snac_codes_generated = []
        
        for token_id in generated_sequence:
            if token_id == end_of_speech_token or token_id == end_of_ai_token:
                break
            if token_id >= AUDIO_CODE_BASE_OFFSET and token_id < (AUDIO_CODE_BASE_OFFSET + 4096 * 7):
                snac_codes_generated.append(token_id)

        if not snac_codes_generated:
            return None, "No SNAC codes generated"
            
        if len(snac_codes_generated) % 7 != 0:
            snac_codes_generated = snac_codes_generated[:-(len(snac_codes_generated) % 7)]
            if not snac_codes_generated:
                return None, "No valid SNAC codes after truncation"

        # 3. De-interleave SNAC codes
        codes_l0, codes_l1, codes_l2 = deinterleave_snac_codes(snac_codes_generated, device)
        if codes_l0 is None:
            return None, "Failed to de-interleave codes"

        # 4. Decode SNAC codes to Audio
        with torch.inference_mode():
            waveform_tensor = snac_model.decode([codes_l0, codes_l1, codes_l2])
            
        if waveform_tensor is None or waveform_tensor.numel() == 0:
            return None, "SNAC decoding produced empty waveform"

        audio_data = waveform_tensor.squeeze().cpu().numpy()
        return audio_data, "Success"
        
    except Exception as e:
        return None, f"Error: {str(e)}"

def process_checkpoint_batch(checkpoint_path, output_base_dir, device_id=0):
    """Process all speakers and sentences for a single checkpoint."""
    checkpoint_name = os.path.basename(checkpoint_path)
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*60}")
    print(f"Processing {checkpoint_name} on {device}")
    print(f"{'='*60}")
    
    # Create checkpoint output directory
    checkpoint_output_dir = os.path.join(output_base_dir, checkpoint_name)
    os.makedirs(checkpoint_output_dir, exist_ok=True)
    
    # Load models for this checkpoint
    tokenizer, orpheus_model, snac_model = load_models(checkpoint_path, device)
    if orpheus_model is None:
        print(f"Failed to load models for {checkpoint_name}")
        return
    
    # Process each speaker
    results = []
    total_files = len(SPEAKERS) * len(HINDI_SENTENCES)
    current_file = 0
    
    start_time = time.time()
    
    for speaker_idx, speaker_id in enumerate(SPEAKERS):
        print(f"\n--- Processing Speaker {speaker_idx + 1}/{len(SPEAKERS)}: {speaker_id} ---")
        
        speaker_output_dir = os.path.join(checkpoint_output_dir, speaker_id)
        os.makedirs(speaker_output_dir, exist_ok=True)
        
        for sentence_idx, prompt_text in enumerate(HINDI_SENTENCES):
            current_file += 1
            print(f"  [{current_file}/{total_files}] Sentence {sentence_idx + 1}: {prompt_text[:50]}...")
            
            sentence_start_time = time.time()
            
            # Generate audio
            audio_data, status = generate_single_audio(
                orpheus_model, snac_model, tokenizer, 
                prompt_text, speaker_id, device
            )
            
            if audio_data is not None:
                # Save audio file
                safe_prompt = safe_filename(prompt_text, 30)
                output_filename = f"{speaker_id}_sentence_{sentence_idx + 1:02d}_{safe_prompt}.wav"
                output_filepath = os.path.join(speaker_output_dir, output_filename)
                
                try:
                    sf.write(output_filepath, audio_data, TARGET_AUDIO_SAMPLING_RATE)
                    processing_time = time.time() - sentence_start_time
                    print(f"    ✓ Saved: {output_filename} ({processing_time:.2f}s)")
                    
                    results.append({
                        "checkpoint": checkpoint_name,
                        "speaker": speaker_id,
                        "sentence_idx": sentence_idx + 1,
                        "sentence": prompt_text,
                        "output_file": output_filepath,
                        "processing_time": processing_time,
                        "status": "success"
                    })
                except Exception as e:
                    print(f"    ✗ Failed to save {output_filename}: {e}")
                    results.append({
                        "checkpoint": checkpoint_name,
                        "speaker": speaker_id,
                        "sentence_idx": sentence_idx + 1,
                        "sentence": prompt_text,
                        "error": str(e),
                        "status": "save_failed"
                    })
            else:
                print(f"    ✗ Generation failed: {status}")
                results.append({
                    "checkpoint": checkpoint_name,
                    "speaker": speaker_id,
                    "sentence_idx": sentence_idx + 1,
                    "sentence": prompt_text,
                    "error": status,
                    "status": "generation_failed"
                })
    
    # Save results summary
    total_time = time.time() - start_time
    summary = {
        "checkpoint": checkpoint_name,
        "total_files": total_files,
        "successful": len([r for r in results if r["status"] == "success"]),
        "failed": len([r for r in results if r["status"] != "success"]),
        "total_processing_time": total_time,
        "avg_time_per_file": total_time / total_files,
        "results": results
    }
    
    summary_path = os.path.join(checkpoint_output_dir, f"{checkpoint_name}_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n{checkpoint_name} Summary:")
    print(f"  Total files: {total_files}")
    print(f"  Successful: {summary['successful']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Avg time per file: {summary['avg_time_per_file']:.2f}s")
    print(f"  Summary saved: {summary_path}")
    
    # Clean up GPU memory
    del orpheus_model, snac_model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

def run_parallel_inference(checkpoint_paths, output_dir, max_parallel_checkpoints=2):
    """Run inference on multiple checkpoints in parallel."""
    print(f"Starting parallel inference for {len(checkpoint_paths)} checkpoints")
    print(f"Maximum parallel checkpoints: {max_parallel_checkpoints}")
    print(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Use ThreadPoolExecutor to process checkpoints in parallel
    with ThreadPoolExecutor(max_workers=max_parallel_checkpoints) as executor:
        futures = []
        
        for i, checkpoint_path in enumerate(checkpoint_paths):
            device_id = i % torch.cuda.device_count() if torch.cuda.is_available() else 0
            future = executor.submit(process_checkpoint_batch, checkpoint_path, output_dir, device_id)
            futures.append((future, os.path.basename(checkpoint_path)))
        
        # Wait for all tasks to complete
        for future, checkpoint_name in futures:
            try:
                future.result()
                print(f"✓ Completed processing {checkpoint_name}")
            except Exception as e:
                print(f"✗ Error processing {checkpoint_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run batch inference with multiple fine-tuned Orpheus TTS model checkpoints.")
    parser.add_argument(
        "--checkpoint_paths",
        nargs='+',
        required=True,
        help="List of paths to the fine-tuned Orpheus model checkpoint directories."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="batch_inference_output",
        help="Base directory to save all generated audio files."
    )
    parser.add_argument(
        "--max_parallel",
        type=int,
        default=2,
        help="Maximum number of checkpoints to process in parallel (default: 2)."
    )
    parser.add_argument(
        "--single_checkpoint",
        type=str,
        help="Process only a single checkpoint (for testing)."
    )

    args = parser.parse_args()

    # Validate checkpoint paths
    valid_checkpoints = []
    checkpoint_paths_to_process = [args.single_checkpoint] if args.single_checkpoint else args.checkpoint_paths
    
    for checkpoint_path in checkpoint_paths_to_process:
        if not os.path.isdir(checkpoint_path):
            print(f"Warning: Checkpoint directory not found: {checkpoint_path}")
        else:
            valid_checkpoints.append(checkpoint_path)
    
    if not valid_checkpoints:
        print("Error: No valid checkpoint directories found.")
        exit(1)

    print(f"Valid checkpoints found: {len(valid_checkpoints)}")
    for cp in valid_checkpoints:
        print(f"  - {cp}")
    
    print(f"\nTest configuration:")
    print(f"  Speakers: {len(SPEAKERS)} ({', '.join(SPEAKERS)})")
    print(f"  Sentences: {len(HINDI_SENTENCES)}")
    print(f"  Total files per checkpoint: {len(SPEAKERS) * len(HINDI_SENTENCES)}")
    print(f"  Total files across all checkpoints: {len(valid_checkpoints) * len(SPEAKERS) * len(HINDI_SENTENCES)}")
    
    # Run inference
    start_time = time.time()
    
    if args.single_checkpoint:
        process_checkpoint_batch(args.single_checkpoint, args.output_dir, 0)
    else:
        run_parallel_inference(valid_checkpoints, args.output_dir, args.max_parallel)
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"BATCH INFERENCE COMPLETED")
    print(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()