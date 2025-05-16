# app_runpod.py
import os
import io
import torch
import torchaudio
# import torchaudio.transforms as T # Not used directly
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC # Assuming 'snac' is installed via requirements.txt
import soundfile as sf
import numpy as np
import time
import runpod
from pydantic import BaseModel, Field
from typing import Optional
import base64 # For returning audio

# --- Global variables to hold models and tokenizer ---
orpheus_model = None
snac_model = None
tokenizer = None
device = "cpu"
MODELS_LOADED_SUCCESSFULLY = False

# --- Configuration ---
FLASH_ATTN_AVAILABLE = False
SNAC_MODEL_NAME = "hubertsiuzdak/snac_24khz"
TARGET_AUDIO_SAMPLING_RATE = 24000
tokeniser_length = 128256
start_of_speech_token = tokeniser_length + 1
end_of_speech_token = tokeniser_length + 2
start_of_human_token = tokeniser_length + 3
end_of_human_token = tokeniser_length + 4
start_of_ai_token = tokeniser_length + 5
end_of_ai_token = tokeniser_length + 6
AUDIO_CODE_BASE_OFFSET = tokeniser_length + 10
PAD_TOKEN_ID = 128263
SNAC_OFFSETS = [
    AUDIO_CODE_BASE_OFFSET + 0 * 4096, AUDIO_CODE_BASE_OFFSET + 1 * 4096,
    AUDIO_CODE_BASE_OFFSET + 2 * 4096, AUDIO_CODE_BASE_OFFSET + 3 * 4096,
    AUDIO_CODE_BASE_OFFSET + 4 * 4096, AUDIO_CODE_BASE_OFFSET + 5 * 4096,
    AUDIO_CODE_BASE_OFFSET + 6 * 4096,
]
MODEL_ID = "bharathkumar1922001/checkpoints_aisha_H200_good_run_v1"

# --- PATHS FOR MODELS ON THE NETWORK VOLUME ---
# Assumes Network Volume is mounted at /runpod-volume/ in the serverless worker
# And you downloaded models into subdirectories like 'orpheus_models' and 'snac_models' AT THE ROOT of the volume.
NETWORK_VOLUME_MOUNT_PATH = os.getenv("RUNPOD_NETWORK_VOLUME_PATH", "/runpod-volume") # Use env var or default
ORPHEUS_CACHE_PATH = os.path.join(NETWORK_VOLUME_MOUNT_PATH, "orpheus_models")
SNAC_CACHE_PATH = os.path.join(NETWORK_VOLUME_MOUNT_PATH, "snac_models")

AUDIO_QUALITY_PRESETS = {
    "low": {"sample_rate": 16000, "subtype": "PCM_16"},
    "medium": {"sample_rate": 24000, "subtype": "PCM_16"},
    "high": {"sample_rate": 24000, "subtype": "PCM_24"},
}

class TTSRequest(BaseModel):
    text: str = Field(..., description="The text to convert to speech")
    speaker: str = Field(default="aisha", description="The speaker voice to use")
    max_tokens: int = Field(default=2048, description="Maximum number of new tokens to generate")
    temperature: float = Field(default=1.0, ge=0.0, le=2.0, description="Generation temperature (0.0-2.0)")
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0, description="Repetition penalty for generation (1.0-2.0)")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top p sampling parameter (0.0-1.0)")
    stream: bool = Field(default=False, description="Whether to stream the audio response")
    output_sample_rate: int = Field(default=24000, description="Output audio sample rate (Hz)")
    audio_quality: str = Field(default="high", description="Audio quality preset (low, medium, high)")
    speed_adjustment: float = Field(default=1.0, ge=0.5, le=1.5, description="Speech speed adjustment factor (0.5-1.5)")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible generation")
    early_stopping: bool = Field(default=True, description="Whether to use early stopping for generation")

def _deinterleave_snac_codes(snac_codes_flat):
    global device
    if not snac_codes_flat or len(snac_codes_flat) % 7 != 0:
        print(f"Warning: Invalid flat SNAC code list length ({len(snac_codes_flat)}).")
        return None, None, None
    num_frames = len(snac_codes_flat) // 7
    codes_lvl0, codes_lvl1, codes_lvl2 = [], [], []
    for i in range(0, len(snac_codes_flat), 7):
        codes_lvl0.append(snac_codes_flat[i]   - SNAC_OFFSETS[0])
        codes_lvl1.append(snac_codes_flat[i+1] - SNAC_OFFSETS[1])
        codes_lvl2.append(snac_codes_flat[i+2] - SNAC_OFFSETS[2])
        codes_lvl2.append(snac_codes_flat[i+3] - SNAC_OFFSETS[3])
        codes_lvl1.append(snac_codes_flat[i+4] - SNAC_OFFSETS[4])
        codes_lvl2.append(snac_codes_flat[i+5] - SNAC_OFFSETS[5])
        codes_lvl2.append(snac_codes_flat[i+6] - SNAC_OFFSETS[6])
    try:
        codes_lvl0_tensor = torch.tensor(codes_lvl0, dtype=torch.long, device=device).unsqueeze(0)
        codes_lvl1_tensor = torch.tensor(codes_lvl1, dtype=torch.long, device=device).unsqueeze(0)
        codes_lvl2_tensor = torch.tensor(codes_lvl2, dtype=torch.long, device=device).unsqueeze(0)
        if codes_lvl0_tensor.shape[1] != num_frames: raise ValueError("L0 shape mismatch")
        if codes_lvl1_tensor.shape[1] != 2 * num_frames: raise ValueError("L1 shape mismatch")
        if codes_lvl2_tensor.shape[1] != 4 * num_frames: raise ValueError("L2 shape mismatch")
    except Exception as e:
        print(f"Error creating tensors during de-interleaving: {e}")
        return None, None, None
    return codes_lvl0_tensor, codes_lvl1_tensor, codes_lvl2_tensor

def _adjust_audio_speed(waveform_tensor, speed_factor):
    global device
    if speed_factor == 1.0:
        return waveform_tensor
    orig_length = waveform_tensor.shape[1]
    new_length = int(orig_length / speed_factor)
    if new_length == 0:
        return torch.empty((waveform_tensor.shape[0], 0), device=device, dtype=waveform_tensor.dtype)
    indices = torch.linspace(0, orig_length - 1, new_length, device=device)
    indices_floor = indices.long()
    indices_ceil = torch.min(indices_floor + 1, torch.tensor(orig_length - 1, device=device))
    weight_ceil = indices - indices_floor.float()
    weight_floor = 1.0 - weight_ceil
    adjusted_waveform = waveform_tensor[:, indices_floor] * weight_floor.unsqueeze(0) + \
                      waveform_tensor[:, indices_ceil] * weight_ceil.unsqueeze(0)
    return adjusted_waveform

def _generate_speech_internal(
    text_prompt: str, speaker_id: str, max_tokens: int, temperature: float,
    repetition_penalty: float, top_p: float, output_sample_rate: int,
    audio_quality: str, speed_adjustment: float, seed: Optional[int],
    early_stopping: bool
) -> bytes:
    global orpheus_model, snac_model, tokenizer, device

    if not MODELS_LOADED_SUCCESSFULLY or tokenizer is None or orpheus_model is None or snac_model is None:
        raise RuntimeError("Models or tokenizer not loaded. init() might have failed or not completed.")

    print(f"\n--- Generating speech for: '{text_prompt}', Speaker: {speaker_id} ---")
    start_time_total = time.time()

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    full_prompt_text = f"{speaker_id}: {text_prompt}"
    try:
        input_ids_list = (
            [start_of_human_token] +
            tokenizer.encode(full_prompt_text, add_special_tokens=False) +
            [end_of_human_token, start_of_ai_token, start_of_speech_token]
        )
        input_ids = torch.tensor([input_ids_list], device=device)
    except Exception as e:
        print(f"Error tokenizing prompt: {e}")
        raise ValueError(f"Tokenization error: {e}")

    print("Generating SNAC codes with Orpheus model...")
    gen_start_time = time.time()
    try:
        with torch.inference_mode():
            generation_kwargs = {
                "input_ids": input_ids, "max_new_tokens": max_tokens,
                "num_beams": 1, "do_sample": temperature > 0.001,
                "temperature": temperature, "repetition_penalty": repetition_penalty,
                "top_p": top_p, "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": [end_of_speech_token, end_of_ai_token],
            }
            if early_stopping: generation_kwargs["early_stopping"] = True
            generated_ids = orpheus_model.generate(**generation_kwargs)

        print(f"Orpheus generation took {time.time() - gen_start_time:.2f} seconds.")
        generated_sequence = generated_ids[0][input_ids.shape[1]:].tolist()
        snac_codes_generated = []
        for token_id in generated_sequence:
            if token_id == end_of_speech_token or token_id == end_of_ai_token: break
            if token_id >= AUDIO_CODE_BASE_OFFSET and token_id < (AUDIO_CODE_BASE_OFFSET + 4096 * 7):
                snac_codes_generated.append(token_id)

        if not snac_codes_generated: raise ValueError("No SNAC codes generated.")
        if len(snac_codes_generated) % 7 != 0:
            print(f"Warning: Truncating SNAC codes from {len(snac_codes_generated)} to multiple of 7.")
            snac_codes_generated = snac_codes_generated[:-(len(snac_codes_generated) % 7)]
            if not snac_codes_generated: raise ValueError("No valid SNAC codes after truncation.")
        print(f"Extracted {len(snac_codes_generated)} SNAC code tokens.")
    except Exception as e:
        print(f"Error during Orpheus generation: {e}")
        raise ValueError(f"Orpheus generation error: {e}")

    codes_l0, codes_l1, codes_l2 = _deinterleave_snac_codes(snac_codes_generated)
    if codes_l0 is None: raise ValueError("Failed to de-interleave SNAC codes.")

    print("Decoding SNAC codes to waveform...")
    decode_start_time = time.time()
    try:
        with torch.inference_mode():
            waveform_tensor = snac_model.decode([codes_l0, codes_l1, codes_l2])
        print(f"SNAC decoding took {time.time() - decode_start_time:.2f} seconds.")
        if waveform_tensor is None or waveform_tensor.numel() == 0:
            raise ValueError("SNAC decoding produced an empty waveform.")

        if speed_adjustment != 1.0:
            waveform_tensor = _adjust_audio_speed(waveform_tensor, speed_adjustment)
        audio_data_np = waveform_tensor.squeeze().cpu().numpy()

        bytes_io = io.BytesIO()
        quality_settings = AUDIO_QUALITY_PRESETS.get(audio_quality, AUDIO_QUALITY_PRESETS["high"])
        final_sample_rate = output_sample_rate or quality_settings["sample_rate"]

        sf.write(
            bytes_io, audio_data_np, final_sample_rate,
            format='WAV', subtype=quality_settings["subtype"]
        )
        wav_bytes = bytes_io.getvalue()
        print(f"Total generation time: {time.time() - start_time_total:.2f} seconds.")
        return wav_bytes
    except Exception as e:
        print(f"Error during SNAC decoding or audio conversion: {e}")
        raise ValueError(f"SNAC decoding/conversion error: {e}")

def _generate_speech_streaming_internal(
    text_prompt: str, speaker_id: str, max_tokens: int, temperature: float,
    repetition_penalty: float, top_p: float, output_sample_rate: int,
    audio_quality: str, speed_adjustment: float, seed: Optional[int]
):
    global orpheus_model, snac_model, tokenizer, device

    if not MODELS_LOADED_SUCCESSFULLY or tokenizer is None or orpheus_model is None or snac_model is None:
        yield f"Error: Models or tokenizer not loaded. init() might have failed.".encode('utf-8')
        return

    print(f"\n--- Streaming speech for: '{text_prompt}', Speaker: {speaker_id} ---")
    if seed is not None:
        torch.manual_seed(seed); np.random.seed(seed)

    full_prompt_text = f"{speaker_id}: {text_prompt}"
    try:
        input_ids_list = (
            [start_of_human_token] + tokenizer.encode(full_prompt_text, add_special_tokens=False) +
            [end_of_human_token, start_of_ai_token, start_of_speech_token]
        )
        input_ids = torch.tensor([input_ids_list], device=device)
    except Exception as e:
        yield f"Error: Tokenization error: {e}".encode('utf-8')
        return

    print("Generating SNAC codes with Orpheus model (for streaming)...")
    current_snac_chunk = []
    chunk_size_tokens = 28 * 2
    quality_settings = AUDIO_QUALITY_PRESETS.get(audio_quality, AUDIO_QUALITY_PRESETS["high"])
    final_sample_rate = output_sample_rate or quality_settings["sample_rate"]

    def process_and_yield_chunk(snac_code_list_for_chunk):
        if not snac_code_list_for_chunk or len(snac_code_list_for_chunk) % 7 != 0: return
        codes_l0, codes_l1, codes_l2 = _deinterleave_snac_codes(snac_code_list_for_chunk)
        if codes_l0 is None: return
        try:
            with torch.inference_mode():
                waveform_tensor = snac_model.decode([codes_l0, codes_l1, codes_l2])
            if waveform_tensor is None or waveform_tensor.numel() == 0: return
            if speed_adjustment != 1.0:
                waveform_tensor = _adjust_audio_speed(waveform_tensor, speed_adjustment)
            audio_data_np = waveform_tensor.squeeze().cpu().numpy()
            bytes_io = io.BytesIO()
            sf.write(bytes_io, audio_data_np, final_sample_rate, format='WAV', subtype=quality_settings["subtype"])
            yield bytes_io.getvalue()
        except Exception as e_chunk:
            print(f"Error processing chunk for streaming: {e_chunk}")

    try:
        with torch.inference_mode():
            generated_ids = orpheus_model.generate(
                input_ids, max_new_tokens=max_tokens, num_beams=1,
                do_sample=temperature > 0.001, temperature=temperature,
                repetition_penalty=repetition_penalty, top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=[end_of_speech_token, end_of_ai_token]
            )
        generated_sequence = generated_ids[0][input_ids.shape[1]:].tolist()

        for token_id in generated_sequence:
            if token_id == end_of_speech_token or token_id == end_of_ai_token: break
            if token_id >= AUDIO_CODE_BASE_OFFSET and token_id < (AUDIO_CODE_BASE_OFFSET + 4096 * 7):
                current_snac_chunk.append(token_id)
                if len(current_snac_chunk) >= chunk_size_tokens:
                    valid_len = (len(current_snac_chunk) // 7) * 7
                    if valid_len > 0:
                        for wav_chunk_bytes in process_and_yield_chunk(current_snac_chunk[:valid_len]):
                            yield wav_chunk_bytes
                    current_snac_chunk = current_snac_chunk[valid_len:]
        if current_snac_chunk:
             valid_len = (len(current_snac_chunk) // 7) * 7
             if valid_len > 0:
                for wav_chunk_bytes in process_and_yield_chunk(current_snac_chunk[:valid_len]):
                    yield wav_chunk_bytes
        print("Streaming generation completed.")
    except Exception as e:
        print(f"Error during streaming generation: {e}")
        yield f"Error: Streaming generation failed: {e}".encode('utf-8')

def init():
    print("--- INIT FUNCTION STARTED (Network Volume - DEBUGGING PATHS) ---")
    global orpheus_model, snac_model, tokenizer, device, FLASH_ATTN_AVAILABLE, MODELS_LOADED_SUCCESSFULLY
    MODELS_LOADED_SUCCESSFULLY = False

    # --- PATHS FOR MODELS ON THE NETWORK VOLUME (from your script) ---
    # This will use the environment variable if set, otherwise defaults to /runpod-volume
    NETWORK_VOLUME_MOUNT_PATH_FROM_ENV = os.getenv("RUNPOD_NETWORK_VOLUME_PATH", "/runpod-volume")
    ORPHEUS_EXPECTED_PATH = os.path.join(NETWORK_VOLUME_MOUNT_PATH_FROM_ENV, "orpheus_models")
    SNAC_EXPECTED_PATH = os.path.join(NETWORK_VOLUME_MOUNT_PATH_FROM_ENV, "snac_models")

    print(f"INIT_DEBUG: Value of NETWORK_VOLUME_MOUNT_PATH_FROM_ENV: {NETWORK_VOLUME_MOUNT_PATH_FROM_ENV}")
    print(f"INIT_DEBUG: Expecting Orpheus at: {ORPHEUS_EXPECTED_PATH}")
    print(f"INIT_DEBUG: Expecting SNAC at: {SNAC_EXPECTED_PATH}")

    print("\nINIT_DEBUG: Listing contents of common potential mount points:")
    common_paths_to_check = ["/", "/runpod-volume", "/workspace", "/mnt", "/data", NETWORK_VOLUME_MOUNT_PATH_FROM_ENV]
    # Add any other path you suspect or see in the UI

    for path_to_check in set(common_paths_to_check): # Use set to avoid duplicate listings
        print(f"\nINIT_DEBUG: Listing contents of '{path_to_check}':")
        if os.path.exists(path_to_check) and os.path.isdir(path_to_check):
            try:
                for item in os.listdir(path_to_check):
                    print(f"INIT_DEBUG:   {os.path.join(path_to_check, item)}")
            except Exception as e:
                print(f"INIT_DEBUG:   Could not list contents of {path_to_check}: {e}")
        else:
            print(f"INIT_DEBUG:   Path {path_to_check} does not exist or is not a directory.")

    print("\nINIT_DEBUG: Trying to list specific expected model directories:")
    for model_dir_path in [ORPHEUS_EXPECTED_PATH, SNAC_EXPECTED_PATH]:
        print(f"\nINIT_DEBUG: Listing contents of '{model_dir_path}':")
        if os.path.exists(model_dir_path) and os.path.isdir(model_dir_path):
            try:
                for item in os.listdir(model_dir_path):
                    print(f"INIT_DEBUG:   {os.path.join(model_dir_path, item)}")
            except Exception as e:
                print(f"INIT_DEBUG:   Could not list contents of {model_dir_path}: {e}")
        else:
            print(f"INIT_DEBUG:   Path {model_dir_path} does not exist or is not a directory.")

    # --- Now, proceed with your actual init logic using the determined paths ---
    # For this debug run, you might even comment out the actual model loading
    # if you just want to confirm paths first.
    # Or, let it run and see if the FileNotFoundError is resolved.

    # (Your actual model loading logic from the previous version of app_runpod.py for Network Volume)
    # Make sure ORPHEUS_CACHE_PATH and SNAC_CACHE_PATH use the NETWORK_VOLUME_MOUNT_PATH_FROM_ENV
    global ORPHEUS_CACHE_PATH, SNAC_CACHE_PATH # Ensure these are updated if not already global with this logic
    ORPHEUS_CACHE_PATH = ORPHEUS_EXPECTED_PATH
    SNAC_CACHE_PATH = SNAC_EXPECTED_PATH

    # (Paste your full model loading logic here, it will use the updated paths)
    # For brevity, I'm not pasting it all again, but it's the part that starts with:
    # print(f"INIT: Checking for Orpheus model at: {ORPHEUS_CACHE_PATH}")
    # and includes tokenizer and model loading.

    # Example of trying to load after path checks (simplified):
    if os.path.exists(ORPHEUS_CACHE_PATH) and os.path.isdir(ORPHEUS_CACHE_PATH) and os.listdir(ORPHEUS_CACHE_PATH):
        print(f"INIT_DEBUG: Attempting to load tokenizer from {ORPHEUS_CACHE_PATH}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(ORPHEUS_CACHE_PATH) # Example load
            print(f"INIT_DEBUG: Tokenizer loaded successfully from {ORPHEUS_CACHE_PATH}")
            MODELS_LOADED_SUCCESSFULLY = True # Set this if all critical parts load
        except Exception as e:
            print(f"INIT_DEBUG: FAILED to load tokenizer from {ORPHEUS_CACHE_PATH}: {e}")
            import traceback; traceback.print_exc()
            raise # Fail init
    else:
        print(f"INIT_DEBUG: Skipping model load as path {ORPHEUS_CACHE_PATH} seems invalid.")
        raise FileNotFoundError(f"Model path {ORPHEUS_CACHE_PATH} is invalid.")

    print("--- INIT FUNCTION FINISHED (Network Volume - DEBUGGING PATHS) ---")


def handler(event):
    global MODELS_LOADED_SUCCESSFULLY

    if not MODELS_LOADED_SUCCESSFULLY:
        print("ERROR: Handler called but models are not loaded. Init (Network Volume) might have failed.")
        return {"error": "Models are not loaded. Worker initialization failed.", "status_code": 503}

    print(f"Received event: {event}")
    job_input = event.get("input", {})

    if job_input.get("health_check"):
        return {
            "status": "healthy", "models_loaded": MODELS_LOADED_SUCCESSFULLY,
            "model_id": MODEL_ID, "snac_model_name": SNAC_MODEL_NAME,
            "flash_attention_available": FLASH_ATTN_AVAILABLE, "device": device,
            "orpheus_path_exists": os.path.exists(ORPHEUS_CACHE_PATH) and os.path.isdir(ORPHEUS_CACHE_PATH) and bool(os.listdir(ORPHEUS_CACHE_PATH)),
            "snac_path_exists": os.path.exists(SNAC_CACHE_PATH) and os.path.isdir(SNAC_CACHE_PATH) and bool(os.listdir(SNAC_CACHE_PATH))
        }
    try:
        request_data = TTSRequest(**job_input)
    except Exception as p_exc:
        print(f"Input validation error: {p_exc}")
        return {"error": f"Invalid input: {p_exc}", "status_code": 400}

    if request_data.stream:
        print("Starting streaming response...")
        return _generate_speech_streaming_internal(
            text_prompt=request_data.text, speaker_id=request_data.speaker,
            max_tokens=request_data.max_tokens, temperature=request_data.temperature,
            repetition_penalty=request_data.repetition_penalty, top_p=request_data.top_p,
            output_sample_rate=request_data.output_sample_rate, audio_quality=request_data.audio_quality,
            speed_adjustment=request_data.speed_adjustment, seed=request_data.seed
        )
    else:
        try:
            wav_bytes = _generate_speech_internal(
                text_prompt=request_data.text, speaker_id=request_data.speaker,
                max_tokens=request_data.max_tokens, temperature=request_data.temperature,
                repetition_penalty=request_data.repetition_penalty, top_p=request_data.top_p,
                output_sample_rate=request_data.output_sample_rate, audio_quality=request_data.audio_quality,
                speed_adjustment=request_data.speed_adjustment, seed=request_data.seed,
                early_stopping=request_data.early_stopping
            )
            audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')
            return {"audio_base64": audio_base64, "content_type": "audio/wav"}
        except ValueError as ve:
            print(f"Generation error: {ve}")
            return {"error": str(ve), "status_code": 500}
        except Exception as e:
            print(f"Unexpected error during generation: {e}")
            import traceback; traceback.print_exc()
            return {"error": "An unexpected error occurred during speech generation.", "status_code": 500}

if __name__ == "__main__":
    print("Starting Runpod serverless worker (Network Volume Version)...")
    runpod.serverless.start({
        "init": init,
        "handler": handler
    })