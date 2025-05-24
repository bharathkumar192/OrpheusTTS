import modal
import os
import io
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from snac import SNAC # Assuming the library is imported as 'snac' after installing 'snac-codec'
import soundfile as sf
import numpy as np
import time
import fastapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response 
from pydantic import BaseModel, Field, validator
from typing import Optional

# --- Configuration Constants ---
MODEL_ID = "bharathkumar1922001/10speaker-aws-9ksteps-7epochs-8B2G"
ORPHEUS_CACHE_PATH = "/model_cache/orpheus" # Internal cache path within the container
SNAC_CACHE_PATH = "/model_cache/snac"     # Internal cache path within the container

# Define available speakers for the model - Updated to support all 10 speakers
AVAILABLE_SPEAKERS = ["aisha", "anika", "arfa", "asmr", "nikita", "raju", "rhea", "ruhaan", "sangeeta", "shayana"]
DEFAULT_SPEAKER = "shayana"  # Set default speaker to one that exists in the model

SNAC_MODEL_NAME = "hubertsiuzdak/snac_24khz"
TARGET_AUDIO_SAMPLING_RATE = 24000

TOKENISER_LENGTH = 128256
START_OF_SPEECH_TOKEN = TOKENISER_LENGTH + 1
END_OF_SPEECH_TOKEN = TOKENISER_LENGTH + 2
START_OF_HUMAN_TOKEN = TOKENISER_LENGTH + 3
END_OF_HUMAN_TOKEN = TOKENISER_LENGTH + 4
START_OF_AI_TOKEN = TOKENISER_LENGTH + 5
END_OF_AI_TOKEN = TOKENISER_LENGTH + 6
AUDIO_CODE_BASE_OFFSET = TOKENISER_LENGTH + 10
PAD_TOKEN_ID_EXPECTED = 128263

SNAC_OFFSETS = [AUDIO_CODE_BASE_OFFSET + i * 4096 for i in range(7)]

# --- GPU Configuration ---
_GPU_TYPE_ENV = "H100" # Your hardcoded choice

if "H100" in _GPU_TYPE_ENV.upper():
    GPU_CONFIG = _GPU_TYPE_ENV
    TORCH_CUDA_ARCH_LIST_FOR_BUILD = "9.0" # Hopper
elif "A100" in _GPU_TYPE_ENV.upper(): # Catches A100-40GB and A100-80GB
    GPU_CONFIG = _GPU_TYPE_ENV
    TORCH_CUDA_ARCH_LIST_FOR_BUILD = "8.0" # Ampere (A100)
elif "A10G" in _GPU_TYPE_ENV.upper():
    GPU_CONFIG = _GPU_TYPE_ENV
    TORCH_CUDA_ARCH_LIST_FOR_BUILD = "8.6" # Ampere (A10G)
else: # Default or unknown, fallback to A10G
    GPU_CONFIG = "A10G"
    TORCH_CUDA_ARCH_LIST_FOR_BUILD = "8.6"
print(f"Using GPU: {GPU_CONFIG}, Target CUDA Arch for Flash-Attn build: {TORCH_CUDA_ARCH_LIST_FOR_BUILD}")

CONTAINER_TIMEOUT = int(os.environ.get("CONTAINER_TIMEOUT", "300")) # 5 minutes default

# --- Request and Response Models (Pydantic) ---
class TTSRequest(BaseModel):
    text: str = Field(..., description="The text to convert to speech")
    speaker_id: str = Field(default=DEFAULT_SPEAKER, description="The speaker voice ID to use")
    max_new_tokens: int = Field(default=2048, ge=100, le=4096, description="Maximum number of new (SNAC) tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Generation temperature. Use >0 for sampling.")
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0, description="Repetition penalty for generation.")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling parameter. Used if temperature > 0.")
    output_sample_rate: int = Field(default=TARGET_AUDIO_SAMPLING_RATE, description="Output audio sample rate (Hz).")
    audio_quality_preset: str = Field(default="high", description="Audio quality preset (low, medium, high).")
    speed_adjustment: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed adjustment factor.")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible generation.")
    early_stopping: bool = Field(default=True, description="Enable early stopping in generation.")
    
    class Config:
        json_schema_extra = {"example": {"text": "नमस्ते", "speaker_id": "shayana", "temperature": 0.7}}
        
    @validator('speaker_id')
    def validate_speaker(cls, v):
        if v.lower() not in AVAILABLE_SPEAKERS:
            raise ValueError(f"Speaker '{v}' not available. Choose from: {', '.join(AVAILABLE_SPEAKERS)}")
        return v.lower()  # Normalize to lowercase for consistency

AUDIO_QUALITY_PRESETS = {
    "low": {"target_sr": 16000, "subtype": "PCM_16"},
    "medium": {"target_sr": 24000, "subtype": "PCM_16"},
    "high": {"target_sr": TARGET_AUDIO_SAMPLING_RATE, "subtype": "PCM_16"},
}

# Ensure your secret is named "huggingface-secret" in Modal
hf_secret = modal.Secret.from_name("huggingface-secrets")
print(hf_secret)

def download_all_models():
    # This function runs during image build. HF_TOKEN is from the secret.
    hf_token_for_download = os.environ.get("HF_TOKEN")

    os.makedirs(ORPHEUS_CACHE_PATH, exist_ok=True)
    print(f"Downloading Orpheus model ({MODEL_ID}) to {ORPHEUS_CACHE_PATH}...")
    snapshot_download(
        repo_id=MODEL_ID, 
        local_dir=ORPHEUS_CACHE_PATH, 
        local_dir_use_symlinks=False, 
        token=hf_token_for_download
    )
    print("Orpheus model downloaded.")
    
    os.makedirs(SNAC_CACHE_PATH, exist_ok=True)
    print(f"Downloading SNAC model ({SNAC_MODEL_NAME}) to {SNAC_CACHE_PATH}...")
    snapshot_download(
        repo_id=SNAC_MODEL_NAME, 
        local_dir=SNAC_CACHE_PATH, 
        local_dir_use_symlinks=False, 
        token=hf_token_for_download # SNAC is public, but good practice
    )
    print("SNAC model downloaded.")

cuda_image_tag = "12.4.1-devel-ubuntu22.04" # Robust CUDA version for PyTorch & Flash Attn
orpheus_image = (
    modal.Image.from_registry(f"nvidia/cuda:{cuda_image_tag}", add_python="3.11")
    .apt_install("git")
    .pip_install(["ninja", "packaging", "wheel", "torch==2.3.1", "torchaudio==2.3.1"])
    .env({
        "TORCH_CUDA_ARCH_LIST": TORCH_CUDA_ARCH_LIST_FOR_BUILD, 
        "MAX_JOBS": str(os.cpu_count() or 4) # Use available CPUs for build
    })
    # Corrected flash-attn installation command
    .run_commands("pip install flash-attn==2.5.9.post1 --no-build-isolation")
    .pip_install([
        "transformers", "huggingface_hub", "hf_transfer",
        "snac",  # Corrected package name for SNAC
        "soundfile", "numpy", "scipy", "hf_xet",
        "fastapi", "uvicorn[standard]", "pydantic",
    ])
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}) # For snapshot_download
    .run_function(
        download_all_models, 
        secrets=[hf_secret], # Make secret available to download_all_models
        timeout=3600 # 30+30 minutes for model downloads
    )
)

app = modal.App("orpheus-tts-api-10speakers-v2") # Updated app name for 10-speaker support

@app.cls(
    gpu=GPU_CONFIG, 
    image=orpheus_image, 
    scaledown_window=CONTAINER_TIMEOUT,
    secrets=[hf_secret], # Make secret available to the class instance methods if needed (e.g. if SNAC re-downloads)
    max_containers=3, # Max concurrent requests; A100-80GB can handle more, adjust based on testing
)
class OrpheusTTSAPI:
    def __init__(self):
        self.web_app = fastapi.FastAPI(
            title="Orpheus TTS API - 10 Speakers (Refined V2)", 
            description="Text-to-Speech using Orpheus and SNAC with support for 10 Hindi speakers: aisha, anika, arfa, asmr, nikita, raju, rhea, ruhaan, sangeeta, shayana.",
            version="2.0.0"
        )
        self.web_app.add_middleware(
            CORSMiddleware, allow_origins=["*"], allow_credentials=True,
            allow_methods=["*"], allow_headers=["*"],
        )
        self._add_api_routes()

    def _add_api_routes(self):
        @self.web_app.post("/generate", response_class=Response)
        async def http_generate_speech(request: TTSRequest):
            try:
                wav_bytes, _ = self._perform_tts(request) # Call the internal method
                return Response(content=wav_bytes, media_type="audio/wav")
            except Exception as e:
                print(f"Error in /generate endpoint: {e}")
                import traceback; traceback.print_exc()
                # Return a JSON error response
                return Response(
                    content=f'{{"error": "TTS generation failed: {str(e)}"}}',
                    media_type="application/json", 
                    status_code=500
                )

        @self.web_app.get("/")
        async def root():
            return {
                "message": "Orpheus TTS API - 10 Speakers (Refined V2) is running.", 
                "model_id": MODEL_ID,
                "snac_model_id": SNAC_MODEL_NAME, 
                "gpu_config": GPU_CONFIG,
                "flash_attention_available": self.flash_attn_available, # Set in @modal.enter
                "available_speakers": AVAILABLE_SPEAKERS,
                "default_speaker": DEFAULT_SPEAKER,
                "total_speakers": len(AVAILABLE_SPEAKERS),
                "docs_url": "/docs" # FastAPI typically serves docs here
            }

        @self.web_app.get("/health")
        async def health_check():
            # Could add more sophisticated checks (e.g., model responsiveness)
            return {"status": "healthy", "timestamp": time.time()}

        @self.web_app.get("/speakers")
        async def get_speakers():
            """Get detailed information about available speakers"""
            return {
                "available_speakers": AVAILABLE_SPEAKERS,
                "default_speaker": DEFAULT_SPEAKER,
                "total_speakers": len(AVAILABLE_SPEAKERS),
                "speaker_descriptions": {
                    "aisha": "Girlfriend female speaker - warm and intimate voice",
                    "anika": "Social female speaker - energetic and friendly voice", 
                    "arfa": "Professional female speaker - confident and clear voice",
                    "asmr": "ASMR style speaker - soft and calming whisper voice",
                    "nikita": "Youthful female speaker - bright and cheerful voice",
                    "raju": "Relatable male speaker - casual and approachable voice",
                    "rhea": "Late-night girlfriend speaker - sultry and soothing voice", 
                    "ruhaan": "Mature male speaker - deep and authoritative voice",
                    "sangeeta": "Warm female speaker - gentle and nurturing voice",
                    "shayana": "Customer care female speaker - polite and helpful voice (default)"
                }
            }

    @modal.enter()
    def load_all_models_and_tokenizer(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- Initializing models on device: {self.device} ---")
        self.flash_attn_available = False # Default, will be updated

        # --- Debugging path existence (runs inside container after image build) ---
        print(f"Checking existence of ORPHEUS_CACHE_PATH: {ORPHEUS_CACHE_PATH}")
        if os.path.exists(ORPHEUS_CACHE_PATH) and os.path.isdir(ORPHEUS_CACHE_PATH):
            print(f"Contents of {ORPHEUS_CACHE_PATH}: {os.listdir(ORPHEUS_CACHE_PATH)}")
        else:
            # This would be a critical failure if models aren't baked into the image correctly.
            print(f"CRITICAL ERROR: {ORPHEUS_CACHE_PATH} does not exist or is not a directory!")
        
        print(f"Checking existence of SNAC_CACHE_PATH: {SNAC_CACHE_PATH}")
        if os.path.exists(SNAC_CACHE_PATH) and os.path.isdir(SNAC_CACHE_PATH):
            print(f"Contents of {SNAC_CACHE_PATH}: {os.listdir(SNAC_CACHE_PATH)}")
        else:
            print(f"CRITICAL ERROR: {SNAC_CACHE_PATH} does not exist or is not a directory!")
        # --- End debugging ---

        print(f"Loading Orpheus tokenizer from cached path: {ORPHEUS_CACHE_PATH}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            ORPHEUS_CACHE_PATH, 
            trust_remote_code=True,
            local_files_only=True # Explicitly load from local files
        )
        
        if self.tokenizer.pad_token_id is None:
            print(f"Warning: Tokenizer pad_token_id is None. Setting to expected: {PAD_TOKEN_ID_EXPECTED}")
            self.tokenizer.add_special_tokens({'pad_token': str(PAD_TOKEN_ID_EXPECTED)})
            self.tokenizer.pad_token_id = PAD_TOKEN_ID_EXPECTED
        elif self.tokenizer.pad_token_id != PAD_TOKEN_ID_EXPECTED:
             print(f"Warning: Loaded tokenizer pad_token_id ({self.tokenizer.pad_token_id}) "
                   f"differs from expected ({PAD_TOKEN_ID_EXPECTED}). Using loaded tokenizer's pad_token_id.")
        else:
            print(f"Tokenizer pad_token_id ({self.tokenizer.pad_token_id}) is correctly set.")

        print(f"Loading Orpheus model from cached path: {ORPHEUS_CACHE_PATH}")
        model_load_args = {
            "trust_remote_code": True, 
            "local_files_only": True # Explicitly load from local files
        }
        if self.device == "cuda":
            if torch.cuda.is_bf16_supported(): model_load_args["torch_dtype"] = torch.bfloat16
            else: model_load_args["torch_dtype"] = torch.float16
        
        pt_version = torch.__version__
        if tuple(map(int, pt_version.split('.')[:2])) >= (2, 0) and \
           self.device == "cuda" and torch.cuda.get_device_capability()[0] >= 8: # Ampere (8.0+) or Hopper (9.0)
            try:
                import flash_attn # Check if importable in the container environment
                model_load_args["attn_implementation"] = "flash_attention_2"
                self.flash_attn_available = True
                print("Attempting to use Flash Attention 2 for Orpheus model.")
            except ImportError: 
                self.flash_attn_available = False # Ensure it's set if import fails
                print("Flash Attention 2 import failed, using default attention.")
        else: 
            self.flash_attn_available = False
            print("Flash Attention 2 not used (PyTorch/CUDA/GPU requirements not met).")

        self.orpheus_model = AutoModelForCausalLM.from_pretrained(ORPHEUS_CACHE_PATH, **model_load_args)
        
        model_vocab_size = self.orpheus_model.get_input_embeddings().weight.shape[0]
        tokenizer_vocab_size = len(self.tokenizer)
        if tokenizer_vocab_size != model_vocab_size:
            print(f"CRITICAL MISMATCH: Tokenizer vocab ({tokenizer_vocab_size}) vs Model vocab ({model_vocab_size})!")
            # This is a serious issue with the model assets. Do not resize in inference.
        
        self.orpheus_model.to(self.device); self.orpheus_model.eval()
        print("Orpheus model loaded.")

        print(f"Loading SNAC model ({SNAC_MODEL_NAME}) from {SNAC_CACHE_PATH}...")
        try:
            # SNAC library might not support local_files_only; it usually handles local paths well.
            self.snac_model = SNAC.from_pretrained(SNAC_CACHE_PATH) 
        except Exception as e:
            # Fallback if loading from explicit cache path fails (e.g., if SNAC lib expects a specific structure or Hub ID)
            print(f"Failed to load SNAC from {SNAC_CACHE_PATH} directly ({e}). "
                  f"Attempting to load {SNAC_MODEL_NAME} (may trigger download if not in SNAC's internal cache).")
            self.snac_model = SNAC.from_pretrained(SNAC_MODEL_NAME)

        self.snac_model.to(self.device); self.snac_model.eval()
        print("SNAC model loaded.")
        print("--- All models loaded successfully. ---")

    def _deinterleave_snac_codes(self, snac_codes_flat: list[int]) -> tuple[Optional[torch.Tensor], ...]:
        if not snac_codes_flat or len(snac_codes_flat) % 7 != 0: return None, None, None
        num_frames = len(snac_codes_flat) // 7
        # Initialize lists more concisely
        codes_lvl0, codes_lvl1, codes_lvl2 = ([] for _ in range(3))
        for i in range(0, len(snac_codes_flat), 7):
            codes_lvl0.append(snac_codes_flat[i]   - SNAC_OFFSETS[0])
            codes_lvl1.append(snac_codes_flat[i+1] - SNAC_OFFSETS[1]) # L1a
            codes_lvl2.append(snac_codes_flat[i+2] - SNAC_OFFSETS[2]) # L2a
            codes_lvl2.append(snac_codes_flat[i+3] - SNAC_OFFSETS[3]) # L2b
            codes_lvl1.append(snac_codes_flat[i+4] - SNAC_OFFSETS[4]) # L1b
            codes_lvl2.append(snac_codes_flat[i+5] - SNAC_OFFSETS[5]) # L2c
            codes_lvl2.append(snac_codes_flat[i+6] - SNAC_OFFSETS[6]) # L2d
        try:
            t_kwargs = {"dtype": torch.long, "device": self.device}
            tensors = [torch.tensor(l, **t_kwargs).unsqueeze(0) for l in [codes_lvl0, codes_lvl1, codes_lvl2]]
            if not (tensors[0].shape[1] == num_frames and \
                    tensors[1].shape[1] == 2 * num_frames and \
                    tensors[2].shape[1] == 4 * num_frames):
                raise ValueError("Shape mismatch after tensor creation in deinterleave.")
            return tuple(tensors)
        except Exception as e: 
            print(f"Error in _deinterleave_snac_codes: {e}"); return None, None, None

    def _adjust_audio_speed(self, waveform_tensor: torch.Tensor, speed_factor: float) -> torch.Tensor:
        if abs(speed_factor - 1.0) < 1e-3: return waveform_tensor # No change needed
        
        original_sr = TARGET_AUDIO_SAMPLING_RATE
        # Ensure waveform is mono and 2D [1, T] for torchaudio.functional.speed
        if waveform_tensor.ndim == 1: waveform_tensor = waveform_tensor.unsqueeze(0)
        if waveform_tensor.shape[0] != 1: waveform_tensor = torch.mean(waveform_tensor, dim=0, keepdim=True)
        
        try:
            import torchaudio.functional as AF
            # torchaudio.functional.speed returns (Tensor waveform, int sample_rate)
            # The output sample rate will be original_sr / speed_factor. We want to maintain original_sr.
            # So, we use it to change speed, then resample back if needed.
            # However, for direct speed adjustment, it modifies the length.
            resampled_waveform, _ = AF.speed(waveform_tensor.squeeze(0), sample_rate=original_sr, factor=speed_factor)
            return resampled_waveform.unsqueeze(0) # Add channel dim back
        except Exception as e:
            print(f"Torchaudio speed adjustment failed ({e}), falling back to basic interpolation (less robust).")
            orig_length = waveform_tensor.shape[1]
            new_length = int(orig_length / speed_factor)
            if new_length == 0: return torch.empty((1,0), device=self.device) # Avoid div by zero

            indices = torch.linspace(0, orig_length - 1, new_length, device=self.device)
            indices_floor = torch.clamp(indices.long(), 0, orig_length - 1) # Clamp to avoid out-of-bounds
            indices_ceil = torch.clamp(indices_floor + 1, 0, orig_length - 1) # Clamp to avoid out-of-bounds
            
            weight_ceil = indices - indices_floor.float(); weight_floor = 1.0 - weight_ceil
            # Ensure weights are applied to correct waveform parts (using broadcasting)
            adjusted_waveform = waveform_tensor[:, indices_floor] * weight_floor + \
                                waveform_tensor[:, indices_ceil] * weight_ceil
            return adjusted_waveform

    def _perform_tts(self, request: TTSRequest) -> tuple[bytes, float]:
        request_start_time = time.time()
        if request.seed is not None: torch.manual_seed(request.seed); np.random.seed(request.seed)

        full_prompt = f"{request.speaker_id}: {request.text}"
        try:
            input_ids = [START_OF_HUMAN_TOKEN] + \
                        self.tokenizer.encode(full_prompt, add_special_tokens=False) + \
                        [END_OF_HUMAN_TOKEN, START_OF_AI_TOKEN, START_OF_SPEECH_TOKEN]
            input_ids_t = torch.tensor([input_ids], device=self.device)
        except Exception as e: raise RuntimeError(f"Tokenization failed: {e}")

        try:
            with torch.inference_mode():
                generation_params = {
                    "input_ids": input_ids_t, "max_new_tokens": request.max_new_tokens, "num_beams": 1,
                    "do_sample": request.temperature > 0.001 and request.temperature <=1.0, # Be precise with sampling condition
                    "repetition_penalty": request.repetition_penalty,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": [END_OF_SPEECH_TOKEN, END_OF_AI_TOKEN], # List of EOS tokens
                    "early_stopping": request.early_stopping
                }
                if generation_params["do_sample"]:
                    generation_params["temperature"] = max(0.01, request.temperature) # Ensure temp is positive if sampling
                    generation_params["top_p"] = request.top_p
                else: # remove sampling-specific keys if not sampling
                    generation_params.pop("temperature", None)
                    generation_params.pop("top_p", None)
                    
                generated_ids = self.orpheus_model.generate(**generation_params)
            
            # Extract SNAC codes robustly
            generated_part = generated_ids[0][input_ids_t.shape[1]:].tolist()
            snac_codes = []
            for token_val in generated_part:
                if token_val == END_OF_SPEECH_TOKEN or token_val == END_OF_AI_TOKEN: break
                if AUDIO_CODE_BASE_OFFSET <= token_val < (AUDIO_CODE_BASE_OFFSET + 4096 * 7):
                    snac_codes.append(token_val)

            if not snac_codes: raise RuntimeError("No SNAC codes generated.")
            # Truncate to be multiple of 7
            if len(snac_codes) % 7 != 0:
                snac_codes = snac_codes[:-(len(snac_codes) % 7)]
            if not snac_codes: raise RuntimeError("No valid SNAC codes after truncation.")
        except Exception as e: raise RuntimeError(f"Orpheus model generation failed: {e}")

        codes_l0, codes_l1, codes_l2 = self._deinterleave_snac_codes(snac_codes)
        if codes_l0 is None: raise RuntimeError("Failed to deinterleave SNAC codes.")

        try:
            with torch.inference_mode():
                waveform_t = self.snac_model.decode([codes_l0, codes_l1, codes_l2])
            if waveform_t is None or waveform_t.numel() == 0: raise RuntimeError("SNAC decoding resulted in an empty waveform.")
            
            if abs(request.speed_adjustment - 1.0) > 1e-3: # Apply speed adjustment if not 1.0
                waveform_t = self._adjust_audio_speed(waveform_t, request.speed_adjustment)

            audio_np = waveform_t.squeeze().cpu().numpy()
            
            quality_cfg = AUDIO_QUALITY_PRESETS.get(request.audio_quality_preset, AUDIO_QUALITY_PRESETS["high"])
            final_sr_to_use = request.output_sample_rate 
            
            # Resample only if the target sample rate is different from the *current* sample rate of audio_np
            # The _adjust_audio_speed using torchaudio.functional.speed might change the sample rate.
            # For simplicity, let's assume _adjust_audio_speed outputs at TARGET_AUDIO_SAMPLING_RATE
            # (or we resample after it if it doesn't).
            # If AF.speed was used, current SR is TARGET_AUDIO_SAMPLING_RATE (because we didn't use its output SR).
            # If interpolation was used, current SR is also TARGET_AUDIO_SAMPLING_RATE.
            current_audio_sr = TARGET_AUDIO_SAMPLING_RATE 

            if final_sr_to_use != current_audio_sr:
                print(f"Resampling audio from {current_audio_sr}Hz to {final_sr_to_use}Hz.")
                try:
                    import torchaudio.transforms as AT
                    # Ensure audio_np is on the correct device for resampling if needed by AT.Resample
                    temp_tensor_for_resample = torch.from_numpy(audio_np).to(self.device).unsqueeze(0)
                    resampler = AT.Resample(orig_freq=current_audio_sr, new_freq=final_sr_to_use).to(self.device)
                    resampled_audio_tensor = resampler(temp_tensor_for_resample)
                    audio_np = resampled_audio_tensor.squeeze().cpu().numpy()
                except Exception as resample_e:
                    print(f"Warning: Could not resample audio with torchaudio: {resample_e}. Outputting at {current_audio_sr}Hz.")
                    final_sr_to_use = current_audio_sr # Fallback to current SR

            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio_np, final_sr_to_use, format='WAV', subtype=quality_cfg["subtype"])
            return wav_buffer.getvalue(), time.time() - request_start_time
        except Exception as e: raise RuntimeError(f"SNAC decoding or audio post-processing failed: {e}")

    @modal.exit()
    def cleanup(self):
        print(f"{self.__class__.__name__} (GPU: {GPU_CONFIG}) container shutting down.")
        # Explicitly delete models to help free GPU memory
        if hasattr(self, 'orpheus_model'): del self.orpheus_model
        if hasattr(self, 'snac_model'): del self.snac_model
        if hasattr(self, 'tokenizer'): del self.tokenizer
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # This makes the FastAPI app accessible via Modal's web serving.
    @modal.asgi_app()
    def run_fastapi_app(self): return self.web_app

# --- Local CLI for Testing (runs on Modal infrastructure) ---
@app.local_entrypoint()
def cli_test_generation(
    prompt: str = "नमस्ते, आपका दिन कैसा रहा?", 
    speaker: str = DEFAULT_SPEAKER, # Use DEFAULT_SPEAKER constant
    output_file: str = "cli_test_v2_output.wav" # Default output filename
):
    # This directory will be created on your *local* machine
    output_dir_local = "modal_test_outputs_refined_v2" 
    os.makedirs(output_dir_local, exist_ok=True)
    
    print(f"--- Running CLI Test on Modal for {app.name} ---")
    print(f"Prompt: '{prompt}', Speaker: '{speaker}'")

    request_data = TTSRequest(text=prompt, speaker_id=speaker) # Using default params from TTSRequest

    try:
        # Instantiate the Modal class (this will trigger @modal.enter on Modal's side)
        service_instance = OrpheusTTSAPI()
        
        # Call the internal TTS method. .remote() executes it on Modal.
        audio_bytes, duration_secs = service_instance._perform_tts.remote(request_data)
        
        full_output_path = os.path.join(output_dir_local, output_file)
        with open(full_output_path, "wb") as f:
            f.write(audio_bytes)
        
        print(f"SUCCESS: CLI test audio saved to local file '{full_output_path}'")
        print(f"Generation duration on Modal: {duration_secs:.2f}s")

    except Exception as e:
        print(f"CLI Test FAILED: {e}")
        import traceback
        traceback.print_exc()