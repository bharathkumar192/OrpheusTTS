import modal
import os
import logging
import warnings

# Suppress redundant CUDA banners and verbose logging before importing torch/vLLM
os.environ["PYTORCH_CUDA_ERROR_REPORTING"] = "0"
os.environ["VLLM_ENGINE_ITERATION_TIMEOUT_S"] = "120"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Configure logging to reduce noise
for logger_name in ("torch", "vllm", "transformers"):
    logging.getLogger(logger_name).setLevel(logging.WARNING)
warnings.filterwarnings('ignore', category=UserWarning)

import torch
import time
import fastapi
import asyncio
import struct
import json
from typing import Optional, AsyncGenerator, List, Dict, Any
from pydantic import BaseModel, Field, validator
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response, JSONResponse
import numpy as np

# ===== CONFIGURATION CONSTANTS =====
MODEL_ID = "bharathkumar1922001/10speaker-aws-9ksteps-7epochs-8B2G"
VEENA_CACHE_PATH = "/persistent_cache/veena"
SNAC_CACHE_PATH = "/persistent_cache/snac"
AVAILABLE_SPEAKERS = [
    "aisha", "anika", "arfa", "asmr", "nikita", "raju", "rhea",
    "ruhaan", "sangeeta", "shayana",
]
DEFAULT_SPEAKER = "shayana"

# Speaker mapping: Customer-facing name -> Internal model name
SPEAKER_MAPPING = {
    "charu_soft": "aisha",
    "ishana_spark": "anika", 
    "kyra_prime": "arfa",
    "mohini_whispers": "asmr",
    "keerti_joy": "nikita",
    "varun_chat": "raju",
    "soumya_calm": "rhea",
    "agastya_impact": "ruhaan",
    "maitri_connect": "sangeeta",
    "vinaya_assist": "shayana",
}

# Load speaker details from JSON file
def load_speaker_details():
    # Try multiple locations for the speakers.json file
    possible_paths = [
        "/app/speakers.json",  # Modal container location
        os.path.join(os.path.dirname(__file__), "speakers.json"),  # Local development
        "speakers.json"  # Current directory fallback
    ]
    
    for speakers_file_path in possible_paths:
        try:
            with open(speakers_file_path, "r", encoding="utf-8") as f:
                print(f"Loaded speaker details from: {speakers_file_path}")
                return json.load(f)
        except FileNotFoundError:
            continue
        except json.JSONDecodeError as e:
            print(f"Error parsing speakers.json at {speakers_file_path}: {e}")
            raise
    
    # If none found, use fallback
    print("Warning: speakers.json not found in any location, using fallback speaker configuration")
    return {
        "vinaya_assist": {
            "id": "vinaya_assist",
            "name": "Vinaya Assist",
            "description": "Assistant & Helpful",
            "gender": "female",
            "language": "hindi",
            "use_cases": ["virtual_assistant", "tutorials", "instructions"],
            "voice_characteristics": ["helpful", "clear", "instructional"]
        }
    }

# Customer-facing speaker details loaded from JSON
SPEAKER_DETAILS = load_speaker_details()

DEFAULT_CUSTOMER_SPEAKER = "vinaya_assist"
DEFAULT_INTERNAL_SPEAKER = SPEAKER_MAPPING[DEFAULT_CUSTOMER_SPEAKER]

# Reverse mapping: Internal model name -> Customer-facing name (for efficiency)
INTERNAL_TO_CUSTOMER_MAPPING = {v: k for k, v in SPEAKER_MAPPING.items()}

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
PAD_TOKEN_ID = 128263

# Streaming window configuration
SNAC_WINDOW_SIZE_TOKENS = 28
SNAC_HOP_SIZE_TOKENS = 7
NUM_LLM_FRAMES_PER_DECODE_CHUNK = 4

GPU_CONFIG = "H100"
TORCH_CUDA_ARCH_LIST_FOR_BUILD = "9.0"
CONTAINER_TIMEOUT = 600
VLLM_INSTALL_VERSION = "0.8.5"

# Analytics configuration
ENABLE_ANALYTICS = True
ANALYTICS_OVERHEAD_THRESHOLD_MS = 1.0


# ===== REQUEST MODEL =====
class TTSRequest(BaseModel):
    text: str = Field(..., description="The text to convert to speech")
    speaker_id: str = Field(default=DEFAULT_CUSTOMER_SPEAKER, description="The speaker voice ID to use", json_schema_extra={"enum": list(SPEAKER_MAPPING.keys())})
    max_new_tokens: int = Field(default=1536, ge=100, le=3072)
    temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    repetition_penalty: float = Field(default=1.05, ge=1.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    streaming: bool = Field(default=True, description="Enable streaming response")
    seed: Optional[int] = Field(default=None)

    @validator("speaker_id")
    def validate_speaker(cls, v):
        speaker_key = v.lower()
        
        # Check if it's a valid customer-facing speaker ID
        if speaker_key in SPEAKER_MAPPING:
            return speaker_key
            
        # For backward compatibility, also accept internal speaker names
        if speaker_key in INTERNAL_TO_CUSTOMER_MAPPING:
            return INTERNAL_TO_CUSTOMER_MAPPING[speaker_key]
            
        # Neither customer nor internal speaker ID found
        available_speakers = list(SPEAKER_MAPPING.keys())
        raise ValueError(
            f"Speaker '{v}' not available. Choose from: {', '.join(available_speakers)}"
        )


# ===== ANALYTICS CLASS =====
class GenerationAnalytics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.start_time = time.time()
        self.first_token_time = None
        self.first_audio_time = None
        self.end_time = None
        self.total_tokens = 0
        self.snac_tokens = 0
        self.audio_chunks = 0
        self.snac_decode_time = 0.0
        self.llm_generation_time = 0.0
    
    def mark_first_token(self):
        if self.first_token_time is None:
            self.first_token_time = time.time()
    
    def mark_first_audio(self):
        if self.first_audio_time is None:
            self.first_audio_time = time.time()
    
    def mark_end(self):
        self.end_time = time.time()
    
    def add_snac_decode_time(self, duration: float):
        self.snac_decode_time += duration
    
    def get_metrics(self) -> Dict[str, Any]:
        total_time = (self.end_time or time.time()) - self.start_time
        ttft = (self.first_token_time - self.start_time) if self.first_token_time else None
        ttfa = (self.first_audio_time - self.start_time) if self.first_audio_time else None
        
        return {
            "total_time_s": round(total_time, 3),
            "ttft_s": round(ttft, 3) if ttft else None,
            "ttfa_s": round(ttfa, 3) if ttfa else None,
            "tokens_per_second": round(self.total_tokens / total_time, 2) if total_time > 0 else 0,
            "snac_tokens": self.snac_tokens,
            "audio_chunks": self.audio_chunks,
            "snac_decode_time_s": round(self.snac_decode_time, 3),
            "snac_decode_overhead_pct": round((self.snac_decode_time / total_time) * 100, 1) if total_time > 0 else 0
        }


# ===== IMAGE DEFINITION =====
hf_secret = modal.Secret.from_name("huggingface-secrets")

def download_models():
    from huggingface_hub import snapshot_download
    hf_token = os.environ.get("HF_TOKEN")
    
    # Download to temporary paths during image build
    # These will be used as fallback if persistent cache is empty
    temp_veena_path = "/tmp/veena_models"
    temp_snac_path = "/tmp/snac_models"
    
    print(f"Downloading Veena model to {temp_veena_path}...")
    os.makedirs(temp_veena_path, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=temp_veena_path,
        local_dir_use_symlinks=False,
        token=hf_token,
        resume_download=True,
    )
    
    print(f"Downloading SNAC model to {temp_snac_path}...")
    os.makedirs(temp_snac_path, exist_ok=True)
    snapshot_download(
        repo_id=SNAC_MODEL_NAME,
        local_dir=temp_snac_path,
        local_dir_use_symlinks=False,
        token=hf_token,
        resume_download=True,
    )
    
    print("Model downloads to temporary paths complete!")


cuda_image_tag = "12.4.1-devel-ubuntu22.04"

# Only copy speakers.json if it exists
if os.path.exists("speakers.json"):
    veena_image = (
        modal.Image.from_registry(f"nvidia/cuda:{cuda_image_tag}", add_python="3.11")
        .apt_install("git")
        .pip_install(["ninja", "packaging", "wheel"])
        .copy_local_file("speakers.json", "/app/speakers.json")
        .env(
            {
                "TORCH_CUDA_ARCH_LIST": TORCH_CUDA_ARCH_LIST_FOR_BUILD,
                "MAX_JOBS": str(os.cpu_count() or 4),
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "CUDA_LAUNCH_BLOCKING": "0",
                "PYTHONUNBUFFERED": "1",
            }
        )
        .run_commands(
            "pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
            f"pip3 install --upgrade 'vllm=={VLLM_INSTALL_VERSION}'",
            "pip3 install flash-attn --no-build-isolation || echo 'Flash attention optional, continuing...'",
            "pip3 install flashinfer -i https://flashinfer.ai/whl/cu121 || echo 'FlashInfer optional, continuing...'",
        )
        .pip_install(
            [
                "transformers==4.52.3",
                "tokenizers==0.21.1",
                "accelerate",
                "snac",
                "huggingface_hub",
                "hf_transfer",
                "soundfile",
                "numpy",
                "scipy",
                "fastapi",
                "uvicorn[standard]",
                "pydantic>=2.9,<3.0",
                "xformers",
                "ray",
                "sentencepiece",
                "protobuf",
            ]
        )
        .run_function(download_models, secrets=[hf_secret], timeout=3600)
    )
else:
    veena_image = (
        modal.Image.from_registry(f"nvidia/cuda:{cuda_image_tag}", add_python="3.11")
        .apt_install("git")
        .pip_install(["ninja", "packaging", "wheel"])
        .env(
            {
                "TORCH_CUDA_ARCH_LIST": TORCH_CUDA_ARCH_LIST_FOR_BUILD,
                "MAX_JOBS": str(os.cpu_count() or 4),
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "CUDA_LAUNCH_BLOCKING": "0",
                "PYTHONUNBUFFERED": "1",
            }
        )
        .run_commands(
            "pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
            f"pip3 install --upgrade 'vllm=={VLLM_INSTALL_VERSION}'",
            "pip3 install flash-attn --no-build-isolation || echo 'Flash attention optional, continuing...'",
            "pip3 install flashinfer -i https://flashinfer.ai/whl/cu121 || echo 'FlashInfer optional, continuing...'",
        )
        .pip_install(
            [
                "transformers==4.52.3",
                "tokenizers==0.21.1",
                "accelerate",
                "snac",
                "huggingface_hub",
                "hf_transfer",
                "soundfile",
                "numpy",
                "scipy",
                "fastapi",
                "uvicorn[standard]",
                "pydantic>=2.9,<3.0",
                "xformers",
                "ray",
                "sentencepiece",
                "protobuf",
            ]
        )
        .run_function(download_models, secrets=[hf_secret], timeout=3600)
    )


app = modal.App("Veena")
model_volume = modal.Volume.from_name("veena-model-cache", create_if_missing=True)

# ===== SNAC PROCESSOR CLASS =====
class SNACProcessor:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.num_levels = 3
        self.llm_codebook_base_offsets = [AUDIO_CODE_BASE_OFFSET + i * 4096 for i in range(7)]

    def load(self):
        from snac import SNAC
        print("Loading SNAC model...")
        
        # Try loading from persistent cache first
        if os.path.exists(SNAC_CACHE_PATH) and os.listdir(SNAC_CACHE_PATH):
            try:
                print(f"Loading SNAC from persistent cache: {SNAC_CACHE_PATH}")
                self.model = SNAC.from_pretrained(SNAC_CACHE_PATH).eval().to(self.device)
                self._post_load_setup()
                return
            except Exception as e:
                print(f"Failed to load from persistent cache: {e}")
        
        # Try loading from temp path (image build cache)
        temp_snac_path = "/tmp/snac_models"
        if os.path.exists(temp_snac_path) and os.listdir(temp_snac_path):
            try:
                print(f"Loading SNAC from temp cache: {temp_snac_path}")
                self.model = SNAC.from_pretrained(temp_snac_path).eval().to(self.device)
                # Copy to persistent cache for future use
                self._copy_to_persistent_cache(temp_snac_path, SNAC_CACHE_PATH)
                self._post_load_setup()
                return
            except Exception as e:
                print(f"Failed to load from temp cache: {e}")
        
        # Fallback to downloading from HuggingFace
        print(f"Downloading SNAC from HuggingFace: {SNAC_MODEL_NAME}")
        self.model = SNAC.from_pretrained(SNAC_MODEL_NAME).eval().to(self.device)
        self._post_load_setup()

    def _post_load_setup(self):
        """Setup after model is loaded"""
        # Dynamically determine num_levels
        if hasattr(self.model, 'quantizer') and hasattr(self.model.quantizer, 'layers'):
            self.num_levels = len(self.model.quantizer.layers)
        elif hasattr(self.model, 'quantizer') and hasattr(self.model.quantizer, 'n_q'):
            self.num_levels = self.model.quantizer.n_q
        
        # Set up level strides
        if self.num_levels == 3:
            self.level_strides = [4, 2, 1]
        elif self.num_levels == 4:
            self.level_strides = [8, 4, 2, 1]
        else:
            self.level_strides = [2 ** (self.num_levels - 1 - i) for i in range(self.num_levels)]
        
        # Compile decoder for performance
        try:
            if hasattr(torch, "compile") and self.device == "cuda":
                if tuple(map(int, torch.__version__.split(".")[:2])) >= (2, 0):
                    self.model.decoder = torch.compile(self.model.decoder, dynamic=True)
                    print("SNAC decoder compiled with torch.compile")
        except Exception:
            pass
        
        self._warmup()
        print("SNAC model ready")

    def _copy_to_persistent_cache(self, src_path: str, dst_path: str):
        """Copy model from temp cache to persistent cache"""
        try:
            import shutil
            os.makedirs(dst_path, exist_ok=True)
            if os.path.exists(dst_path) and os.listdir(dst_path):
                print(f"Persistent cache already exists at {dst_path}")
                return
            print(f"Copying model from {src_path} to {dst_path}")
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            print(f"Successfully copied to persistent cache: {dst_path}")
        except Exception as e:
            print(f"Failed to copy to persistent cache: {e}")

    def _warmup(self):
        if not self.model:
            return
        with torch.inference_mode():
            dummy_data_for_snac_levels = [
                torch.randint(0, 4096, (1, 4), device=self.device, dtype=torch.int32),
                torch.randint(0, 4096, (1, 8), device=self.device, dtype=torch.int32),
                torch.randint(0, 4096, (1, 16), device=self.device, dtype=torch.int32)
            ]
            if self.num_levels != 3:
                dummy_data_for_snac_levels = [
                    torch.randint(0, 4096, (1, 4 * (2**i)), device=self.device, dtype=torch.int32) 
                    for i in range(self.num_levels)
                ]
            try:
                _ = self.model.decode(dummy_data_for_snac_levels[:self.num_levels])
            except Exception:
                pass

    def decode_chunk(self, snac_tokens_global_ids: List[int]) -> bytes:
        if not self.model or not snac_tokens_global_ids: 
            return b""
        
        # Ensure multiple of 7 for LLM frames
        if len(snac_tokens_global_ids) % 7 != 0:
            valid_len = (len(snac_tokens_global_ids) // 7) * 7
            if valid_len == 0: 
                return b""
            snac_tokens_global_ids = snac_tokens_global_ids[:valid_len]

        num_coarse_frames = len(snac_tokens_global_ids) // 7
        if num_coarse_frames == 0: 
            return b""

        codes_lvl0_local, codes_lvl1_local, codes_lvl2_local = [], [], []
        for i in range(0, len(snac_tokens_global_ids), 7):
            codes_lvl0_local.append(snac_tokens_global_ids[i]   - self.llm_codebook_base_offsets[0])
            codes_lvl1_local.append(snac_tokens_global_ids[i+1] - self.llm_codebook_base_offsets[1])
            codes_lvl1_local.append(snac_tokens_global_ids[i+4] - self.llm_codebook_base_offsets[4])
            codes_lvl2_local.append(snac_tokens_global_ids[i+2] - self.llm_codebook_base_offsets[2])
            codes_lvl2_local.append(snac_tokens_global_ids[i+3] - self.llm_codebook_base_offsets[3])
            codes_lvl2_local.append(snac_tokens_global_ids[i+5] - self.llm_codebook_base_offsets[5])
            codes_lvl2_local.append(snac_tokens_global_ids[i+6] - self.llm_codebook_base_offsets[6])
        
        hier_for_snac_decode_direct = [
            torch.tensor(codes_lvl0_local, dtype=torch.int32, device=self.device).unsqueeze(0),
            torch.tensor(codes_lvl1_local, dtype=torch.int32, device=self.device).unsqueeze(0),
            torch.tensor(codes_lvl2_local, dtype=torch.int32, device=self.device).unsqueeze(0)
        ]

        # Validate token ranges
        for lvl_tensor in hier_for_snac_decode_direct:
            if torch.any((lvl_tensor < 0) | (lvl_tensor > 4095)):
                return b""

        with torch.inference_mode():
            audio_hat = self.model.decode(hier_for_snac_decode_direct)
        
        if audio_hat is None or audio_hat.numel() == 0:
            return b""

        audio_data_np = audio_hat.squeeze().clamp(-1, 1).cpu().numpy()
        return (audio_data_np * 32767).astype(np.int16).tobytes()

    def decode_window_and_get_hop_slice(self, snac_tokens_global_ids: List[int]) -> bytes:
        """
        Decode a window of SNAC tokens and return only the hop slice to avoid overlap artifacts.
        """
        if not self.model or not snac_tokens_global_ids:
            return b""
            
        # For streaming, we expect exactly SNAC_WINDOW_SIZE_TOKENS
        if len(snac_tokens_global_ids) != SNAC_WINDOW_SIZE_TOKENS:
            return self.decode_chunk(snac_tokens_global_ids)
        
        # Deinterleave the tokens
        num_coarse_frames = len(snac_tokens_global_ids) // 7
        codes_lvl0_local, codes_lvl1_local, codes_lvl2_local = [], [], []
        
        for i in range(0, len(snac_tokens_global_ids), 7):
            codes_lvl0_local.append(snac_tokens_global_ids[i]   - self.llm_codebook_base_offsets[0])
            codes_lvl1_local.append(snac_tokens_global_ids[i+1] - self.llm_codebook_base_offsets[1])
            codes_lvl1_local.append(snac_tokens_global_ids[i+4] - self.llm_codebook_base_offsets[4])
            codes_lvl2_local.append(snac_tokens_global_ids[i+2] - self.llm_codebook_base_offsets[2])
            codes_lvl2_local.append(snac_tokens_global_ids[i+3] - self.llm_codebook_base_offsets[3])
            codes_lvl2_local.append(snac_tokens_global_ids[i+5] - self.llm_codebook_base_offsets[5])
            codes_lvl2_local.append(snac_tokens_global_ids[i+6] - self.llm_codebook_base_offsets[6])
        
        hier_for_snac_decode_direct = [
            torch.tensor(codes_lvl0_local, dtype=torch.int32, device=self.device).unsqueeze(0),
            torch.tensor(codes_lvl1_local, dtype=torch.int32, device=self.device).unsqueeze(0),
            torch.tensor(codes_lvl2_local, dtype=torch.int32, device=self.device).unsqueeze(0)
        ]
        
        # Validate token ranges
        for lvl_tensor in hier_for_snac_decode_direct:
            if torch.any((lvl_tensor < 0) | (lvl_tensor > 4095)):
                return b""
        
        with torch.inference_mode():
            audio_hat = self.model.decode(hier_for_snac_decode_direct)
        
        if audio_hat is None or audio_hat.numel() == 0:
            return b""
        
        # Extract second frame audio to avoid overlap artifacts
        if num_coarse_frames == NUM_LLM_FRAMES_PER_DECODE_CHUNK and audio_hat.shape[-1] >= 4096:
            audio_slice_for_yield = audio_hat[:, :, 2048:4096]
        else:
            # Fallback: proportional slice
            samples_per_frame = audio_hat.shape[-1] // num_coarse_frames
            hop_frames = SNAC_HOP_SIZE_TOKENS // 7
            start_idx = samples_per_frame
            end_idx = start_idx + (samples_per_frame * hop_frames)
            audio_slice_for_yield = audio_hat[:, :, start_idx:end_idx]
        
        audio_data_np = audio_slice_for_yield.squeeze().clamp(-1, 1).cpu().numpy()
        return (audio_data_np * 32767).astype(np.int16).tobytes()


# ===== MAIN TTS API CLASS =====
@app.cls(
    gpu=GPU_CONFIG,
    image=veena_image,
    scaledown_window=CONTAINER_TIMEOUT,
    secrets=[hf_secret],
    max_containers=5,
    volumes={"/persistent_cache": model_volume},
)
class VeenaTTSAPI:
    @modal.enter()
    async def initialize(self):
        # Initialize warmup state management
        self._warmup_lock = asyncio.Lock()
        self._ready_event = asyncio.Event()  # initially not set
        self._failed_event = asyncio.Event()  # initially not set
        self._warmup_started = False
        self._warmup_error = None
        self._llm_initialised = False
        
        # Set device early
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda" and hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        # Create FastAPI app first so endpoints are available immediately
        self._create_app()
        
        # Start warm-up in background so /status can reflect progress
        print("Starting background warmup...")
        self._warmup_started = True
        asyncio.create_task(self._startup_warmup())

    async def _startup_warmup(self):
        """Background warmup process with thread-safe initialization"""
        async with self._warmup_lock:
            if self._ready_event.is_set() or self._failed_event.is_set():
                return  # already warm or failed
            
            # self._warmup_started = True  # Already set before acquiring lock
            try:
                print("🔥 Warming up tokenizer...")
                await self._init_tokenizer()
                
                print("🔥 Warming up vLLM engine...")
                await self._init_vllm_engine()
                self._llm_initialised = True
                
                print("🔥 Warming up SNAC processor...")
                self.snac_processor = SNACProcessor(self.device)
                self.snac_processor.load()
                
                print("✅ WARMUP COMPLETE - System ready!")
                self._ready_event.set()
                
            except Exception as e:
                print(f"❌ Warmup failed: {e}")
                self._warmup_error = str(e)
                self._failed_event.set()

    async def _ensure_ready(self):
        """Wait for the system to be ready before processing requests"""
        if self._failed_event.is_set():
            raise RuntimeError(f"System failed to initialize: {self._warmup_error}")
        
        await self._ready_event.wait()
        
        if not hasattr(self, 'llm_engine') or not hasattr(self, 'snac_processor'):
            raise RuntimeError("System failed to initialize properly")

    def _get_model_path(self):
        """Get the best available model path using smart fallback"""
        # Try persistent cache first
        if os.path.exists(VEENA_CACHE_PATH) and os.listdir(VEENA_CACHE_PATH):
            print(f"Using model from persistent cache: {VEENA_CACHE_PATH}")
            return VEENA_CACHE_PATH
        
        # Try temp cache (image build cache)
        temp_veena_path = "/tmp/veena_models"
        if os.path.exists(temp_veena_path) and os.listdir(temp_veena_path):
            print(f"Using model from temp cache: {temp_veena_path}")
            # Copy to persistent cache for future use
            self._copy_to_persistent_cache(temp_veena_path, VEENA_CACHE_PATH)
            return temp_veena_path
        
        # Fallback to HuggingFace model ID
        print(f"Using model from HuggingFace: {MODEL_ID}")
        return MODEL_ID

    def _copy_to_persistent_cache(self, src_path: str, dst_path: str):
        """Copy model from temp cache to persistent cache"""
        try:
            import shutil
            os.makedirs(dst_path, exist_ok=True)
            if os.path.exists(dst_path) and os.listdir(dst_path):
                print(f"Persistent cache already exists at {dst_path}")
                return
            print(f"Copying model from {src_path} to {dst_path}")
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            print(f"Successfully copied to persistent cache: {dst_path}")
        except Exception as e:
            print(f"Failed to copy to persistent cache: {e}")

    async def _init_tokenizer(self):
        from transformers import AutoTokenizer
        print("Loading tokenizer...")
        
        # Try persistent cache first
        if os.path.exists(VEENA_CACHE_PATH) and os.listdir(VEENA_CACHE_PATH):
            try:
                print(f"Loading tokenizer from persistent cache: {VEENA_CACHE_PATH}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    VEENA_CACHE_PATH,
                    trust_remote_code=True,
                    local_files_only=True,
                    use_fast=True,
                )
                print("✓ Loaded tokenizer from persistent cache")
                return
            except Exception as e:
                print(f"Failed to load from persistent cache: {e}")
        
        # Try temp cache (image build cache)
        temp_veena_path = "/tmp/veena_models"
        if os.path.exists(temp_veena_path) and os.listdir(temp_veena_path):
            try:
                print(f"Loading tokenizer from temp cache: {temp_veena_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    temp_veena_path,
                    trust_remote_code=True,
                    use_fast=True,
                )
                # Copy to persistent cache for future use
                self._copy_to_persistent_cache(temp_veena_path, VEENA_CACHE_PATH)
                print("✓ Loaded tokenizer from temp cache")
                return
            except Exception as e:
                print(f"Failed to load from temp cache: {e}")
        
        # Fallback to downloading from HuggingFace
        print(f"Downloading tokenizer from HuggingFace: {MODEL_ID}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            use_fast=True,
        )
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = PAD_TOKEN_ID
        print("✓ Tokenizer ready")

    async def _init_vllm_engine(self):
        from vllm import AsyncLLMEngine, AsyncEngineArgs
        print(f"Initializing vLLM engine (v{VLLM_INSTALL_VERSION})...")
        
        # Determine model path using smart fallback
        model_path = self._get_model_path()

        engine_args = AsyncEngineArgs(
            model=model_path,
            tokenizer=model_path,
            tokenizer_mode="auto",
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            enforce_eager=True,  # Enable eager execution for stability
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            use_v2_block_manager=True,  # Use V2 block manager for better performance
        )
        self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("✓ vLLM engine created")
        await self._warmup_llm()

    async def _warmup_llm(self):
        from vllm import SamplingParams, TokensPrompt
        print("Warming up vLLM engine...")
        warm_text_for_vllm_prompt = f"{DEFAULT_INTERNAL_SPEAKER}: नमस्ते"
        warm_prompt_ids_list = self._format_prompt(warm_text_for_vllm_prompt)
        
        tokens_prompt_for_warmup = TokensPrompt(prompt_token_ids=warm_prompt_ids_list)
        
        stop_ids = list(set([END_OF_SPEECH_TOKEN, END_OF_AI_TOKEN, self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else END_OF_AI_TOKEN]))
        sampling_params = SamplingParams(max_tokens=10, temperature=0.1, stop_token_ids=stop_ids)
        req_id = f"vllm_warmup_{time.time_ns()}"
        
        try:
            results_gen = self.llm_engine.generate(
                prompt=tokens_prompt_for_warmup,
                sampling_params=sampling_params,
                request_id=req_id,
            )
            async for out in results_gen:
                if out.finished:
                    break
            print("✓ vLLM warmup complete")
        except Exception as e:
            print(f"vLLM warmup failed: {e}")

    def _format_prompt(self, text_with_speaker: str) -> List[int]:
        speaker_tokens = self.tokenizer.encode(text_with_speaker, add_special_tokens=False)
        return [
            START_OF_HUMAN_TOKEN,
            *speaker_tokens,
            END_OF_HUMAN_TOKEN,
            START_OF_AI_TOKEN,
            START_OF_SPEECH_TOKEN,
        ]

    def _create_wav_header(self, sr=TARGET_AUDIO_SAMPLING_RATE, ch=1, bps=16) -> bytes:
        data_chunk_size_for_streaming = 0 
        riff_chunk_size = 36 + data_chunk_size_for_streaming 
        return struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", riff_chunk_size, 
            b"WAVE", b"fmt ", 16, 1, ch, sr,
            sr * ch * (bps // 8), ch * (bps // 8), bps,
            b"data", data_chunk_size_for_streaming 
        )

    async def _generate_audio_unified(
        self, request: TTSRequest, analytics: GenerationAnalytics
    ) -> AsyncGenerator[bytes, None]:
        from vllm import SamplingParams, TokensPrompt

        # Ensure system is ready before processing
        await self._ensure_ready()

        # Convert customer speaker ID to internal speaker name
        internal_speaker_id = SPEAKER_MAPPING.get(request.speaker_id, request.speaker_id)
        prompt_ids_list = self._format_prompt(f"{internal_speaker_id}: {request.text}")
        tokens_prompt_for_generation = TokensPrompt(prompt_token_ids=prompt_ids_list)
        
        stop_ids = list(set([END_OF_SPEECH_TOKEN, END_OF_AI_TOKEN, self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else END_OF_AI_TOKEN]))
        
        sampling_params = SamplingParams(
            max_tokens=request.max_new_tokens,
            temperature=max(0.01, request.temperature) if request.temperature > 0 else 0.0,
            top_p=request.top_p if request.temperature > 0 else 1.0,
            repetition_penalty=request.repetition_penalty,
            stop_token_ids=stop_ids,
            seed=request.seed,
        )
        req_id = f"tts_{int(time.time()*1000)}_{np.random.randint(1000)}"

        yield self._create_wav_header()

        snac_token_buffer: List[int] = []
        previous_output_length_in_ids = 0
        first_token_marked = False

        try:
            results_stream = self.llm_engine.generate(
                prompt=tokens_prompt_for_generation,
                sampling_params=sampling_params,
                request_id=req_id,
            )
            async for result in results_stream:
                if not result.outputs: 
                    continue
                
                current_generated_token_ids = result.outputs[0].token_ids
                newly_generated_token_ids = current_generated_token_ids[previous_output_length_in_ids:]
                previous_output_length_in_ids = len(current_generated_token_ids)

                if newly_generated_token_ids and not first_token_marked:
                    if analytics:
                        analytics.mark_first_token()
                    first_token_marked = True

                if analytics:
                    analytics.total_tokens += len(newly_generated_token_ids)

                for token_id in newly_generated_token_ids:
                    if AUDIO_CODE_BASE_OFFSET <= token_id < (AUDIO_CODE_BASE_OFFSET + 7 * 4096):
                        snac_token_buffer.append(token_id)
                        if analytics:
                            analytics.snac_tokens += 1

                # Process buffer with sliding window approach
                while len(snac_token_buffer) >= SNAC_WINDOW_SIZE_TOKENS:
                    current_window_tokens = snac_token_buffer[:SNAC_WINDOW_SIZE_TOKENS]
                    
                    # Time SNAC decoding for analytics
                    snac_start = time.time() if analytics else None
                    audio_bytes_slice = await asyncio.get_event_loop().run_in_executor(
                        None, self.snac_processor.decode_window_and_get_hop_slice, current_window_tokens
                    )
                    if analytics:
                        snac_duration = time.time() - snac_start
                        analytics.add_snac_decode_time(snac_duration)
                    
                    if audio_bytes_slice:
                        if analytics:
                            analytics.mark_first_audio()
                            analytics.audio_chunks += 1
                        yield audio_bytes_slice
                    
                    snac_token_buffer = snac_token_buffer[SNAC_HOP_SIZE_TOKENS:]
                
                if result.finished: 
                    break
                    
        except Exception as e:
            print(f"[{req_id}] Generation error: {e}")

        # Flush remaining tokens
        if len(snac_token_buffer) >= 7:
            final_flush_len = (len(snac_token_buffer) // 7) * 7
            if final_flush_len > 0:
                chunk_to_decode = snac_token_buffer[:final_flush_len]
                snac_start = time.time() if analytics else None
                audio_bytes = await asyncio.get_event_loop().run_in_executor(
                    None, self.snac_processor.decode_chunk, chunk_to_decode
                )
                if analytics:
                    analytics.add_snac_decode_time(time.time() - snac_start)
                if audio_bytes: 
                    if analytics:
                        analytics.audio_chunks += 1
                    yield audio_bytes
        
        if analytics:
            analytics.mark_end()

    def _create_app(self):
        self.app = fastapi.FastAPI(
            title="Veena TTS API - Maya Research",
            version="1.0.0",
            description="Advanced Text-to-Speech API with 10 Hindi voices",
        )
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.app.post("/generate", response_class=Response)
        async def generate_endpoint(request: TTSRequest) -> Response:
            analytics = GenerationAnalytics() if ENABLE_ANALYTICS else None
            
            if request.streaming:
                async def stream_with_analytics():
                    async for chunk in self._generate_audio_unified(request, analytics):
                        yield chunk
                    
                    # Add analytics if enabled and overhead is acceptable
                    if analytics:
                        metrics = analytics.get_metrics()
                        overhead_ms = (time.time() - analytics.start_time) * 1000 - metrics.get("total_time_s", 0) * 1000
                        if overhead_ms <= ANALYTICS_OVERHEAD_THRESHOLD_MS:
                            print(f"Analytics: {metrics}")
                
                return StreamingResponse(stream_with_analytics(), media_type="audio/wav")
            else:
                # Non-streaming: collect all chunks
                all_audio_bytes = b""
                async for chunk in self._generate_audio_unified(request, analytics):
                    all_audio_bytes += chunk
                
                # Add analytics headers for non-streaming
                headers = {}
                if analytics:
                    metrics = analytics.get_metrics()
                    headers.update({
                        "X-TTFT": str(metrics.get("ttft_s", 0)),
                        "X-TTFA": str(metrics.get("ttfa_s", 0)),
                        "X-Total-Time": str(metrics.get("total_time_s", 0)),
                        "X-Tokens-Per-Second": str(metrics.get("tokens_per_second", 0)),
                        "X-SNAC-Tokens": str(metrics.get("snac_tokens", 0)),
                        "X-Audio-Chunks": str(metrics.get("audio_chunks", 0)),
                    })
                
                return Response(content=all_audio_bytes, media_type="audio/wav", headers=headers)

        @self.app.get("/")
        async def root_endpoint() -> Dict[str, Any]:
            """Veena TTS API Information - Maya Research"""
            return {
                "product": "Veena",
                "company": "Maya Research",
                "description": "Advanced Text-to-Speech API with 10 Hindi voices",
                "version": self.app.version,
                "status": "operational",
                "developers": ["Dheemanth", "Bharath Kumar"],
                "languages_supported": ["Hindi"],
                "total_speakers": len(SPEAKER_DETAILS),
                "speaker_types": {
                    "female_voices": 8,
                    "male_voices": 2
                },
                "features": [
                    "Real-time streaming TTS",
                    "Multiple voice personalities", 
                    "Professional audio quality",
                    "Low latency generation"
                ],
                "endpoints": {
                    "generate_audio": "/generate",
                    "list_speakers": "/speakers", 
                    "speaker_details": "/speakers/{speaker_id}",
                    "system_status": "/status",
                    "health_check": "/health"
                },
                "default_speaker": DEFAULT_CUSTOMER_SPEAKER,
                "analytics_enabled": ENABLE_ANALYTICS
            }

        @self.app.get("/status")
        async def status_endpoint() -> Dict[str, Any]:
            """Fast readiness probe - client can poll this before submitting jobs"""
            if self._ready_event.is_set():
                state = "ready"
            elif self._failed_event.is_set():
                state = "failed"
            elif self._warmup_started:
                state = "warming_up"
            else:
                state = "initializing"
            
            response = {
                "ready": self._ready_event.is_set(),
                "state": state,
                "gpu": GPU_CONFIG,
                "version": self.app.version,
                "warmup_started": self._warmup_started
            }
            
            if self._failed_event.is_set() and self._warmup_error:
                response["error"] = self._warmup_error
                
            return response

        @self.app.get("/health")
        async def health_endpoint() -> Dict[str, Any]:
            """Health check endpoint"""
            return {
                "status": "healthy" if self._ready_event.is_set() else "warming_up", 
                "gpu": GPU_CONFIG,
                "ready": self._ready_event.is_set()
            }

        @self.app.get("/speakers")
        async def speakers_endpoint() -> JSONResponse:
            """Get all available speakers - does not require warmup"""
            speakers_sorted = sorted(SPEAKER_DETAILS.values(), key=lambda s: s["name"].lower())
            payload = {
                "speakers": speakers_sorted,
                "total_count": len(SPEAKER_DETAILS),
                "default_speaker": DEFAULT_CUSTOMER_SPEAKER,
                "languages_supported": ["Hindi"]
            }
            return JSONResponse(content=payload, headers={"Cache-Control": "public, max-age=3600"})

        @self.app.get("/speakers/{speaker_id:path}")
        async def speaker_detail_endpoint(speaker_id: str) -> JSONResponse:
            """Get detailed information about a specific speaker - does not require warmup"""
            speaker_key = speaker_id.lower()
            
            if speaker_key not in SPEAKER_DETAILS:
                available_speakers = list(SPEAKER_DETAILS.keys())
                payload = {
                    "error": "Speaker not found",
                    "speaker_id": speaker_id,
                    "available_speakers": available_speakers
                }
                return JSONResponse(
                    status_code=404,
                    content=payload,
                    headers={"Cache-Control": "public, max-age=3600"}
                )
            
            speaker_info = SPEAKER_DETAILS[speaker_key].copy()
            speaker_info["internal_model_id"] = SPEAKER_MAPPING.get(speaker_key, "unknown")
            return JSONResponse(content=speaker_info, headers={"Cache-Control": "public, max-age=3600"})

    @modal.asgi_app()
    def asgi_app(self):
        return self.app
    
    @modal.exit()
    async def cleanup(self):
        print("Veena TTS API shutting down...")
        
        # Gracefully shutdown vLLM engine
        if getattr(self, '_llm_initialised', False) and hasattr(self, 'llm_engine'):
            if self.llm_engine is not None:
                try:
                    # Check if vLLM has an async shutdown method
                    if hasattr(self.llm_engine, 'shutdown'):
                        await self.llm_engine.shutdown()
                    elif hasattr(self.llm_engine, 'stop'):
                        await self.llm_engine.stop()
                except Exception as e:
                    print(f"Error during vLLM shutdown: {e}")
                finally:
                    del self.llm_engine 
                    self.llm_engine = None
        
        # Clean up SNAC processor
        if hasattr(self, 'snac_processor') and hasattr(self.snac_processor, 'model') and self.snac_processor.model is not None:
            del self.snac_processor.model
            self.snac_processor.model = None
            
        # Clean up tokenizer
        if hasattr(self, 'tokenizer'): 
            del self.tokenizer
            self.tokenizer = None
            
        # Clear CUDA cache
        if torch.cuda.is_available(): 
            torch.cuda.empty_cache()
            
        print("Cleanup complete")


# ===== LOCAL TESTING ENTRYPOINT =====
@app.local_entrypoint()
def test_both_modes(
    text: str = "नमस्ते, आपका दिन कैसा रहा?",
    speaker: str = "vinaya_assist",
):
    print(f"Testing Veena TTS API - Maya Research (Version: 1.0.0):")
    print(f"  Text: '{text}'")
    print(f"  Speaker: '{speaker}'")
    print(f"  Company: Maya Research")
    print(f"  Product: Veena TTS")
    print(f"  Analytics: {'Enabled' if ENABLE_ANALYTICS else 'Disabled'}")
    print(f"  Speaker Configuration: speakers.json ({len(SPEAKER_DETAILS)} voices loaded)")
    print("\nTo deploy: modal deploy stream_app.py\n")
    print("API Information:")
    print("curl YOUR_MODAL_APP_URL/")
    print("\nStatus check:")
    print("curl YOUR_MODAL_APP_URL/status")
    print("\nList speakers:")
    print("curl YOUR_MODAL_APP_URL/speakers")
    print("\nStreaming TTS generation:")
    print(f'''curl -N -X POST "YOUR_MODAL_APP_URL/generate" \\
  -H "Content-Type: application/json" \\
  -d '{{"text": "{text}", "speaker_id": "{speaker}", "streaming": true, "max_new_tokens": 700}}' \\
  --output veena_streaming_output.wav''')
    print("\nNon-streaming TTS generation:")
    print(f'''curl -X POST "YOUR_MODAL_APP_URL/generate" \\
  -H "Content-Type: application/json" \\
  -d '{{"text": "{text}", "speaker_id": "{speaker}", "streaming": false, "max_new_tokens": 700}}' \\
  --output veena_output.wav''')
    print("\nReplace YOUR_MODAL_APP_URL with the actual URL from 'modal deploy'.")
    print("\n🎤 Veena - Advanced Hindi TTS by Maya Research 🎤")