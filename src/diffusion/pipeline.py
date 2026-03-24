import torch
from huggingface_hub import login

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusion3Pipeline
from src.config import MODELS_DIR

def load_pipeline():
    """Loads the Stable Diffusion 3 Medium pipeline with memory optimizations"""

    login()

    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16,
        cache_dir=str(MODELS_DIR)
    )

    pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()
    
    return pipe

def load_pipeline_sdxl():

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir=str(MODELS_DIR),
        use_safetensors=True
    )

    pipe.to("cuda")
    return pipe