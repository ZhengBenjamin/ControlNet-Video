import torch
from huggingface_hub import login
from pathlib import Path

from diffusers import StableDiffusionXLPipeline, StableDiffusion3Pipeline, StableDiffusionXLControlNetPipeline, ControlNetModel
from src.config import MODELS_DIR
from typing import Optional

DEFAULT_SDXL_CONTROLNET_PATH = MODELS_DIR / "controlnet-freihand-sdxl"

class Pipeline():
    
    def __init__(self, model_name: Optional[str] = None):
        
        if model_name == "sdxl":
            self.pipe = self.load_pipeline_sdxl()
        elif model_name == "sd3":
            self.pipe = self.load_pipeline_sd3()
        else:
            raise ValueError(f"Invalid model name: {model_name}. Must be 'sdxl' or 'sd3'")


    def load_pipeline_sd3(self) -> StableDiffusion3Pipeline:
        login()

        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=torch.float16,
            cache_dir=str(MODELS_DIR)
        )

        pipe.enable_attention_slicing()
        pipe.enable_model_cpu_offload()
        
        return pipe

    def load_pipeline_sdxl(self) -> StableDiffusionXLPipeline:

        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            # variant="fp16",
            cache_dir=str(MODELS_DIR),
        )

        # pipe.enable_attention_slicing()
        # pipe.enable_model_cpu_offload()

        pipe.to("cuda")
        return pipe
