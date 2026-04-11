import torch
from pathlib import Path

from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from src.config import MODELS_DIR
from typing import Optional, Union

DEFAULT_SD15_CONTROLNET_PATH = MODELS_DIR / "controlnet-freihand-sd15"

class Pipeline:
    """Load and manage stable diffusion pipelines"""
    
    def __init__(self, model_name: Optional[str] = None, controlnet_path: Optional[Path] = None) -> None:
        """Init pipeline with specified model"""
        if model_name == "sd15":
            self.pipe = self.load_pipeline_sd15()
        elif model_name == "sd15_controlnet":
            if controlnet_path is None:
                controlnet_path = DEFAULT_SD15_CONTROLNET_PATH
            self.pipe = self.load_pipeline_sd15_controlnet(controlnet_path)
        else:
            raise ValueError(f"Invalid model name: {model_name}. Must be 'sd15' or 'sd15_controlnet'")

    def load_pipeline_sd15(self) -> StableDiffusionPipeline:
        """Load base stable diffusion v1.5 pipeline"""
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            cache_dir=str(MODELS_DIR)
        )

        pipe.enable_attention_slicing()
        pipe.enable_model_cpu_offload()
        
        return pipe

    def load_pipeline_sd15_controlnet(self, controlnet_path: Path) -> StableDiffusionControlNetPipeline:
        """Load stable diffusion v1.5 pipeline with ControlNet"""
        controlnet = ControlNetModel.from_pretrained(
            controlnet_path,
            torch_dtype=torch.float16,
        )

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            cache_dir=str(MODELS_DIR),
        )

        pipe.enable_attention_slicing()
        pipe.enable_model_cpu_offload()

        return pipe
