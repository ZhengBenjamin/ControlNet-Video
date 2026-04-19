from __future__ import annotations

import cv2
import numpy as np
import torch

from pathlib import Path
from typing import Optional
from PIL import Image

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
)

from src.config import MODELS_DIR, DATA_DIR
from src.control.controlnet_train import ControlNetTrainer

DEFAULT_CONDITIONING_DIR = DATA_DIR / "input"
DEFAULT_VIDEO_OUTPUT_DIR = Path("output") / "video"


def _warp_frame(
    frame: np.ndarray,
    prev_cond: np.ndarray,
    curr_cond: np.ndarray) -> np.ndarray:
    """Warp frame using dense optical flow est bt consecutive conditioning images
    Flow comp on condition img for cleaner motion est"""

    prev_gray = cv2.cvtColor(prev_cond, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr_cond, cv2.COLOR_RGB2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0)
    
    h, w = flow.shape[:2]
    map_x = np.arange(w, dtype=np.float32)[None, :] + flow[:, :, 0]
    map_y = np.arange(h, dtype=np.float32)[:, None] + flow[:, :, 1]

    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _save_video(frame_dir: Path, output_path: Path, fps: float = 24.0) -> None:
    """Assemble sorted PNG frames from frame_dir into an mp4 at output_path."""

    frames = sorted(frame_dir.glob("frame_*.png"))
    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for path in frames:
        img = cv2.imread(str(path))
        
        if img is not None:
            writer.write(img)

    writer.release()
    print(f"[video] Video saved to {output_path}")


class VideoGenerator:

    def __init__(self, model_dir: Path = MODELS_DIR / "controlnet-hagrid-sd15") -> None:
        self.model_dir = model_dir
        self._device = torch.device("cuda:0")

    def _load_pipes(self) -> tuple:
        """Load ControlNet weights once; share them between both pipelines."""

        trainer = ControlNetTrainer(self.model_dir)
        sd15_source, use_local = trainer.get_model_source()

        controlnet = ControlNetModel.from_pretrained(self.model_dir, torch_dtype=torch.float16)

        common_kwargs: dict = dict(
            controlnet=controlnet,
            torch_dtype=torch.float16,
            local_files_only=use_local,
            safety_checker=None,
        )

        txt2img = StableDiffusionControlNetPipeline.from_pretrained(
            sd15_source, **common_kwargs).to(self._device)
        txt2img.enable_attention_slicing()

        img2img = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            sd15_source, **common_kwargs).to(self._device)
        img2img.enable_attention_slicing()

        return txt2img, img2img

    def generate_video(
        self,
        conditioning_dir: Path = DEFAULT_CONDITIONING_DIR,
        output_dir: Path = DEFAULT_VIDEO_OUTPUT_DIR,
        prompt: str = "",
        num_inference_steps: int = 20,
        strength: float = 0.5,
        temporal_blend: float = 0.65,
        controlnet_conditioning_scale: float = 1.0,
        fps: float = 24.0) -> None:

        """Generate a video frame by frame with flow guided temporal conditioning.
        First frame generated from text prompt, then subsequent frames generated from prev frame + cond img flow warp

        strength = controls how strong new frame gen vs staying close to prev warped frame
        temporal_blend = pixel space blend bt warped prv and new gen
        """
        conditioning_dir = Path(conditioning_dir)
        cond_paths = sorted(conditioning_dir.glob("*.png"))

        output_dir = Path(output_dir)
        frame_dir = output_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)

        print(f"[video] {len(cond_paths)} conditioning frames from {conditioning_dir}")

        txt2img, img2img = self._load_pipes()
        generator = torch.Generator(device=self._device)

        prev_frame_np: Optional[np.ndarray] = None
        prev_cond_np: Optional[np.ndarray] = None

        for idx, cond_path in enumerate(cond_paths):
            cond_img = Image.open(cond_path).convert("RGB").resize((512, 512))
            cond_np = np.array(cond_img)

            if prev_frame_np is None:
                # full text to img gen for first frame
                out = txt2img(
                    prompt=prompt,
                    image=cond_img,
                    num_inference_steps=num_inference_steps,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    generator=generator).images[0]
                prev_frame_np = np.array(out)
            else:
                # n+1: warp previous output, use as img2img init
                warped_np = _warp_frame(prev_frame_np, prev_cond_np, cond_np)
                init_img = Image.fromarray(warped_np)
                out = img2img(
                    prompt=prompt,
                    image=init_img,
                    control_image=cond_img,
                    strength=strength,
                    num_inference_steps=num_inference_steps,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    generator=generator).images[0]
                
                # pix blend, anchors gened frame to warped prev, helps reduce color drift
                out_np = np.array(out, dtype=np.float32)
                
                blended_np = (
                    temporal_blend * warped_np.astype(np.float32)
                    + (1.0 - temporal_blend) * out_np).clip(0, 255).astype(np.uint8)
                
                out = Image.fromarray(blended_np)
                prev_frame_np = blended_np

            frame_path = frame_dir / f"frame_{idx:04d}.png"
            out.save(frame_path)
            prev_cond_np = cond_np
            print(f"[video] {idx + 1}/{len(cond_paths)} -> {frame_path.name}")

        _save_video(frame_dir, output_dir / "video.mp4", fps=fps)
