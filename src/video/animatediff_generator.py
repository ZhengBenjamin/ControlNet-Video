from __future__ import annotations

import torch

from pathlib import Path
from PIL import Image

from src.config import MODELS_DIR, DATA_DIR
from src.control.controlnet_train import ControlNetTrainer
from src.video.video_generator import _save_video
from diffusers import AnimateDiffControlNetPipeline, MotionAdapter, DDIMScheduler, ControlNetModel

DEFAULT_CONDITIONING_DIR = DATA_DIR / "input"
DEFAULT_VIDEO_OUTPUT_DIR = Path("output") / "video"

# AnimateDiff motion adapter — SD1.5; 16-frame window
MOTION_ADAPTER_ID = "guoyww/animatediff-motion-adapter-v1-5-2"


class AnimateDiffVideoGenerator:
    """Generate temporally consistent video using AnimateDiff motion modules + ControlNet
    All frames denoised together in single pass + cond input per frame
    """

    def __init__(self, model_dir: Path = MODELS_DIR / "controlnet-hagrid-sd15") -> None:
        self.model_dir = model_dir
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def generate_video(
        self,
        conditioning_dir: Path = DEFAULT_CONDITIONING_DIR,
        output_dir: Path = DEFAULT_VIDEO_OUTPUT_DIR,
        prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 9.0,
        controlnet_conditioning_scale: float = 1.0,
        fps: float = 24.0) -> None:
        """Single animatediff pass to gen all frames"""

        conditioning_dir = Path(conditioning_dir)
        cond_paths = sorted(conditioning_dir.glob("*.png"))

        output_dir = Path(output_dir)
        frame_dir = output_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)

        n_frames = len(cond_paths)
        print(f"[animatediff] generating {n_frames} frames")

        trainer = ControlNetTrainer(self.model_dir)
        sd15_source, use_local = trainer.get_model_source()

        adapter = MotionAdapter.from_pretrained(MOTION_ADAPTER_ID, torch_dtype=torch.float16)
        controlnet = ControlNetModel.from_pretrained(self.model_dir, torch_dtype=torch.float16)

        scheduler = DDIMScheduler.from_pretrained(
            sd15_source,
            subfolder="scheduler",
            local_files_only=use_local,
            clip_sample=False,
            timestep_spacing="linspace",
            beta_schedule="linear",
            steps_offset=1,
        )

        pipe = AnimateDiffControlNetPipeline.from_pretrained(
            sd15_source,
            controlnet=controlnet,
            motion_adapter=adapter,
            scheduler=scheduler,
            torch_dtype=torch.float16,
            local_files_only=use_local,
            safety_checker=None,
        ).to(self._device)

        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()

        # freenoise -> extends larger window
        pipe.enable_free_noise(context_length=16, context_stride=4)

        all_conds = [
            Image.open(p).convert("RGB").resize((512, 512))
            for p in cond_paths
        ]

        generator = torch.Generator(device="cpu")

        print(f"[animatediff] running pipeline with {n_frames} frames...")
        result = pipe(
            prompt=prompt,
            conditioning_frames=all_conds,
            num_frames=n_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator)
        
        frames = result.frames[0]

        for idx, frame in enumerate(frames):
            frame.save(frame_dir / f"frame_{idx:04d}.png")
            print(f"[animatediff] saved frame_{idx:04d}.png")

        _save_video(frame_dir, output_dir / "video.mp4", fps=fps)

        overlay_path = output_dir / "video_skeleton_overlay.mp4"
        print(f"[animatediff] writing skeleton overlay video -> {overlay_path}")
        self._save_skeleton_overlay_video(frame_dir, cond_paths, overlay_path, fps=fps)

    def _save_skeleton_overlay_video(
        self,
        frame_dir: Path,
        cond_paths: list,
        output_path: Path,
        fps: float = 24.0,
        alpha: float = 0.4) -> None:
        """Video with skeleton overlay"""

        frame_paths = sorted(frame_dir.glob("frame_*.png"))
        if not frame_paths:
            return

        first = Image.open(frame_paths[0]).convert("RGBA")
        w, h = first.size

        composite_dir = output_path.parent / "frames_overlay"
        composite_dir.mkdir(parents=True, exist_ok=True)

        for idx, (fp, cp) in enumerate(zip(frame_paths, cond_paths)):
            base = Image.open(fp).convert("RGBA")
            skel = Image.open(cp).convert("RGBA").resize((w, h))
            r, g, b, a = skel.split()
            skel = Image.merge("RGBA", (r, g, b, a.point(lambda p: int(p * alpha))))
            composite = Image.alpha_composite(base, skel).convert("RGB")
            composite.save(composite_dir / f"frame_{idx:04d}.png")

        _save_video(composite_dir, output_path, fps=fps)
