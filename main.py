from scripts import *
from pathlib import Path
from src import *

def preprocess_hagrid_data() -> None:
    """Preprocess HAGRID dataset for training."""
    training_preprocesor.main()

def generate_sample_image() -> None:
    """Generate a sample image with stable diffusion"""
    sd = diffusion.Pipeline(model_name="sd15")
    generator = diffusion.Generator()

    pipe = sd.pipe
    generator.generate_image(pipe, "cat", output_path="output.png")

def train_controlnet(epochs: int = 1, batch_size: int = 1, learning_rate: float = 1e-5, max_samples: int = 50, resume: bool = False) -> None:
    """Train ControlNet model on dataset"""
    from src.control.controlnet_train import ControlNetTrainer
    trainer = ControlNetTrainer()
    trainer.train(num_epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, max_samples=max_samples, resume=resume)

def generate_video_frames(
    conditioning_dir: str = "data/input",
    output_dir: str = "output/video",
    prompt: str = "",
    num_inference_steps: int = 20,
    strength: float = 0.7,
    temporal_blend: float = 0.5,
    fps: float = 24.0,
) -> None:
    """Generate a temporally-consistent video from a directory of conditioning images."""
    from src.video.video_generator import VideoGenerator
    VideoGenerator().generate_video(
        conditioning_dir=Path(conditioning_dir),
        output_dir=Path(output_dir),
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        strength=strength,
        temporal_blend=temporal_blend,
        fps=fps,
    )

def generate_video_animatediff(
    conditioning_dir: str = "data/input",
    output_dir: str = "output/video",
    prompt: str = "",
    num_inference_steps: int = 20,
    fps: float = 24.0,
) -> None:
    """Generate video using AnimateDiff latent temporal attention (requires diffusers>=0.24)."""
    from src.video.animatediff_generator import AnimateDiffVideoGenerator
    AnimateDiffVideoGenerator().generate_video(
        conditioning_dir=Path(conditioning_dir),
        output_dir=Path(output_dir),
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        fps=fps,
    )

def generate_controlnet_image() -> None:
    """Generate images using trained ControlNet model"""
    from src.control.controlnet_train import ControlNetTrainer
    trainer = ControlNetTrainer()

    # for i in range(1, 50000, 5000):
    #     trainer.generate_image_grid(
    #         prompt="A realistic photo of a human hand",
    #         conditioning_image_path=str(Path(f"data/hagrid_train/conditioning_images/{i}.png")),
    #         output_path=f"output_controlnet_10x10_{i}.png",
    #         rows=4,
    #         cols=4,
    #         # num_inference_steps=50
    #     )

    # trainer.generate_image_grid(
    #     prompt="A realistic photo of a human hand",
    #     conditioning_image_path=str(Path(f"data/input/0.png")),
    #     output_path=f"output_controlnet_10x10_0.png",
    #     rows=4,
    #     cols=4,
    # )

    for i in range(23):
        trainer.generate_image(
            prompt="A realistic photo of a human hand",
            conditioning_image_path=str(Path(f"data/input/{i}.png")),
            output_path=f"output_controlnet_{i}.png",
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1, help="num epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size / gpu")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="lr")
    # parser.add_argument("--max-samples", type=int, default=50, help="max samples for train")
    parser.add_argument("--skip-generate", action="store_true", help="skip img generation after train")
    parser.add_argument("--restart", action="store_true", help="start from a fresh instead of the last save")
    parser.add_argument("--generate-video", action="store_true", help="generate video from conditioning PNGs in --video-conditioning-dir")
    parser.add_argument("--video-method", choices=["flow", "animatediff"], default="flow", help="flow=optical-flow pixel-blend (default); animatediff=latent temporal attention")
    parser.add_argument("--video-prompt", type=str, default="", help="prompt for video generation")
    parser.add_argument("--video-conditioning-dir", type=str, default="data/input", help="directory of skeleton PNG frames (sorted by name = frame order)")
    parser.add_argument("--video-output-dir", type=str, default="output/video", help="output directory for frames and video")
    parser.add_argument("--video-strength", type=float, default=0.5, help="img2img denoising strength (lower=more stable; int(strength*steps)>=8 required)")
    parser.add_argument("--video-blend", type=float, default=0.65, help="pixel-space temporal blend (0=pure generated, 1=pure warped prev frame)")
    parser.add_argument("--video-steps", type=int, default=50, help="num inference steps per frame")
    parser.add_argument("--video-fps", type=float, default=24.0, help="fps for assembled output video")
    args = parser.parse_args()

    # preprocess_hagrid_data()
    # generate_sample_image()
    # train_controlnet(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, max_samples=None, resume=not args.restart)
    if not args.skip_generate:
        generate_controlnet_image()
    if args.generate_video:
        if args.video_method == "animatediff":
            generate_video_animatediff(
                conditioning_dir=args.video_conditioning_dir,
                output_dir=args.video_output_dir,
                prompt=args.video_prompt,
                num_inference_steps=args.video_steps,
                fps=args.video_fps,
            )
        else:
            generate_video_frames(
                conditioning_dir=args.video_conditioning_dir,
                output_dir=args.video_output_dir,
                prompt=args.video_prompt,
                num_inference_steps=args.video_steps,
                strength=args.video_strength,
                temporal_blend=args.video_blend,
                fps=args.video_fps,
            )