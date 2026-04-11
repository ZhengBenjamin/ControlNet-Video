from scripts import *
from pathlib import Path
from accelerate import Accelerator
from src import *

def preprocess_freihand_data() -> None:
    """Preprocess FreiHAND dataset for training"""
    accelerator = Accelerator()
    if accelerator.is_main_process:
        preprocessor_freihand.main()

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

def generate_controlnet_image() -> None:
    """Generate images using trained ControlNet model"""
    from src.control.controlnet_train import ControlNetTrainer
    trainer = ControlNetTrainer()
    
    if not trainer.accelerator.is_main_process:
        return
    
    trainer.generate_image_grid(
        prompt="",
        conditioning_image_path=str(Path("data/freihand_controlnet/conditioning_images/00000000.png")),
        output_path="output_controlnet_5x5.png",
        rows=5,
        cols=5,
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
    args = parser.parse_args()

    # preprocess_freihand_data()
    # generate_sample_image()
    train_controlnet(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, max_samples=1000, resume=not args.restart)
    if not args.skip_generate:
        generate_controlnet_image()
