import os

from scripts import *
from pathlib import Path

from src import *

def preprocess_freihand_data() -> None:
    print("[main] Running FreiHAND preprocessing...")
    preprocessor_freihand.main()

def generate_sample_image() -> None:
    sd = diffusion.Pipeline(model_name="sdxl")
    generator = diffusion.Generator()

    pipe = sd.pipe
    generator.generate_image(pipe, "cat", output_path="output.png")

if __name__ == "__main__":
    preprocess_freihand_data()
    generate_sample_image()
