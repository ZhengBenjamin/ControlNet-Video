import src

if __name__ == "__main__":
    # pipe = src.diffusion.load_pipeline_sdxl()

    pipe = src.diffusion.load_pipeline()
    src.diffusion.generate_image(pipe, "shit", f"output.png")