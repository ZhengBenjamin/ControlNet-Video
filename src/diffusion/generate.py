import time

def generate_image(pipe, prompt, output_path="output.png"):
    """Generates an image from a prompt and saves it to the specified path"""

    image = pipe(prompt).images[0]
    image.save(output_path)


def generate_image_with_timing(pipe, prompt, output_path="output.png"):
    """Generates an image and returns the generation time"""
    start_time = time.time()
    image = pipe(prompt).images[0]
    generation_time = time.time() - start_time
    
    image.save(output_path)
    return generation_time