from typing import Optional


class Generator:

    DEFAULT_IMAGE_SIZE = 512

    def __init__(self):
        pass

    def generate_image(
        self,
        pipe,
        prompt,
        output_path="output.png",
        conditioning_image: Optional[object] = None,
        width: int = DEFAULT_IMAGE_SIZE,
        height: int = DEFAULT_IMAGE_SIZE,
        num_inference_steps: int = 30,
        controlnet_conditioning_scale: float = 1.0,
    ):
        """Generates an image and optionally applies ControlNet conditioning."""

        call_kwargs = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
        }

        if conditioning_image is not None:
            call_kwargs["image"] = conditioning_image
            call_kwargs["controlnet_conditioning_scale"] = controlnet_conditioning_scale

        image = pipe(**call_kwargs).images[0]
        image.save(output_path)
