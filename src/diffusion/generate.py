from typing import Optional, Any


class Generator:
    """Generate images using diffusion pipelines"""

    DEFAULT_IMAGE_SIZE = 512

    def __init__(self) -> None:
        """Initialize generator"""
        pass

    def generate_image(
        self,
        pipe: Any,
        prompt: str,
        output_path: str = "output.png",
        conditioning_image: Optional[Any] = None,
        width: int = DEFAULT_IMAGE_SIZE,
        height: int = DEFAULT_IMAGE_SIZE,
        num_inference_steps: int = 30,
        controlnet_conditioning_scale: float = 1.0) -> None:
        """Generate image with optional ControlNet conditioning and save to file"""

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
