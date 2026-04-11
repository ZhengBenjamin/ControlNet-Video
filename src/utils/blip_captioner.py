from pathlib import Path
from typing import List, Sequence

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


class BLIPCaptioner:
    """Generate captions for images using BLIP model"""
    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        device: str | None = None) -> None:
        """Initialize captioner with model name and device"""
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = None
        self._model = None

    def _lazy_load(self) -> None:
        """Lazy load model and processor on first use"""
        if self._processor is not None and self._model is not None:
            return

        print(f"[caption] Loading BLIP model: {self.model_name} on {self.device}")
        self._processor = BlipProcessor.from_pretrained(self.model_name)
        self._model = BlipForConditionalGeneration.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

    def _open_rgb(self, image_path: Path) -> Image.Image:
        """Open image and convert to RGB"""
        with Image.open(image_path) as image:
            return image.convert("RGB")

    def caption_images(self, image_paths: Sequence[str | Path], max_new_tokens: int = 24) -> List[str]:
        """Generate captions for multiple images"""
        self._lazy_load()
        assert self._processor is not None
        assert self._model is not None

        pil_images = [self._open_rgb(Path(path)) for path in image_paths]
        inputs = self._processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)

        captions = self._processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [caption.strip() for caption in captions]

    def caption_image(self, image_path: str | Path, max_new_tokens: int = 24) -> str:
        """Generate caption for a single image"""
        return self.caption_images([image_path], max_new_tokens=max_new_tokens)[0]
