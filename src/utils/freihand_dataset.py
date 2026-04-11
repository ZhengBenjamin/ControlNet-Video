import json
import torch
import numpy as np

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from typing import Any

class FreiHandDataset(Dataset):
    """PyTorch Dataset for FreiHAND data with ControlNet conditioning"""
    def __init__(self, data_root: Path, tokenizer: Any, size: int = 512) -> None:
        """Initialize dataset with metadata from jsonl file"""
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.size = size

        with open(data_root / "metadata.jsonl", "r") as f:
            self.metadata = [json.loads(line) for line in f]

    def __len__(self) -> int:
        """Return total number of samples"""
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.metadata[idx]

        image_path = self.data_root / item["file_name"]
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.size, self.size), Image.BILINEAR)

        cond_path = self.data_root / item["conditioning_image"]
        conditioning_image = Image.open(cond_path).convert("RGB")
        conditioning_image = conditioning_image.resize((self.size, self.size), Image.BILINEAR)

        # Tokenize caption
        caption = item["caption"]
        text_inputs = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "pixel_values": torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1,
            "conditioning_pixel_values": torch.from_numpy(np.array(conditioning_image)).permute(2, 0, 1).float() / 127.5 - 1,
            "input_ids": text_inputs.input_ids.squeeze(),
        }