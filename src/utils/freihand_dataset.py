import json
import torch
import numpy as np

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class FreiHandDataset(Dataset):
    def __init__(self, data_root: Path, tokenizer, size=512):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.size = size

        with open(data_root / "metadata.jsonl", "r") as f:
            self.metadata = [json.loads(line) for line in f]

        # store in mem
        self.image_cache = {}
        self.cond_cache = {}
        for item in self.metadata:
            image_paht = self.data_root / item["file_name"]
            cond_path = self.data_root / item["conditioning_image"]
            image = Image.open(image_paht).convert("RGB").resize((size, size), Image.BILINEAR)
            conditioning_image = Image.open(cond_path).convert("RGB").resize((size, size), Image.BILINEAR)
            self.image_cache[item["file_name"]] = image
            self.cond_cache[item["conditioning_image"]] = conditioning_image

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]

        # uncomment if no cache, mem constraint

        # image_path = self.data_root / item["file_name"]
        # image = Image.open(image_path).convert("RGB")
        # image = image.resize((self.size, self.size), Image.BILINEAR)

        # cond_path = self.data_root / item["conditioning_image"]
        # conditioning_image = Image.open(cond_path).convert("RGB")
        # conditioning_image = conditioning_image.resize((self.size, self.size), Image.BILINEAR)

        image = self.image_cache[item["file_name"]]
        conditioning_image = self.cond_cache[item["conditioning_image"]]

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