import torch
import torch.nn.functional as F
import os 
from typing import Optional

from src.utils import FreiHandDataset
from tqdm import tqdm
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from src.config import MODELS_DIR, DATA_DIR
from pathlib import Path
from accelerate import Accelerator
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from PIL import Image 

# NCCL env settings needed to work on markov cluster, not too sure why, crashes w/o
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

class ControlNetTrainer:
    def __init__(self, model_dir: Path = MODELS_DIR / "controlnet-freihand-sd15"):
        self.model_dir = model_dir
        self.training_state_path = self.model_dir / "training_state.pt"
        self.accelerator = Accelerator(
            mixed_precision="fp16",
            gradient_accumulation_steps=4,
            device_placement=True,
            split_batches=True,
        )

    def train(
            self, 
            data_root: Path = DATA_DIR / "freihand_controlnet", 
            num_epochs=1, 
            batch_size=1, 
            learning_rate=1e-5, 
            max_samples=10000,
            resume: Optional[bool] = True):
        
        print(f"Training on {self.accelerator.num_processes} device(s)")
        if self.accelerator.is_main_process:
            print(f"Main process running on device: {self.accelerator.device}")
        
        sd15_source, use_local = self.get_model_source()

        # Load models
        vae = AutoencoderKL.from_pretrained(
            sd15_source, subfolder="vae",
            torch_dtype=torch.float16,
            local_files_only=use_local
        )
        unet = UNet2DConditionModel.from_pretrained(
            sd15_source, subfolder="unet",
            torch_dtype=torch.float16,
            local_files_only=use_local
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            sd15_source, subfolder="tokenizer",
            local_files_only=use_local
        )

        # Resume if has existing weights
        if resume and (self.model_dir / "config.json").exists():
            print(f"Resume from weights {self.model_dir}")
            controlnet = ControlNetModel.from_pretrained(
                self.model_dir, torch_dtype=torch.float16, local_files_only=True
            )
        else:
            controlnet = ControlNetModel.from_unet(unet)

        controlnet.enable_gradient_checkpointing()

        vae.requires_grad_(False) # freeze vae and unet, only train controlnet
        unet.requires_grad_(False)
        
        vae = vae.to(self.accelerator.device, dtype=torch.float16)
        dataset = FreiHandDataset(data_root, tokenizer, size=256) # load dataset

        if max_samples:
            dataset.metadata = dataset.metadata[:max_samples]

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(controlnet.parameters(), lr=learning_rate)

        lr_scheduler = get_scheduler(
            "constant",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=len(dataloader) * num_epochs,
        )

        noise_scheduler = DDPMScheduler.from_pretrained(
            sd15_source, subfolder="scheduler", local_files_only=use_local
        )

        text_encoder = CLIPTextModel.from_pretrained( # text encoder
            sd15_source, subfolder="text_encoder",
            torch_dtype=torch.float16,
            local_files_only=use_local
        )

        text_encoder.requires_grad_(False) # freeze txt encoder
        text_encoder = text_encoder.to(self.accelerator.device, dtype=torch.float16)

        vae.eval()
        unet.eval()
        text_encoder.eval()

        # get params from args 
        controlnet, unet, optimizer, dataloader, lr_scheduler = self.accelerator.prepare(
            controlnet, unet, optimizer, dataloader, lr_scheduler 
        )

        # load wieghts if resume
        if resume and self.training_state_path.exists():
            print("Loading weights from previous training state")
            training_state = torch.load(self.training_state_path, map_location=self.accelerator.device)
            optimizer.load_state_dict(self._deserialize_state(training_state["optimizer"]))
            lr_scheduler.load_state_dict(self._deserialize_state(training_state["lr_scheduler"]))
    
        controlnet.train()
        for epoch in range(num_epochs):
            for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
                with self.accelerator.accumulate(controlnet):
                    # Move batch to device
                    batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}

                    with torch.no_grad():
                        latents = vae.encode(batch["pixel_values"].to(dtype=torch.float16)).latent_dist.sample() # encode latent 

                    latents = latents * vae.config.scaling_factor

                    controlnet_image = batch["conditioning_pixel_values"] # encode condition
                    
                    with torch.no_grad():
                        encoder_hidden_states = text_encoder(batch["input_ids"])[0] # text encoder

                    noise = torch.randn_like(latents) # sample step
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    down_block_res_samples, mid_block_res_sample = controlnet( # forward
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=controlnet_image,
                        return_dict=False,
                    )

                    noise_pred = unet( # forward unet with controlnet res
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    ).sample

                    loss = F.mse_loss(noise_pred.to(dtype=torch.float16), noise.to(dtype=torch.float16))

                    self.accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

        # Save model
        if self.accelerator.is_main_process:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            training_state = {
                "optimizer": self._serialize_state(optimizer.state_dict()),
                "lr_scheduler": self._serialize_state(lr_scheduler.state_dict()),
            }
            
            torch.save(training_state, self.training_state_path)
            controlnet = self.accelerator.unwrap_model(controlnet)
            controlnet.save_pretrained(self.model_dir)
        
        if hasattr(self.accelerator, "end_training"):
            self.accelerator.end_training()
        
        print("Training completed!")

    def generate_image(self, prompt: str, conditioning_image_path: str, output_path: str = "output_controlnet.png"):
        
        device = torch.device("cuda:0") # some reason crash if using multiple gpus inf, temp fix
        torch.cuda.set_device(device)
        sd15_source, use_local = self.get_model_source()

        controlnet = ControlNetModel.from_pretrained(self.model_dir, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            sd15_source,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            local_files_only=use_local
        )
        pipe = pipe.to(device)
        # pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()

        conditioning_image = Image.open(conditioning_image_path).convert("RGB").resize((512, 512)) # cond img

        # Generate
        image = pipe(
            prompt=prompt,
            image=conditioning_image,
            num_inference_steps=20,
            controlnet_conditioning_scale=1.0,
        ).images[0]

        image.save(output_path)
        print(f"Image saved to {output_path}")

    def get_model_source(self):
        """Returns local source or get from remote repo if not"""

        repo_cache = MODELS_DIR / "models--runwayml--stable-diffusion-v1-5"
        refs_main = repo_cache / "refs" / "main"
        
        if refs_main.exists():
            revision = refs_main.read_text(encoding="utf-8").strip()
            snapshot_dir = repo_cache / "snapshots" / revision
            if snapshot_dir.exists():
                return snapshot_dir, True

        snapshots_dir = repo_cache / "snapshots"
        if snapshots_dir.exists():
            snapshot_candidates = sorted(p for p in snapshots_dir.iterdir() if p.is_dir())
            if snapshot_candidates:
                return snapshot_candidates[0], True

        return "runwayml/stable-diffusion-v1-5", False
    
    def _serialize_state(self, value):
        """Recrusively move tensors to CPU and convert to float16 for serialization"""
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                return value.detach().cpu().to(torch.float16)
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: self._serialize_state(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._serialize_state(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._serialize_state(item) for item in value)
        return value

    def _deserialize_state(self, value):
        """Recrusively move tensors to accelerator device and keep float16 precision"""
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                return value.to(dtype=torch.float16, device=self.accelerator.device)
            return value.to(device=self.accelerator.device)
        if isinstance(value, dict):
            return {key: self._deserialize_state(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._deserialize_state(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._deserialize_state(item) for item in value)
        return value