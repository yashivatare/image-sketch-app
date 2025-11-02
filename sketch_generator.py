import os
import time
from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image
# --- CHANGE: Import the AutoencoderKL class for memory optimization ---
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, LCMScheduler, AutoencoderKL

class SketchGenerator:
    def __init__(
        self,
        device: Union[str, torch.device] = None,
        output_dir: str = "outputs",
    ) -> None:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        # --- CHANGE: Load a smaller, more memory-efficient VAE ---
        # This is a key fix for low VRAM GPUs.
        # CORRECT LINE
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=dtype
        )

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=dtype,
        )

        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            vae=vae,  # --- CHANGE: Pass the efficient VAE into the pipeline ---
            torch_dtype=dtype,
            safety_checker=None,
        )

        pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        pipe.fuse_lora()
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        
        pipe.enable_model_cpu_offload()
        
        # VAE slicing is still beneficial even with CPU offload
        if self.device.type == "cuda":
            pipe.enable_vae_slicing()

        self.pipe = pipe.to(self.device)


    def _create_canny_conditioning_image(self, image: Image.Image) -> Image.Image:
        """Creates a clean Canny edge map."""
        rgb_array = np.array(image.convert("RGB"))
        gray_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray_array, (3, 3), 0)
        v = float(np.median(blurred))
        lower = int(max(0, 0.7 * v))
        upper = int(min(255, 1.3 * v))
        edges = cv2.Canny(blurred, lower, upper)
        return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))

    def generate_sketch(
        self,
        input_image: Union[str, Image.Image],
        prompt: str,
        negative_prompt: str = "",
        strength: float = 0.6,
        guidance_scale: float = 1.0,
        num_inference_steps: int = 4,
        controlnet_conditioning_scale: float = 1.0,
        seed: int = None,
    ) -> str:
        base_image = Image.open(input_image).convert("RGB") if isinstance(input_image, str) else input_image.convert("RGB")
        
        # --- CHANGE: Add a resolution cap to prevent crashes from huge images ---
        MAX_RESOLUTION = 1024
        if max(base_image.width, base_image.height) > MAX_RESOLUTION:
            base_image.thumbnail((MAX_RESOLUTION, MAX_RESOLUTION))

        control_image = self._create_canny_conditioning_image(base_image)
        
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=base_image,
            control_image=control_image,
            strength=strength,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

        out_image: Image.Image = result.images[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(self.output_dir, f"sketch_{timestamp}.png")
        out_image.save(out_path)
        
        return out_path