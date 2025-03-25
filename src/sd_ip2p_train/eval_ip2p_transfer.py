import argparse
import math
import os
import shutil
from contextlib import nullcontext
from pathlib import Path

import accelerate
import datasets
import numpy as np
import PIL
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPTokenizer, AutoProcessor

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel

from pipeline_ip2p_transfer import StableDiffusionInstructPix2PixTransferPipeline
from PIL import Image

from datasets import concatenate_datasets
from projection_model import ImageProjModel

import time

def pil_to_tensor(pil_image, resize=(256, 256)):
    if isinstance(pil_image, str):
        pil_image = Image.open(pil_image).convert("RGB").resize(resize)
        
    image_tensor = transforms.ToTensor()(pil_image).unsqueeze(0)
    
    return image_tensor

def save_to_grid(image1, image2, image3, image4, output_path, resize=(256, 256)):
    width, height = resize
    image1 = image1.resize(resize)
    image2 = image2.resize(resize)
    image3 = image3.resize(resize)
    image4 = image4.resize(resize)

    # Create a blank canvas for the 2x2 grid
    grid_width = 2 * width
    grid_height = 2 * height
    grid_image = Image.new("RGB", (grid_width, grid_height))

    # Paste the images into the grid
    grid_image.paste(image1, (0, 0))                      # Top-left
    grid_image.paste(image2, (width, 0))                 # Top-right
    grid_image.paste(image3, (0, height))                # Bottom-left
    grid_image.paste(image4, (width, height))            # Bottom-right
    
    grid_image.save(output_path)


def encode_image_transfer_clip(image_encoder, image, device, num_images_per_prompt, output_hidden_states=None,
                               is_use_projection_model=False, image_proj_model=None):
    
    image_encoder = image_encoder.to(device=device)
    image = image.to(device=device)
    
    dtype = next(image_encoder.parameters()).dtype
        
    image_embeds = image_encoder(image)[0]  # .image_embeds
    
    if is_use_projection_model:
        assert image_proj_model is not None
        image_proj_model = image_proj_model.to(device=device)
        
        image_embeds = image_proj_model(image_embeds)
        
        image_embeds = image_embeds.to(dtype=dtype)
    
    image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
    uncond_image_embeds = torch.zeros_like(image_embeds)
    
    return image_embeds, uncond_image_embeds


def log_validation(
    pipeline,
    image_encoder,
    image_processor,
    input_images,
    edited_images,
    query_images,
    device,
    generator,
    global_step,
    output_dir, 
    guidance_scale=7,
    image_guidance_scale=1.5,
    is_use_projection_model=False,
    image_proj_model=None,
    resolution=256,
):
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    
    output_images = []

    # run inference
    for input_image, edited_image, query_image in zip(input_images, edited_images, query_images):
        input_image_name = os.path.basename(input_image).split(".")[0]
        edited_image_name = os.path.basename(edited_image).split(".")[0]
        query_image_name = os.path.basename(query_image).split(".")[0]
        concat_name = f"{input_image_name}_{edited_image_name}_{query_image_name}"
        
        input_image = pil_to_tensor(input_image, resize=(resolution, resolution))
        edited_image = pil_to_tensor(edited_image, resize=(resolution, resolution))
        query_image = Image.open(query_image).resize((resolution, resolution))
        
        input_image_input = image_processor(images=input_image, return_tensors="pt", do_rescale=False)
        edited_image_input = image_processor(images=edited_image, return_tensors="pt", do_rescale=False)
        
        concatenated_pixel_values = torch.cat([input_image_input.pixel_values, edited_image_input.pixel_values], dim=1)
        
        prompt_embeds, negative_prompt_embeds = encode_image_transfer_clip(image_encoder, concatenated_pixel_values, device, 1,
                                                                           is_use_projection_model=is_use_projection_model,
                                                                           image_proj_model=image_proj_model)

        with torch.no_grad():
            output_image = pipeline(
                image=query_image,  # We use the query image as the input image, which can be different from the input image!
                input_image=input_image,
                edited_image=edited_image,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                num_inference_steps=50,
                image_guidance_scale=image_guidance_scale,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]
            
            
        output_images.append(output_image)
        
        # save to path
        input_image = input_image[0].cpu().numpy().transpose(1, 2, 0)
        input_image = Image.fromarray((input_image * 255).astype(np.uint8))
        input_image.save(os.path.join(output_dir, f"input_image_{global_step}_{concat_name}.png"))
        edited_image = edited_image[0].cpu().numpy().transpose(1, 2, 0)
        edited_image = Image.fromarray((edited_image * 255).astype(np.uint8))
        edited_image.save(os.path.join(output_dir, f"edited_image_{global_step}_{concat_name}.png"))
        query_image.save(os.path.join(output_dir, f"query_image_{global_step}_{concat_name}.png"))
        output_image.save(os.path.join(output_dir, f"output_image_{global_step}_{concat_name}.png"))
        
        save_to_grid(input_image, edited_image, query_image, output_image, os.path.join(output_dir, f"grid_image_{global_step}_{concat_name}_g_{guidance_scale}_image_g_{image_guidance_scale}.png"))

        
def collate_fn(examples):
    original_pixel_values = torch.stack([torch.tensor(example["original_pixel_values"]) for example in examples])
    original_pixel_values = original_pixel_values.to(memory_format=torch.contiguous_format).float()
    edited_pixel_values = torch.stack([torch.tensor(example["edited_pixel_values"]) for example in examples])
    edited_pixel_values = edited_pixel_values.to(memory_format=torch.contiguous_format).float()
    concatenated_pixel_values = torch.stack([torch.tensor(example["concatenated_pixel_values"]) for example in examples])
    concatenated_pixel_values = concatenated_pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([torch.tensor(example["input_ids"]) for example in examples])
    return {
        "original_pixel_values": original_pixel_values,
        "edited_pixel_values": edited_pixel_values,
        "concatenated_pixel_values": concatenated_pixel_values,
        "input_ids": input_ids,
    }
        
def main(args, guidance_scale=7, image_guidance_scale=1.5,):        
    
    pretrained_model_name_or_path = "timbrooks/instruct-pix2pix"
    unet_ckpt_path = ckpt_dir
    is_use_projection_model = args.is_use_projection_model
    resolution = args.resolution

    pretrained_clip_path = args.pretrained_clip_path
    image_processor = AutoProcessor.from_pretrained(pretrained_clip_path)
    
    weight_dtype = torch.float16
    seed = 0
    global_step = int(unet_ckpt_path.split("-")[-1])
    device = torch.device("cuda:0")
    save_dir = os.path.dirname(unet_ckpt_path)
    
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)
    
    # load diffusers style into model
    unet = UNet2DConditionModel.from_pretrained(unet_ckpt_path, subfolder="unet")
    unet.register_to_config(**unet.config)
    unet.load_state_dict(unet.state_dict())
    
    image_encoder = CLIPModel.from_pretrained(
        pretrained_clip_path
    ).vision_model
    
    unet = unet.to(device=device, dtype=weight_dtype)
    image_encoder = image_encoder.to(device=device, dtype=weight_dtype)

    if is_use_projection_model:
        image_proj_model = ImageProjModel(
            cross_attention_dim=unet.config.cross_attention_dim,
            clip_embeddings_dim=image_encoder.config.hidden_size,
            clip_extra_context_tokens=1,
        )
        image_proj_model_path = os.path.join(unet_ckpt_path, "image_proj_model.pt")
        image_proj_model.load_state_dict(torch.load(image_proj_model_path))
        image_proj_model = image_proj_model.to(device=device, dtype=weight_dtype)
        
    else:
        image_proj_model = None
    
    
    ############# Load evaluation data
    input_images = [args.input_image_path]
    
    edited_images = [args.edited_image_path]
    
    query_images = [args.query_image_path]
    
    pipeline = StableDiffusionInstructPix2PixTransferPipeline.from_pretrained(
        pretrained_model_name_or_path,
        unet=unet,
        torch_dtype=weight_dtype,
    )
    
    generator = torch.Generator(device=device).manual_seed(seed)

    log_validation(
        pipeline,
        image_encoder,
        image_processor,
        input_images,
        edited_images,
        query_images,
        device,
        generator,
        global_step,
        output_folder,  
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale,
        is_use_projection_model=is_use_projection_model,
        image_proj_model=image_proj_model,
        resolution=resolution,
    )
                        
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--pretrained_clip_path", type=str, require=True, help="path to your HF EditCLIP model")
    args.add_argument("--is_use_projection_model", action="store_true", default=False)
    args.add_argument("--resolution", type=int, default=256)
    args.add_argument("--ckpt_dir", type=str, required=True, help="path to your EditCLIP IP2P dir")
    args.add_argument("--input_image_path", type=str, default="assets/input_image.jpg", help="path to your input image")
    args.add_argument("--edited_image_path", type=str, default="assets/edited_image.jpg", help="path to your edited image")
    args.add_argument("--query_image_path", type=str, default="assets/query_image.jpg", help="path to your query image")
    args.add_argument("--output_folder", type=str, default="eval_ip2p_transfer")
    
    args = args.parse_args()

    main(args)