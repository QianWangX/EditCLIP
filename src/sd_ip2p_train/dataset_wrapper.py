import random
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import CLIPTokenizer, CLIPImageProcessor
from torchvision import transforms
from PIL import Image
import numpy as np


class InstructPix2PixHFDataset(torch.utils.data.Dataset):
    """
    A generalized wrapper around a Hugging Face dataset (e.g. "fusing/instructpix2pix-1000-samples" or ImagenHub)
    that uses a predefined column mapping if available. Returns items in the structure expected by IP-Adapter.
    """
    def __init__(
        self,
        dataset,
        tokenizer,
        image_processor,
        resolution=256,
        original_image_column=None,
        edited_image_column=None,
        edit_prompt_column=None,
        dataset_id=None
    ):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.resolution = resolution
        
        self.original_image_column = original_image_column if original_image_column is not None else "input_image"
        self.edited_image_column = edited_image_column if edited_image_column is not None else "edited_image"
        self.edit_prompt_column = edit_prompt_column if edit_prompt_column is not None else "edit_prompt"
        
        self.train_transforms = transforms.Compose(
            [
                transforms.RandomCrop(self.resolution),
                transforms.RandomHorizontalFlip(),
            ]
        )

    def convert_to_np(self, image, resolution):
        image = image.convert("RGB").resize((resolution, resolution))
        return np.array(image).transpose(2, 0, 1)
        
    def preprocess_images(self, examples):
        original_images = self.convert_to_np(examples[self.original_image_column], self.resolution)
        edited_images = self.convert_to_np(examples[self.edited_image_column], self.resolution)
        # We need to ensure that the original and the edited images undergo the same
        # augmentation transforms.
        images = np.concatenate([original_images, edited_images])
        images = torch.tensor(images)
        images = 2 * (images / 255) - 1
        return self.train_transforms(images)
    
    def preprocess_images_for_clip(self, original_images, edited_images):
        
        original_images = (original_images + 1) / 2
        edited_images = (edited_images + 1) / 2 

        input_image_input = self.image_processor(images=original_images, return_tensors="pt", do_rescale=False)
        edited_image_input = self.image_processor(images=edited_images, return_tensors="pt", do_rescale=False)
        
        concatenated_pixel_values = torch.cat([input_image_input.pixel_values, edited_image_input.pixel_values], dim=1)
        
        concatenated_input = {"pixel_values": concatenated_pixel_values[0]}
        
        return concatenated_input
    
    def tokenize_captions(self, captions):
        inputs = self.tokenizer(
            captions, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        examples = self.dataset[idx]
        # Read images.
        raw_orig = examples[self.original_image_column]
        raw_edit = examples[self.edited_image_column]
        edit_prompt = examples[self.edit_prompt_column]

        output_images = self.preprocess_images(examples)
        # Since the original and edited images were concatenated before
        # applying the transformations, we need to separate them and reshape
        # them accordingly.
        original_images, edited_images = output_images.chunk(2)
        original_images = original_images.reshape(3, self.resolution, self.resolution)  # change from 4 to 3 dim
        edited_images = edited_images.reshape(3, self.resolution, self.resolution)

        # Collate the preprocessed images into the `examples`.
        examples["original_pixel_values"] = original_images
        examples["edited_pixel_values"] = edited_images
        
        # Preprocess the images for CLIP.
        concatenated_input = self.preprocess_images_for_clip(original_images, edited_images)
        
        examples["concatenated_pixel_values"] = concatenated_input["pixel_values"]

        # Preprocess the captions.
        captions = edit_prompt
        examples["input_ids"] = self.tokenize_captions(captions)

        return {
            "original_pixel_values": examples["original_pixel_values"],
            "edited_pixel_values": examples["edited_pixel_values"],
            "input_ids": examples["input_ids"],
            "concatenated_pixel_values": examples["concatenated_pixel_values"],
        }
        
        