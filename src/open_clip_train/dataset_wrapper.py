import random
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import CLIPTokenizer, CLIPImageProcessor
from torchvision import transforms
from PIL import Image


class InstructPix2PixHFDataset(torch.utils.data.Dataset):
    """
    A generalized wrapper around a Hugging Face dataset (e.g. "fusing/instructpix2pix-1000-samples" or ImagenHub)
    that uses a predefined column mapping if available. Returns items in the structure expected by IP-Adapter.
    """
    def __init__(
        self,
        dataset,
        tokenizer,
        image_transform,
        resolution=512,
        original_image_column=None,
        edited_image_column=None,
        edit_prompt_column=None,
        edited_prompt_column=None,
        original_prompt_column=None,
        dataset_id=None
    ):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        
        self.edit_transform = image_transform


        self.original_image_column = original_image_column if original_image_column is not None else "input_image"
        self.edited_image_column = edited_image_column if edited_image_column is not None else "edited_image"
        self.edit_prompt_column = edit_prompt_column if edit_prompt_column is not None else "edit_prompt"
        self.edited_prompt_column = edited_prompt_column if edited_prompt_column is not None else "edited_prompt"
        self.original_prompt_column = original_prompt_column if original_prompt_column is not None else "input_prompt"


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        # Read images.
        raw_orig = row[self.original_image_column]
        raw_edit = row[self.edited_image_column]
        edit_prompt = row[self.edit_prompt_column]
        edited_prompt = row[self.edited_prompt_column]
        original_prompt = row[self.original_prompt_column]

        if not isinstance(raw_orig, Image.Image):
            # Assume raw_orig is a file path or URL.
            raw_orig = Image.open(raw_orig)
        if not isinstance(raw_edit, Image.Image):
            raw_edit = Image.open(raw_edit)

        # Transform images.
        main_tensor = self.edit_transform(raw_orig.convert("RGB"))
        edit_tensor = self.edit_transform(raw_edit.convert("RGB"))

        # Tokenize text.
        edit_prompt_id = self.tokenizer(edit_prompt)[0]
        edited_prompt_id = self.tokenizer(edited_prompt)[0]
        original_prompt_id = self.tokenizer(original_prompt)[0]

        return {
            "original_image": main_tensor,             # e.g. [3, 512, 512]
            "edited_image": edit_tensor,          # e.g. [3, 512, 512]
            "edit_prompt_id": edit_prompt_id,         # e.g. [seq_len]
            "edited_prompt_id": edited_prompt_id,     # e.g. [seq_len]
            "original_prompt_id": original_prompt_id  # e.g. [seq_len]
        }