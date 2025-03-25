# 🖼️ EditCLIP: Representation Learning for Image Editing

![Teaser](./assets/teaser_editclip.png)

This is the implementation of paper **EditCLIP: Representation Learning for Image Editing**. 

This repo is still under testing, please stay tuned!

## Prepare environment
```bash
python3 -m venv .env
source .env/bin/activate
pip install open_clip_torch open_clip_torch[training]
pip install accelerate==1.0.1 transformers==4.49.0
pip install git+https://github.com/huggingface/diffusers
```

## Dataset preparation
Please download training dataset from [IP2P_filtered](https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered).

## EditCLIP weight and training
We provide the pretrained EditCLIP checkpoints for VIT-L-14 [here](https://huggingface.co/QWW/EditCLIP). You can download the `model.safetensors` and `config.json`, and then put them under `clip_ckpt/editclip_vit_l_14`.

If you want to train EditCLIP from scratch, please refer to `scripts/train_editclip.sh` for single-GPU or multi-GPU training. For reference, we trained with VIT-L-14 for 40 epochs on 4 NVIDIA A100 GPUs. By default, the training logs and checkpoints will be saved under folder `logs`.

After training is done, please use `utils/convert_clip_original_pytorch_to_hf.py` to convert the pytorch model to Huggingface format. Save the Huggingface model under folder `clip_ckpt/YOUR/HF/CLIP/PATH`. 

`python utils/convert_clip_original_pytorch_to_hf.py --pytorch_dump_folder_path clip_ckpt/clip_model --checkpoint_path PATH/TO/PYTORCH/CKPT --config_path HF/VIT/CONFIG/FILE --in_channels 6` 

You should also manually download the preprocessor and tokenizer configuration files to your CLIP folder. Here we show an example under `clip_ckpt/editclip_vit_l_14`. You can set the `pytorch_dump_folder_path` to this folder and use `clip_ckpt/editclip_vit_l_14/hf_vit_l_14_config.json` as the `config_path`. After runnning the conversion command, `model.safetensors` and `config.json` should be generated under this folder. Please manually add `"num_channels": 6` under `vision_config` in `config.json`.

## EditCLIP with IP2P
We provide the pretrained IP2P pipeline with EditCLIP VIT-L-14 checkpoint [here](https://huggingface.co/QWW/EditCLIP-IP2P). 

If you want to train IP2P with EditCLIP from scratch, you can refer to the command below. We use the same IP2P-filtered dataset for training the IP2P model with EditCLIP. You can check the training command in `scripts/train_editclip.sh`

By default, the checkpoints will be saved under `instruct-pix2pix-model`. You can evaluate the model by the following command:
`python src/sd_ip2p_train/eval_ip2p_transfer.py --pretrained_clip_path PATH/TO/YOUR/HF/EDITCLIP/PATH --is_use_projection_model --ckpt_dir PATH/TO/YOUR/EDITCLIP/IP2P/DIR --input_image_path PATH/TO/YOUR/INPUT/IMAGE --edited_image_path PATH/TO/YOUR/EDITED/IMAGE --query_image_path PATH/TO/YOUR/QUERY/IMAGE`

You can also use the images we provide in the assets folder. 
