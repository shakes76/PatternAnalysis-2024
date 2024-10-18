from diffusers import DiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DPMSolverMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import torch


repo_id = "stabilityai/stable-diffusion-2-base"

unet_config = UNet2DConditionModel.from_pretrained(repo_id).config
#Retrieved the Untrained Model 
unet = UNet2DConditionModel.from_config(unet_config)

vae_config = AutoencoderKL.from_pretrained(repo_id).config
#Retrieved untrained Autoencoder
vae = AutoencoderKL.from_config(vae_config)

#Models retrieved frol CLIP
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

#Tokenizer retrieved from CLIP
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

#Pipeline
modelPipeline = DiffusionPipeline(unet=unet, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer)