from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler
import torch
import os

def load_models(model_path, device="cuda"):
    """Load and return all required models for SANA."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    text_encoder = AutoModelForCausalLM.from_pretrained(model_path)
    vae = AutoencoderKL.from_pretrained(model_path)
    transformer = UNet2DConditionModel.from_pretrained(model_path)
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path)
    
    # Move models to device
    text_encoder = text_encoder.to(device)
    vae = vae.to(device)
    transformer = transformer.to(device)
    
    return tokenizer, text_encoder, vae, transformer, scheduler

def cleanup_models(*models):
    """Clean up models and free CUDA memory."""
    for model in models:
        if model is not None:
            del model
    torch.cuda.empty_cache()