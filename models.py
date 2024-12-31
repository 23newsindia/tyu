from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler
import torch
import os

def load_models(model_path, device="cuda"):
    """Load and return all required models for SANA."""
    try:
        # First try loading from local path
        if os.path.exists(model_path):
            # Check for model files
            vae_path = os.path.join(model_path, "vae")
            text_encoder_path = os.path.join(model_path, "text_encoder")
            transformer_path = os.path.join(model_path, "transformer")
            tokenizer_path = os.path.join(model_path, "tokenizer")
            scheduler_path = os.path.join(model_path, "scheduler")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
                use_fast=True
            )

            # Load text encoder
            text_encoder = AutoModelForCausalLM.from_pretrained(
                text_encoder_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                use_safetensors=True
            )

            # Load VAE with special handling
            vae = AutoencoderKL.from_pretrained(
                vae_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                use_safetensors=True,
                low_cpu_mem_usage=False,
                device_map=None
            )

            # Load transformer
            transformer = UNet2DConditionModel.from_pretrained(
                transformer_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                use_safetensors=True
            )

            # Load scheduler
            scheduler = DPMSolverMultistepScheduler.from_pretrained(scheduler_path)

        else:
            # Load from Hugging Face Hub
            tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
            text_encoder = AutoModelForCausalLM.from_pretrained(
                model_path, 
                subfolder="text_encoder",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                use_safetensors=True
            )
            vae = AutoencoderKL.from_pretrained(
                model_path,
                subfolder="vae",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                use_safetensors=True,
                low_cpu_mem_usage=False,
                device_map=None
            )
            transformer = UNet2DConditionModel.from_pretrained(
                model_path,
                subfolder="transformer",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                use_safetensors=True
            )
            scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler")

        # Move models to device
        text_encoder = text_encoder.to(device)
        vae = vae.to(device)
        transformer = transformer.to(device)

        return tokenizer, text_encoder, vae, transformer, scheduler

    except Exception as e:
        raise ValueError(f"Error loading SANA models: {str(e)}\nPlease ensure all required model files are present in the correct structure.")


def cleanup_models(*models):
    """Clean up models and free CUDA memory."""
    for model in models:
        if model is not None:
            del model
    torch.cuda.empty_cache()
