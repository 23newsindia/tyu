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

            # Verify required files exist
            required_files = {
                "vae": ["config.json", "diffusion_pytorch_model.safetensors"],
                "text_encoder": ["config.json", "pytorch_model.bin"],
                "transformer": ["config.json", "diffusion_pytorch_model.safetensors"],
                "tokenizer": ["tokenizer_config.json", "vocab.json", "merges.txt"],
                "scheduler": ["scheduler_config.json"]
            }

            for folder, files in required_files.items():
                folder_path = os.path.join(model_path, folder)
                if not os.path.exists(folder_path):
                    raise ValueError(f"Missing required folder: {folder}")
                for file in files:
                    file_path = os.path.join(folder_path, file)
                    if not os.path.exists(file_path):
                        # Try alternative file format
                        alt_file = file.replace(".safetensors", ".bin")
                        alt_path = os.path.join(folder_path, alt_file)
                        if not os.path.exists(alt_path):
                            raise ValueError(f"Missing required file: {file} in {folder}")

            # Load components
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            text_encoder = AutoModelForCausalLM.from_pretrained(text_encoder_path)
            vae = AutoencoderKL.from_pretrained(vae_path)
            transformer = UNet2DConditionModel.from_pretrained(transformer_path)
            scheduler = DPMSolverMultistepScheduler.from_pretrained(scheduler_path)

        else:
            # Try loading from Hugging Face Hub
            tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
            text_encoder = AutoModelForCausalLM.from_pretrained(model_path, subfolder="text_encoder")
            vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
            transformer = UNet2DConditionModel.from_pretrained(model_path, subfolder="transformer")
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
