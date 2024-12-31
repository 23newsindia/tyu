import torch
import os
import folder_paths
from .models import load_models, cleanup_models
from .utils import get_model_path, process_latents, prepare_embeddings

class SANATextEncode:
    models = [
        "Efficient-Large-Model/Sana_600M_512px_diffusers",
        "Efficient-Large-Model/Sana_600M_1024px_diffusers",
        "Efficient-Large-Model/Sana_1600M_512px_diffusers",
        "Efficient-Large-Model/Sana_1600M_1024px_diffusers" 
    ]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"default": '', "multiline": True}),
                "negative_prompt": ("STRING", {"default": '', "multiline": True}),
                "model_path": ([f'diffusers/{i}' for i in os.listdir(folder_paths.get_folder_paths("diffusers")[0]) if os.path.isdir(folder_paths.get_folder_paths("diffusers")[0]+f"/{i}")]+SANATextEncode.models,),
            }
        }

    CATEGORY = "sana"
    RETURN_TYPES = ("class",)
    FUNCTION = "encode"

    def encode(self, prompt, negative_prompt, model_path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        model_path = get_model_path(model_path, base_path, self.models)

        # Load models
        tokenizer, text_encoder, _, _, _ = load_models(model_path, device)
        
        # Prepare embeddings
        embeddings = prepare_embeddings(text_encoder, tokenizer, prompt, negative_prompt, device)
        
        # Cleanup
        cleanup_models(text_encoder)
        
        return (embeddings,)

class SANADiffuse:
    models = SANATextEncode.models

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1}),
                "width": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "img2img": (["disable", "enable"],),
                "embeds": ("class",),
                "model_path": ([f'diffusers/{i}' for i in os.listdir(folder_paths.get_folder_paths("diffusers")[0]) if os.path.isdir(folder_paths.get_folder_paths("diffusers")[0]+f"/{i}")]+SANADiffuse.models,),
            },
            "optional": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    CATEGORY = "sana"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"

    def generate(self, seed, steps, width, height, cfg, img2img, embeds, model_path, strength=None, image=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        model_path = get_model_path(model_path, base_path, self.models)

        # Load models
        _, _, vae, transformer, scheduler = load_models(model_path, device)

        # Set generator
        generator = torch.Generator(device=device).manual_seed(seed)

        # Prepare latents
        batch_size = 1
        if img2img == "enable" and image is not None:
            # Process image for img2img
            if isinstance(image, torch.Tensor):
                image = image.to(device)
            else:
                image = torch.from_numpy(image).to(device)
            
            latents = vae.encode(image).latent_dist.sample(generator)
            latents = latents * vae.config.scaling_factor
            
            # Add noise to latents
            noise = torch.randn_like(latents)
            scheduler.set_timesteps(steps)
            latents = scheduler.add_noise(latents, noise, scheduler.timesteps[0])
        else:
            latents = torch.randn(
                (batch_size, transformer.config.in_channels, height // 8, width // 8),
                generator=generator,
                device=device
            )

        # Extract and prepare embeddings
        prompt_embeds = embeds["prompt_embeds"].to(device)
        prompt_attention_mask = embeds["prompt_attention_mask"].to(device)
        negative_prompt_embeds = embeds["negative_prompt_embeds"].to(device)
        negative_prompt_attention_mask = embeds["negative_prompt_attention_mask"].to(device)

        # Prepare scheduler
        scheduler.set_timesteps(steps)
        timesteps = scheduler.timesteps

        # Combine conditional and unconditional embeddings
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        # Denoising loop
        for t in timesteps:
            # Expand latents for classifier free guidance
            latent_model_input = torch.cat([latents] * 2)

            # Get model prediction
            noise_pred = transformer(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                attention_mask=prompt_attention_mask
            ).sample

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg * (noise_pred_text - noise_pred_uncond)

            # Scheduler step
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Process latents into final image
        image = process_latents(vae, latents)
        
        # Cleanup
        cleanup_models(vae, transformer)
        
        return (image,)

NODE_CLASS_MAPPINGS = {
    "SANADiffuse": SANADiffuse,
    "SANATextEncode": SANATextEncode
}