import os
import torch
import numpy as np

def get_model_path(model_path, base_path, model_list):
    """Resolve the full model path."""
    if model_path in model_list:
        return model_path
    return os.path.join(base_path, "models", model_path)

def process_latents(vae, latents):
    """Process latents into final image."""
    latents = 1 / vae.config.scaling_factor * latents
    image = vae.decode(latents).sample
    
    # Convert to numpy array
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).round().astype("uint8")
    
    return image

def prepare_embeddings(text_encoder, tokenizer, prompt, negative_prompt, device, max_length=300):
    """Prepare text embeddings for both prompt and negative prompt."""
    # Process prompt
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt"
    )

    text_input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)

    prompt_embeds = text_encoder(text_input_ids, attention_mask=attention_mask)
    prompt_embeds = prompt_embeds[0]

    # Process negative prompt
    neg_text_inputs = tokenizer(
        negative_prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt"
    )

    neg_input_ids = neg_text_inputs.input_ids.to(device)
    neg_attention_mask = neg_text_inputs.attention_mask.to(device)

    negative_prompt_embeds = text_encoder(neg_input_ids, attention_mask=neg_attention_mask)
    negative_prompt_embeds = negative_prompt_embeds[0]

    return {
        "prompt_embeds": prompt_embeds,
        "prompt_attention_mask": attention_mask,
        "negative_prompt_embeds": negative_prompt_embeds,
        "negative_prompt_attention_mask": neg_attention_mask
    }