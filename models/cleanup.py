import torch

def cleanup_models(*models):
    """Clean up models and free CUDA memory."""
    for model in models:
        if model is not None:
            del model
    torch.cuda.empty_cache()
