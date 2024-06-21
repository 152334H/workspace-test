import torch
import jax
from transformers import AutoModel

def check_cuda():
    """
    Check if CUDA is available and return the status.
    """
    cuda_status = torch.cuda.is_available()
    return cuda_status

def torch_to_cuda(tensor):
    """
    Move a PyTorch tensor to CUDA device if available.
    """
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor

def jax_to_cuda(array):
    """
    Placeholder for moving a JAX array to CUDA device.
    Currently, JAX does not support direct CUDA manipulation, but this function
    serves as a placeholder for future compatibility.
    """
    return array

def load_transformer_model(model_name):
    """
    Load a transformer model using the specified model name.
    The model is moved to CUDA device if available.
    """
    model = AutoModel.from_pretrained(model_name)
    if torch.cuda.is_available():
        model.cuda()
    return model
