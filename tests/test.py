import pytest
from src.test_package import check_cuda, torch_to_cuda, jax_to_cuda, load_transformer_model

def test_check_cuda():
    assert isinstance(check_cuda(), bool), "check_cuda should return a boolean value"

def test_torch_to_cuda():
    import torch
    tensor = torch.tensor([1, 2, 3])
    if torch.cuda.is_available():
        assert torch_to_cuda(tensor).is_cuda, "Tensor should be moved to CUDA"
    else:
        assert not torch_to_cuda(tensor).is_cuda, "Tensor should not be moved to CUDA if unavailable"

def test_jax_to_cuda():
    import jax.numpy as jnp
    array = jnp.array([1, 2, 3])
    # Currently, JAX does not support direct CUDA manipulation, so this test is a placeholder
    assert jax_to_cuda(array) is array, "jax_to_cuda should return the original array"

def test_load_transformer_model():
    model_name = "bert-base-uncased"
    model = load_transformer_model(model_name)
    # This test assumes CUDA is available and the model is moved to CUDA
    if check_cuda():
        assert next(model.parameters()).is_cuda, "Model parameters should be on CUDA"
    else:
        assert not next(model.parameters()).is_cuda, "Model parameters should not be on CUDA if unavailable"
