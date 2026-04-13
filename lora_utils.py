"""
Utilities for LoRA fine-tuning of MeanFlow DiT models.

Handles:
- Loading pretrained base model weights into LoRA model
- Creating optimizer that freezes base params, trains only LoRA
- Counting and summarizing parameters
"""
from flax.core import freeze, unfreeze, FrozenDict

import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
import optax

from utils.logging_util import log_for_0


def is_lora_param(path_tuple):
    """Check if a flattened parameter path belongs to a LoRA adapter."""
    path_str = "/".join(str(p) for p in path_tuple)
    return "lora_A" in path_str or "lora_B" in path_str


def create_lora_mask(params):
    """Create a boolean mask tree matching params structure.
    
    True = LoRA param (trainable), False = base param (frozen).
    Uses jax.tree.map_with_path to preserve exact tree structure.
    """
    from flax.core import unfreeze
    params = unfreeze(params)  # Convert all FrozenDicts to regular dicts
    
    def label_fn(path, _):
        path_str = "/".join(str(p) for p in path)
        return "lora_A" in path_str or "lora_B" in path_str
    
    return jax.tree_util.tree_map_with_path(label_fn, params)

def zero_non_lora_grads(grads, params=None):
    """Zero gradients for non-LoRA params while preserving PyTree structure."""
    was_frozen = isinstance(grads, FrozenDict)
    grad_tree = unfreeze(grads) if was_frozen else grads

    def maybe_zero(path, g):
        path_str = "/".join(str(p) for p in path)
        if "lora_A" in path_str or "lora_B" in path_str:
            return g
        return jnp.zeros_like(g)

    grad_tree = jax.tree_util.tree_map_with_path(maybe_zero, grad_tree)
    return freeze(grad_tree) if was_frozen else grad_tree

def load_pretrained_into_lora(lora_params, pretrained_params):
    """Load pretrained base model weights into LoRA model params.
    
    The LoRA model has the same base params as the pretrained model,
    plus additional lora_A and lora_B params (initialized to zero).
    
    This function copies pretrained weights into the base param slots
    while keeping LoRA params at their initialized values (zeros for B,
    small random for A).
    
    Args:
        lora_params: Parameters from freshly initialized LoRA model
        pretrained_params: Parameters from pretrained base model checkpoint
    
    Returns:
        Merged params with pretrained base + initialized LoRA
    """
    lora_flat = flatten_dict(unfreeze(lora_params))
    pretrained_flat = flatten_dict(unfreeze(pretrained_params))
    
    merged = {}
    loaded_count = 0
    kept_lora_count = 0
    missing_count = 0
    
    for key, lora_value in lora_flat.items():
        if is_lora_param(key):
            # Keep LoRA params as initialized (zeros for B, random for A)
            merged[key] = lora_value
            kept_lora_count += 1
        elif key in pretrained_flat:
            # Load from pretrained checkpoint
            pretrained_value = pretrained_flat[key]
            if pretrained_value.shape != lora_value.shape:
                log_for_0(f"WARNING: Shape mismatch for {'/'.join(str(k) for k in key)}: "
                         f"pretrained={pretrained_value.shape}, lora={lora_value.shape}. "
                         f"Keeping initialized value.")
                merged[key] = lora_value
                missing_count += 1
            else:
                merged[key] = pretrained_value
                loaded_count += 1
        else:
            # Param exists in LoRA model but not in pretrained
            # (shouldn't happen for non-LoRA params, but handle gracefully)
            path_str = "/".join(str(k) for k in key)
            log_for_0(f"WARNING: {path_str} not found in pretrained checkpoint, "
                     f"keeping initialized value.")
            merged[key] = lora_value
            missing_count += 1
    
    log_for_0(f"Loaded {loaded_count} base params from pretrained checkpoint")
    log_for_0(f"Kept {kept_lora_count} LoRA params at initialized values")
    if missing_count > 0:
        log_for_0(f"WARNING: {missing_count} params not found in pretrained checkpoint")
    
    return freeze(unflatten_dict(merged))


def print_lora_summary(params):
    """Print summary of trainable vs frozen parameters."""
    flat = flatten_dict(unfreeze(params))
    
    total_params = 0
    lora_params = 0
    base_params = 0
    
    for key, value in flat.items():
        size = value.size
        total_params += size
        if is_lora_param(key):
            lora_params += size
        else:
            base_params += size
    
    log_for_0("=" * 50)
    log_for_0("LoRA Fine-tuning Parameter Summary")
    log_for_0("=" * 50)
    log_for_0(f"  Base parameters (frozen):    {base_params:>12,}")
    log_for_0(f"  LoRA parameters (trainable): {lora_params:>12,}")
    log_for_0(f"  Total parameters:            {total_params:>12,}")
    log_for_0(f"  Trainable fraction:          {lora_params/total_params*100:>11.2f}%")
    log_for_0("=" * 50)
    
    return total_params, lora_params


def print_lora_params_detail(params):
    """Print detailed list of all LoRA parameters."""
    flat = flatten_dict(unfreeze(params))
    
    log_for_0("LoRA parameters:")
    for key, value in sorted(flat.items()):
        if is_lora_param(key):
            path_str = "/".join(str(k) for k in key)
            log_for_0(f"  {path_str}: {value.shape} ({value.size:,} params)")
