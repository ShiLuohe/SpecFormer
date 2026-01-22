
from tqdm import tqdm
from typing import Optional, Dict, Tuple, Any

import torch

from transformers.cache_utils import DynamicCache

class SpeculativeDynamicCache(DynamicCache):

    def __init__(self, layer_concatenate_dim_list):
        super().__init__()
        self.layer_concatenate_dim_list = layer_concatenate_dim_list

    def reverse(self, rev_length: int):
        new_length = self._seen_tokens - rev_length
        for idx, c_dim in enumerate(self.layer_concatenate_dim_list):
            if self.key_cache[idx].shape[c_dim] != self._seen_tokens: continue
            if c_dim == -2:
                self.key_cache[idx] = self.key_cache[idx][:, :,  :new_length, :]
                self.value_cache[idx] = self.value_cache[idx][:, :,  :new_length, :]
            elif c_dim == -3:
                self.key_cache[idx] = self.key_cache[idx][:, :new_length, :, :]
                self.value_cache[idx] = self.value_cache[idx][:, :new_length, :, :]
        self._seen_tokens = new_length

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[self.layer_concatenate_dim_list[layer_idx]]

        # Update the cache
        if key_states is not None:
            if len(self.key_cache) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append(torch.tensor([]))
                    self.value_cache.append(torch.tensor([])) 
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            elif (
                not self.key_cache[layer_idx].numel()  # prefers not t.numel() to len(t) == 0 to export the model
            ):  # fills previously skipped layers; checking for tensor causes errors
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key_states], 
                    dim=self.layer_concatenate_dim_list[layer_idx]
                )
                self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], value_states], 
                    dim=self.layer_concatenate_dim_list[layer_idx]
                )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]
