# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pruned LLaMA model configuration"""

from typing import List, Dict, Any
from .configuration_llama import LlamaConfig


class PrunedLlamaConfig(LlamaConfig):
    r"""
    This is the configuration class for the pruned version of LLaMA model. It inherits from [`LlamaConfig`] and adds
    parameters for per-layer pruning configurations.

    Args:
        per_layer_attention_feature_size (`List[int]`):
            A list of integers specifying the attention feature size for each transformer layer.
            The length of this list must match `num_hidden_layers`.
        per_layer_intermediate_size (`List[int]`):
            A list of integers specifying the intermediate (MLP) size for each transformer layer.
            The length of this list must match `num_hidden_layers`.
        **kwargs:
            Additional parameters inherited from [`LlamaConfig`].

    Example:
    ```python
    >>> from transformers import PrunedLlamaConfig

    >>> # Initialize a pruned LLaMA configuration with custom layer sizes
    >>> config = PrunedLlamaConfig(
    ...     per_layer_attention_feature_size=[3072, 3072, 2048, 2048],  # 示例数值
    ...     per_layer_intermediate_size=[8192, 8192, 6144, 6144],       # 示例数值
    ...     num_hidden_layers=4
    ... )
    ```
    """
    model_type: str = "pruned_llama"
    is_composition: bool = True

    def __init__(
        self,
        per_layer_attention_feature_size: List[int] = None,
        per_layer_intermediate_size: List[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.per_layer_attention_feature_size = per_layer_attention_feature_size or [self.hidden_size] * self.num_hidden_layers
        self.per_layer_intermediate_size = per_layer_intermediate_size or [self.intermediate_size] * self.num_hidden_layers

        if len(self.per_layer_attention_feature_size) != self.num_hidden_layers:
            raise ValueError(
                f"Length of per_layer_attention_feature_size ({len(self.per_layer_attention_feature_size)}) "
                f"must match num_hidden_layers ({self.num_hidden_layers})"
            )
        
        if len(self.per_layer_intermediate_size) != self.num_hidden_layers:
            raise ValueError(
                f"Length of per_layer_intermediate_size ({len(self.per_layer_intermediate_size)}) "
                f"must match num_hidden_layers ({self.num_hidden_layers})"
            )

    def get_attention_feature_size(self, layer_index: int) -> int:
        """获取指定层的注意力特征大小"""
        if not 0 <= layer_index < self.num_hidden_layers:
            raise ValueError(f"Layer index {layer_index} is out of range [0, {self.num_hidden_layers})")
        return self.per_layer_attention_feature_size[layer_index]

    def get_intermediate_size(self, layer_index: int) -> int:
        """获取指定层的中间层大小"""
        if not 0 <= layer_index < self.num_hidden_layers:
            raise ValueError(f"Layer index {layer_index} is out of range [0, {self.num_hidden_layers})")
        return self.per_layer_intermediate_size[layer_index]
