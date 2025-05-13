import torch
from torch import nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable, Any

from .adapters import BaseAdapter


class HuggingFaceAdapter(BaseAdapter):
    """
    Adapter for Hugging Face Transformer models

    This adapter is designed to work with models from the Hugging Face transformers
    library, including BERT, GPT, RoBERTa, T5, and other transformer architectures.
    """

    def __init__(self, model_type: str = 'bert', task_type: str = 'text-classification',
                 data_type: str = 'text'):
        """
        Initialize the HuggingFace adapter

        Args:
            model_type: Specific type of transformer model
                        ('bert', 'gpt', 'roberta', 't5', etc.)
            task_type: Type of NLP task
                        ('text-classification', 'token-classification',
                         'question-answering', 'summarization', etc.)
            data_type: Type of data (usually 'text')
        """
        super().__init__(model_type=model_type, task_type=task_type, data_type=data_type)

        # Store specific transformer type
        self.transformer_type = model_type.lower()

        # Store task details
        self.nlp_task = task_type

    def measure_water_strength(self, model: nn.Module, dataloader: Any) -> float:
        """
        Measure Water element (data quality) strength

        For NLP, this evaluates tokenization quality, sequence lengths,
        and input data characteristics.

        Args:
            model: Hugging Face transformer model
            dataloader: DataLoader with sample inputs

        Returns:
            Water strength value between 0 and 1
        """
        # NLP-specific water strength measurements
        water_strength = 0.5  # Default value

        # Check a batch of data
        for batch in dataloader:
            # Check for attention masks (indicating proper padding)
            has_attention_mask = 'attention_mask' in batch

            # Check for token_type_ids (used in some models like BERT)
            has_token_type = 'token_type_ids' in batch

            # Check for input truncation (sequences shouldn't be all identical length)
            if 'input_ids' in batch:
                input_ids = batch['input_ids'] if isinstance(batch, dict) else batch[0]
                if hasattr(input_ids, 'shape'):
                    # Calculate sequence length distribution
                    if has_attention_mask:
                        attention_mask = batch['attention_mask'] if isinstance(batch, dict) else batch[1]
                        seq_lengths = attention_mask.sum(dim=1).float()
                        seq_length_var = seq_lengths.var().item() if seq_lengths.numel() > 1 else 0

                        # Variable sequence lengths suggest proper padding rather than truncation
                        length_diversity = min(1.0, seq_length_var / 10)
                    else:
                        length_diversity = 0.3  # Default if no attention mask

                    # Check if sequences are reasonably sized (not too short)
                    if hasattr(input_ids, 'size'):
                        avg_seq_length = input_ids.size(1)
                        length_adequacy = min(1.0, avg_seq_length / 256)  # Scale to common sequence lengths
                    else:
                        length_adequacy = 0.5  # Default
                else:
                    length_diversity = 0.3
                    length_adequacy = 0.5
            else:
                length_diversity = 0.3
                length_adequacy = 0.5

            # Combine factors
            water_strength = 0.0

            # Proper attention masking improves data quality
            if has_attention_mask:
                water_strength += 0.3

            # Token type IDs indicate more structured inputs (e.g., for BERT)
            if has_token_type:
                water_strength += 0.2

            # Length diversity and adequacy
            water_strength += 0.25 * length_diversity
            water_strength += 0.25 * length_adequacy

            # Only check one batch
            break

        return min(1.0, water_strength)

    def measure_wood_strength(self, model: nn.Module) -> float:
        """
        Measure Wood element (architecture complexity) strength

        For transformers, this evaluates model size, number of layers,
        attention heads, and architectural features.

        Args:
            model: Hugging Face transformer model

        Returns:
            Wood strength value between 0 and 1
        """
        # Get model config if available
        config = getattr(model, 'config', None)

        # If config is available, use it to determine complexity
        if config is not None:
            # Get number of layers
            num_layers = getattr(config, 'num_hidden_layers', None)
            if num_layers is None:
                num_layers = getattr(config, 'n_layer', None)

            if num_layers is None:
                # Try to infer from model structure
                encoder_layers = sum(1 for name, _ in model.named_modules()
                                     if 'encoder.layer' in name or 'h.' in name)
                decoder_layers = sum(1 for name, _ in model.named_modules()
                                     if 'decoder.layer' in name)
                num_layers = max(encoder_layers, 1) + max(decoder_layers, 0)

            # Get number of attention heads
            num_heads = getattr(config, 'num_attention_heads', None)
            if num_heads is None:
                num_heads = getattr(config, 'n_head', 8)  # Default to common value

            # Get hidden dimension
            hidden_dim = getattr(config, 'hidden_size', None)
            if hidden_dim is None:
                hidden_dim = getattr(config, 'd_model', 768)  # Default to common value

            # Calculate layer score (normalized to common transformer sizes)
            layer_score = min(1.0, num_layers / 24)  # Scale to largest common models

            # Calculate head score
            head_score = min(1.0, num_heads / 16)  # Scale to common head counts

            # Calculate dimension score
            dim_score = min(1.0, hidden_dim / 1024)  # Scale to common dimensions

            # Check for architectural enhancements
            has_intermediate_ffn = hasattr(config, 'intermediate_size') and config.intermediate_size > hidden_dim
            advanced_score = 0.5
            if has_intermediate_ffn:
                advanced_score += 0.2
            if getattr(config, 'add_cross_attention', False):
                advanced_score += 0.3  # Cross-attention (like in encoder-decoder models)
            advanced_score = min(1.0, advanced_score)

            # Combine scores
            wood_strength = 0.25 * layer_score + 0.25 * head_score + 0.25 * dim_score + 0.25 * advanced_score

        else:
            # Fallback to parameter counting
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # Scale logarithmically to account for wide range of model sizes
            # Small models ~10M, large models ~1B+
            param_score = min(1.0, np.log(total_params + 1) / np.log(10 ** 10))

            # Count transformer blocks
            blocks = 0
            for name, module in model.named_modules():
                if any(block_name in name.lower() for block_name in
                       ['encoderlayer', 'decoderlayer', 'block', 'transformer']):
                    blocks += 1

            layer_score = min(1.0, blocks / 50)  # Normalize to common range

            # Combine scores
            wood_strength = 0.7 * param_score + 0.3 * layer_score

        return wood_strength

    def measure_fire_strength(self, model: nn.Module) -> float:
        """
        Measure Fire element (training intensity) strength

        For transformers, this evaluates activation functions, attention
        temperature, and training-related parameters.

        Args:
            model: Hugging Face transformer model

        Returns:
            Fire strength value between 0 and 1
        """
        # Get model config if available
        config = getattr(model, 'config', None)

        # Initialize fire components
        activation_score = 0.5
        attention_score = 0.5

        # Determine activation function from config
        if config is not None:
            # Check activation function
            activation_fn = getattr(config, 'hidden_act', None)

            if activation_fn is not None:
                if activation_fn in ['gelu', 'gelu_new']:
                    activation_score = 0.8  # GELU is common in modern transformers
                elif activation_fn == 'relu':
                    activation_score = 0.7
                elif activation_fn == 'swish' or activation_fn == 'silu':
                    activation_score = 0.85  # More advanced activation
                elif activation_fn == 'leakyrelu':
                    activation_score = 0.75
        else:
            # Check modules for activation functions
            activation_counts = {
                'gelu': sum(1 for name, m in model.named_modules() if 'gelu' in name.lower()),
                'relu': sum(1 for name, m in model.named_modules() if 'relu' in name.lower()),
                'swish': sum(1 for name, m in model.named_modules() if 'swish' in name.lower())
            }

            # Determine predominant activation function
            if sum(activation_counts.values()) > 0:
                if activation_counts['gelu'] > activation_counts['relu'] and activation_counts['gelu'] > \
                        activation_counts['swish']:
                    activation_score = 0.8
                elif activation_counts['swish'] > activation_counts['relu']:
                    activation_score = 0.85
                elif activation_counts['relu'] > 0:
                    activation_score = 0.7

        # Check for FFN expansion ratio
        ffn_ratio = 0.0
        if config is not None:
            hidden_size = getattr(config, 'hidden_size', 0)
            intermediate_size = getattr(config, 'intermediate_size', 0)

            if hidden_size > 0 and intermediate_size > 0:
                ffn_ratio = intermediate_size / hidden_size

        if ffn_ratio > 0:
            # Normalize to common values (usually 4x)
            ffn_score = min(1.0, ffn_ratio / 4.0)
        else:
            # Check for FFN modules in model
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and any(x in name.lower() for x in ['intermediate', 'ffn', 'mlp']):
                    if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                        ratio = module.out_features / module.in_features
                        if ratio > ffn_ratio:
                            ffn_ratio = ratio

            ffn_score = min(1.0, ffn_ratio / 4.0) if ffn_ratio > 0 else 0.5

        # Combine scores
        fire_strength = 0.5 * activation_score + 0.5 * ffn_score

        return fire_strength

    def measure_earth_strength(self, model: nn.Module) -> float:
        """
        Measure Earth element (regularization/stability) strength

        For transformers, this evaluates layer normalization, dropout,
        and other stabilizing techniques.

        Args:
            model: Hugging Face transformer model

        Returns:
            Earth strength value between 0 and 1
        """
        # Get model config if available
        config = getattr(model, 'config', None)

        # Initialize earth components
        layer_norm_score = 0.0
        dropout_score = 0.0

        # Check config for dropout and normalization
        if config is not None:
            # Check dropout rate
            dropout = getattr(config, 'hidden_dropout_prob', None)
            if dropout is None:
                dropout = getattr(config, 'dropout', 0.1)  # Default to common value

            # Higher dropout = stronger regularization
            dropout_score = min(1.0, dropout * 5)  # Scale to [0, 1], assumes dropout < 0.2

            # Check layer normalization settings
            has_layer_norm = getattr(config, 'layer_norm_eps', None) is not None
            if has_layer_norm:
                layer_norm_score = 0.8
        else:
            # Count normalization and dropout layers
            norm_count = sum(1 for name, m in model.named_modules()
                             if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)))

            dropout_layers = sum(1 for name, m in model.named_modules()
                                 if isinstance(m, nn.Dropout))

            # Calculate scores based on presence
            layer_norm_score = min(1.0, norm_count / 10) if norm_count > 0 else 0.0

            if dropout_layers > 0:
                # Check one dropout layer for its value
                for name, m in model.named_modules():
                    if isinstance(m, nn.Dropout):
                        dropout_score = min(1.0, m.p * 5)  # Scale to [0, 1]
                        break
            else:
                dropout_score = 0.0

        # Combine scores with a base level
        earth_strength = 0.5 * layer_norm_score + 0.5 * dropout_score

        # Ensure minimum earth strength as transformers always have some stabilization
        earth_strength = max(0.3, earth_strength)

        return earth_strength

    def measure_metal_strength(self, model: nn.Module) -> float:
        """
        Measure Metal element (evaluation precision) strength

        For transformers, this evaluates output layer, embeddings precision,
        and task-specific heads.

        Args:
            model: Hugging Face transformer model

        Returns:
            Metal strength value between 0 and 1
        """
        # For transformers, metal is related to embedding precision and output handling
        config = getattr(model, 'config', None)

        # Initialize metal components
        embedding_score = 0.5
        output_score = 0.5

        # Check embedding dimension and output handling from config
        if config is not None:
            # Get embedding dimension
            embedding_dim = getattr(config, 'hidden_size', None)
            if embedding_dim is None:
                embedding_dim = getattr(config, 'd_model', 768)  # Default

            # Scale to common values (typically 256-1024)
            embedding_score = min(1.0, embedding_dim / 1024)

            # Check output/head configuration based on task
            if hasattr(config, 'num_labels'):
                if config.num_labels == 1:
                    # Regression tasks need high precision
                    output_score = 0.9
                elif config.num_labels == 2:
                    # Binary classification
                    output_score = 0.8
                elif config.num_labels > 2 and config.num_labels < 10:
                    # Multi-class classification with few classes
                    output_score = 0.7
                else:
                    # Many classes or complex output
                    output_score = 0.6
            else:
                # Default for language models
                vocab_size = getattr(config, 'vocab_size', 0)
                if vocab_size > 0:
                    output_score = min(1.0, 0.3 + np.log(vocab_size) / np.log(50000))
        else:
            # Look for embedding and output layers in model
            for name, module in model.named_modules():
                if isinstance(module, nn.Embedding):
                    if hasattr(module, 'embedding_dim'):
                        embedding_score = min(1.0, module.embedding_dim / 1024)
                        break

            # Look for classifier or output heads
            has_classifier = any('classifier' in name for name, _ in model.named_modules())
            has_lm_head = any('lm_head' in name for name, _ in model.named_modules())

            if has_classifier:
                output_score = 0.8  # Classification head
            elif has_lm_head:
                output_score = 0.7  # Language modeling head

        # Combine scores
        metal_strength = 0.5 * embedding_score + 0.5 * output_score

        return metal_strength

    def map_param_to_element(self, param_name: str) -> int:
        """
        Map a transformer parameter name to the Wu Xing element it primarily affects

        Args:
            param_name: Name of the parameter

        Returns:
            Integer index of the corresponding element (0-4)
        """
        # For transformers, we need to analyze the parameter name patterns

        # Embeddings affect data representation (Water)
        if any(x in param_name for x in ['embed', 'token', 'wte', 'wpe']):
            return 0  # Water

        # Attention query/key/value weights affect pattern formation (Wood)
        if 'attention' in param_name:
            if any(x in param_name for x in ['query', 'key', 'q', 'k']):
                return 1  # Wood
            elif any(x in param_name for x in ['value', 'v']):
                return 2  # Fire
            else:
                return 1  # Other attention parts mostly Wood

        # Output dense layers, FFN, and MLP affect transformation (Fire)
        if any(x in param_name for x in ['dense', 'ffn', 'mlp', 'intermediate']):
            if 'down' in param_name or 'output' in param_name:
                return 2  # Fire
            else:
                return 2  # Fire

        # Layer normalization affects stability (Earth)
        if any(x in param_name for x in ['LayerNorm', 'layer_norm', 'norm']):
            return 3  # Earth

        # Classifier, lm_head, output bias affect evaluation (Metal)
        if any(x in param_name for x in ['classifier', 'lm_head', 'output_bias']):
            return 4  # Metal

        # Default if no pattern matches
        return 0  # Default to Water

    def design_intervention_vector(self, param_name: str, param: torch.Tensor,
                                   target_state: 'WuXingStateVector',
                                   current_state: 'WuXingStateVector',
                                   magnitude: float = 0.1) -> torch.Tensor:
        """
        Design intervention vector for a transformer parameter

        Args:
            param_name: Name of the parameter
            param: Parameter tensor
            target_state: Target WuXingStateVector
            current_state: Current WuXingStateVector
            magnitude: Overall magnitude of the intervention

        Returns:
            Intervention tensor with the same shape as the parameter
        """
        # Calculate state change vector
        delta_state = target_state.state - current_state.state

        # Determine which element this parameter primarily affects
        element_index = self.map_param_to_element(param_name)
        delta_element = delta_state[element_index]

        # For transformer parameters, we need to be more careful

        # Embedding interventions (Water)
        if element_index == 0:  # Water
            # For embeddings, make very small changes to avoid disrupting semantics
            std_value = param.data.std().item() * 0.1  # Smaller scale for embeddings

            # Create intervention
            if delta_element > 0:  # Increase water
                intervention = torch.randn_like(param.data) * std_value * magnitude
            else:  # Decrease water
                # Target specific rows (tokens) that might be overused
                row_norms = torch.norm(param.data, dim=1)
                top_indices = torch.topk(row_norms, k=max(1, int(0.05 * row_norms.size(0))), largest=True).indices

                intervention = torch.zeros_like(param.data)
                intervention[top_indices] = -torch.sign(param.data[top_indices]) * std_value * magnitude

        # Attention query/key interventions (Wood)
        elif element_index == 1 and any(x in param_name for x in ['query', 'key', 'q', 'k', 'attention']):
            # For attention matrices, target intervention to enhance or reduce pattern recognition
            std_value = param.data.std().item()

            if delta_element > 0:  # Increase wood
                # Enhance pattern recognition by increasing diversity
                intervention = torch.randn_like(param.data) * std_value * magnitude
            else:  # Decrease wood
                # Reduce pattern recognition by making weights more uniform
                intervention = -param.data * magnitude * 0.1

        # Value and FFN interventions (Fire)
        elif element_index == 2:
            std_value = param.data.std().item()

            if delta_element > 0:  # Increase fire
                # Increase transformation power
                if param.grad is not None:
                    # Use gradient information to guide intervention
                    direction = -torch.sign(param.grad)
                    intervention = direction * std_value * magnitude
                else:
                    # Random intervention weighted by parameter magnitude
                    intervention = torch.randn_like(param.data) * std_value * magnitude
            else:  # Decrease fire
                # Make transformation more moderate
                intervention = -torch.sign(param.data) * param.data.abs() * magnitude * 0.1

        # Layer Norm interventions (Earth)
        elif element_index == 3:
            if 'weight' in param_name:
                if delta_element > 0:  # Increase earth
                    # Move weights toward 1 (stronger normalization)
                    intervention = (torch.ones_like(param.data) - param.data) * magnitude
                else:  # Decrease earth
                    # Increase weights (weaker normalization)
                    intervention = torch.ones_like(param.data) * magnitude
            else:  # bias
                if delta_element > 0:  # Increase earth
                    # Move biases toward 0
                    intervention = -param.data * magnitude
                else:  # Decrease earth
                    # Move biases away from 0
                    intervention = torch.sign(param.data + 1e-5) * magnitude

        # Output/classifier interventions (Metal)
        elif element_index == 4:
            std_value = param.data.std().item()

            if delta_element > 0:  # Increase metal
                # For classification layers, make more decisive
                intervention = torch.sign(param.data) * param.data.abs() * magnitude * 0.2
            else:  # Decrease metal
                # Make less decisive
                intervention = -torch.sign(param.data) * param.data.abs() * magnitude * 0.1

        # Generic fallback
        else:
            std_value = param.data.std().item()

            if delta_element > 0:
                intervention = torch.randn_like(param.data) * std_value * magnitude
            else:
                intervention = -torch.randn_like(param.data) * std_value * magnitude

        return intervention