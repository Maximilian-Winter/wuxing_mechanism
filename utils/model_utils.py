import torch
from torch import nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any


class ModelAnalyzer:
    """
    Utility class for analyzing model characteristics relevant to the Wu Xing framework
    """

    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """
        Count parameters in a PyTorch model

        Args:
            model: PyTorch neural network model

        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        param_counts = {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params,
            'by_layer': {}
        }

        # Count by layer
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.Embedding, nn.LSTM, nn.GRU)):
                layer_params = sum(p.numel() for p in module.parameters())
                param_counts['by_layer'][name] = layer_params

        return param_counts

    @staticmethod
    def analyze_activation_patterns(model: nn.Module, dataloader: torch.utils.data.DataLoader,
                                    device: torch.device) -> Dict[str, Dict[str, float]]:
        """
        Analyze activation patterns in a model

        Args:
            model: PyTorch neural network model
            dataloader: DataLoader with sample data
            device: Device for computation

        Returns:
            Dictionary of activation statistics by layer
        """
        activation_stats = {}

        # Register hooks to capture activations
        activations = {}

        def get_activation(name):
            def hook(module, input, output):
                activations[name] = output.detach()

            return hook

        # Register forward hooks for activations
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU, nn.LSTM, nn.GRU)):
                hook = module.register_forward_hook(get_activation(name))
                hooks.append(hook)

        # Forward pass to get activations
        model.eval()
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                model(inputs)
                break

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Calculate activation statistics
        for name, activation in activations.items():
            if isinstance(activation, torch.Tensor):
                # Convert to numpy for statistics
                if activation.dim() > 1:
                    # For multi-dimensional outputs, flatten all but batch dim
                    flat_activations = activation.view(activation.size(0), -1)

                    # Calculate statistics
                    mean_activation = flat_activations.mean().item()
                    std_activation = flat_activations.std().item()
                    max_activation = flat_activations.max().item()
                    min_activation = flat_activations.min().item()
                    sparsity = (flat_activations == 0).float().mean().item()

                    activation_stats[name] = {
                        'mean': mean_activation,
                        'std': std_activation,
                        'max': max_activation,
                        'min': min_activation,
                        'sparsity': sparsity
                    }

        return activation_stats

    @staticmethod
    def estimate_element_strengths(model: nn.Module) -> Dict[str, float]:
        """
        Estimate all five element strengths based on model characteristics

        Args:
            model: PyTorch neural network model

        Returns:
            Dictionary with estimated strengths for all five elements
        """
        strengths = {
            'water': 0.5,  # Default, would need data analysis for accurate estimate
            'wood': 0.0,
            'fire': 0.0,
            'earth': 0.0,
            'metal': 0.0
        }

        # Estimate Wood strength based on architecture complexity
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_score = min(1.0, np.log(total_params + 1) / np.log(10 ** 8))

        total_layers = sum(1 for _ in model.modules() if isinstance(_, (nn.Conv2d, nn.Linear)))
        layer_score = min(1.0, np.log(total_layers + 1) / np.log(100))

        has_residual = any('res' in name for name, _ in model.named_modules())
        has_attention = any('attention' in name for name, _ in model.named_modules())
        pattern_score = 0.5
        if has_residual:
            pattern_score += 0.2
        if has_attention:
            pattern_score += 0.3
        pattern_score = min(1.0, pattern_score)

        strengths['wood'] = 0.4 * param_score + 0.3 * layer_score + 0.3 * pattern_score

        # Estimate Fire strength based on activation functions
        activation_counts = {
            'relu': sum(1 for m in model.modules() if isinstance(m, nn.ReLU)),
            'leaky_relu': sum(1 for m in model.modules() if isinstance(m, nn.LeakyReLU)),
            'elu': sum(1 for m in model.modules() if isinstance(m, nn.ELU)),
            'gelu': sum(1 for m in model.modules() if isinstance(m, nn.GELU)),
            'sigmoid': sum(1 for m in model.modules() if isinstance(m, nn.Sigmoid)),
            'tanh': sum(1 for m in model.modules() if isinstance(m, nn.Tanh))
        }

        intensity_scores = {
            'relu': 0.7,
            'leaky_relu': 0.8,
            'elu': 0.75,
            'gelu': 0.85,
            'sigmoid': 0.5,
            'tanh': 0.6
        }

        total_activations = sum(activation_counts.values())
        if total_activations > 0:
            weighted_intensity = sum(count * intensity_scores[act_type]
                                     for act_type, count in activation_counts.items())
            strengths['fire'] = weighted_intensity / total_activations
        else:
            strengths['fire'] = 0.5  # Default

        # Estimate Earth strength based on regularization
        has_bn = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
                     for m in model.modules())
        has_ln = any(isinstance(m, nn.LayerNorm) for m in model.modules())
        has_dropout = any(isinstance(m, nn.Dropout) for m in model.modules())

        earth_strength = 0.0
        if has_bn or has_ln:
            earth_strength += 0.4
        if has_dropout:
            earth_strength += 0.4

        earth_strength += 0.2  # Base level
        strengths['earth'] = min(1.0, earth_strength)

        # Estimate Metal strength based on output layer
        output_layer = None
        for module in model.modules():
            if isinstance(module, nn.Linear):
                output_layer = module

        if output_layer is not None:
            out_features = getattr(output_layer, 'out_features', 0)

            if out_features == 0:
                strengths['metal'] = 0.5  # Default
            elif out_features == 1:
                # Binary classification or regression
                strengths['metal'] = 0.8
            elif out_features < 10:
                # Multi-class with few classes
                strengths['metal'] = 0.7
            else:
                # Complex output
                strengths['metal'] = 0.6
        else:
            strengths['metal'] = 0.5  # Default

        return strengths