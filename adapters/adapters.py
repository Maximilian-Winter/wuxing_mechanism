import torch
from torch import nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable, Any

from ..core.state_vector import WuXingStateVector


class BaseAdapter:
    """
    Base adapter class for WuXingMechanism

    This class provides a template for adapting the WuXingMechanism framework
    to different model architectures, task types, and data types.
    """

    def __init__(self, model_type: str = 'generic',
                 task_type: str = 'generic',
                 data_type: str = 'generic'):
        """
        Initialize the adapter

        Args:
            model_type: Type of model architecture
            task_type: Type of ML task
            data_type: Type of data
        """
        self.model_type = model_type
        self.task_type = task_type
        self.data_type = data_type

    def measure_water_strength(self, model: nn.Module,
                               dataloader: Any) -> float:
        """
        Measure Water element (data quality) strength

        Args:
            model: PyTorch neural network model
            dataloader: DataLoader with data samples

        Returns:
            Water strength value between 0 and 1
        """
        # Generic implementation - override in specific adapters
        # Check for input normalization
        has_normalization = False

        # Look for normalization in the first few batches
        for inputs, _ in dataloader:
            if isinstance(inputs, torch.Tensor):
                # Check if data appears normalized
                batch_mean = inputs.mean().item()
                batch_std = inputs.std().item()

                # If mean is close to 0 and std close to 1, likely normalized
                if abs(batch_mean) < 0.1 and 0.9 < batch_std < 1.1:
                    has_normalization = True

            # Only check one batch
            break

        # For generic case, return a default based on normalization
        return 0.7 if has_normalization else 0.4

    def measure_wood_strength(self, model: nn.Module) -> float:
        """
        Measure Wood element (architecture complexity) strength

        Args:
            model: PyTorch neural network model

        Returns:
            Wood strength value between 0 and 1
        """
        # Count parameters and layers
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_layers = sum(1 for _ in model.modules() if isinstance(_, (nn.Conv2d, nn.Linear)))

        # Normalize parameter count with log scaling
        param_score = min(1.0, np.log(total_params + 1) / np.log(10 ** 8))

        # Normalize layer count
        layer_score = min(1.0, np.log(total_layers + 1) / np.log(100))

        # Combine scores
        return 0.7 * param_score + 0.3 * layer_score

    def measure_fire_strength(self, model: nn.Module) -> float:
        """
        Measure Fire element (training intensity) strength

        Args:
            model: PyTorch neural network model

        Returns:
            Fire strength value between 0 and 1
        """
        # For generic implementation, check activation functions
        activation_counts = {
            'relu': 0,
            'leaky_relu': 0,
            'elu': 0,
            'gelu': 0,
            'sigmoid': 0,
            'tanh': 0
        }

        for module in model.modules():
            if isinstance(module, nn.ReLU):
                activation_counts['relu'] += 1
            elif isinstance(module, nn.LeakyReLU):
                activation_counts['leaky_relu'] += 1
            elif isinstance(module, nn.ELU):
                activation_counts['elu'] += 1
            elif isinstance(module, nn.GELU):
                activation_counts['gelu'] += 1
            elif isinstance(module, nn.Sigmoid):
                activation_counts['sigmoid'] += 1
            elif isinstance(module, nn.Tanh):
                activation_counts['tanh'] += 1

        # Assign activation intensity scores
        intensity_scores = {
            'relu': 0.7,
            'leaky_relu': 0.8,
            'elu': 0.75,
            'gelu': 0.85,
            'sigmoid': 0.5,
            'tanh': 0.6
        }

        total_activations = sum(activation_counts.values())
        if total_activations == 0:
            return 0.5  # Default

        # Calculate weighted average of activation intensities
        weighted_intensity = sum(count * intensity_scores[act_type]
                                 for act_type, count in activation_counts.items())

        return weighted_intensity / total_activations

    def measure_earth_strength(self, model: nn.Module) -> float:
        """
        Measure Earth element (regularization/stability) strength

        Args:
            model: PyTorch neural network model

        Returns:
            Earth strength value between 0 and 1
        """
        # Check for regularization techniques
        has_bn = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
                     for m in model.modules())
        has_ln = any(isinstance(m, nn.LayerNorm) for m in model.modules())
        has_dropout = any(isinstance(m, nn.Dropout) for m in model.modules())

        # Calculate Earth strength based on regularization
        earth_strength = 0.0
        if has_bn or has_ln:
            earth_strength += 0.4
        if has_dropout:
            earth_strength += 0.4

        # Add a base level
        earth_strength += 0.2

        return min(1.0, earth_strength)

    def measure_metal_strength(self, model: nn.Module) -> float:
        """
        Measure Metal element (evaluation precision) strength

        Args:
            model: PyTorch neural network model

        Returns:
            Metal strength value between 0 and 1
        """
        # For generic case, check output layer characteristics
        output_layer = None

        # Find the last linear layer, likely the output layer
        for module in model.modules():
            if isinstance(module, nn.Linear):
                output_layer = module

        if output_layer is None:
            return 0.5  # Default if no output layer found

        # Check output dimensionality
        out_features = getattr(output_layer, 'out_features', 0)

        if out_features == 0:
            return 0.5  # Default
        elif out_features == 1:
            # Binary classification or regression (high precision)
            return 0.8
        elif out_features < 10:
            # Multi-class classification with few classes
            return 0.7
        else:
            # Complex output (many classes or features)
            return 0.6

    def map_param_to_element(self, param_name: str) -> int:
        """
        Map a parameter name to the Wu Xing element it primarily affects

        Args:
            param_name: Name of the parameter

        Returns:
            Integer index of the corresponding element (0-4)
        """
        # Generic mapping based on parameter name
        if 'conv' in param_name or 'rnn' in param_name or 'gru' in param_name or 'lstm' in param_name:
            if 'weight' in param_name:
                return 1  # Early weights are Wood (pattern formation)
            else:
                return 0  # Biases affect data flow (Water)
        elif 'bn' in param_name or 'norm' in param_name or 'dropout' in param_name:
            return 3  # Normalization is Earth (stability)
        elif 'fc' in param_name or 'linear' in param_name or 'dense' in param_name:
            if 'weight' in param_name:
                if 'fc1' in param_name or 'linear1' in param_name:
                    return 2  # First FC layer is transformation (Fire)
                else:
                    return 4  # Output layer weights are Metal (evaluation)
            else:
                return 3  # Other FC parameters are Earth
        return 0  # Default to Water

    def calculate_target_state(self, current_state: WuXingStateVector,
                               desired_outcome: str) -> WuXingStateVector:
        """
        Calculate target Wu Xing state based on desired outcome

        Args:
            current_state: Current WuXingStateVector
            desired_outcome: String indicating the desired outcome

        Returns:
            Target WuXingStateVector
        """
        water, wood, fire, earth, metal = current_state.state

        if desired_outcome == 'accuracy':
            # Increase Fire (training) and Metal (evaluation precision)
            target_water = water * 0.9  # Slight decrease
            target_wood = wood * 1.1  # Slight increase
            target_fire = min(fire * 1.3, 1.0)  # Significant increase
            target_earth = max(earth * 0.8, 0.2)  # Decrease
            target_metal = min(metal * 1.2, 1.0)  # Moderate increase

        elif desired_outcome == 'generalization':
            # Increase Water (data) and Earth (regularization)
            target_water = min(water * 1.3, 1.0)  # Significant increase
            target_wood = wood  # Maintain
            target_fire = max(fire * 0.9, 0.2)  # Slight decrease
            target_earth = min(earth * 1.4, 1.0)  # Significant increase
            target_metal = metal  # Maintain

        elif desired_outcome == 'efficiency':
            # Increase Metal (precision) and decrease Wood (complexity)
            target_water = water  # Maintain
            target_wood = max(wood * 0.8, 0.2)  # Decrease
            target_fire = fire  # Maintain
            target_earth = earth  # Maintain
            target_metal = min(metal * 1.2, 1.0)  # Moderate increase

        else:  # Balance
            # Move toward balance (0.6 for all elements)
            target_water = 0.6 + (water - 0.6) * 0.5
            target_wood = 0.6 + (wood - 0.6) * 0.5
            target_fire = 0.6 + (fire - 0.6) * 0.5
            target_earth = 0.6 + (earth - 0.6) * 0.5
            target_metal = 0.6 + (metal - 0.6) * 0.5

        return WuXingStateVector(
            target_water, target_wood, target_fire, target_earth, target_metal
        )

    def design_intervention_vector(self, param_name: str, param: torch.Tensor,
                                   target_state: WuXingStateVector,
                                   current_state: WuXingStateVector,
                                   magnitude: float = 0.1) -> torch.Tensor:
        """
        Design intervention vector for a parameter

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

        # Create intervention based on parameter type
        if 'weight' in param_name:
            # For weights, scale based on existing parameter distribution
            std_value = param.data.std().item()

            # Direction based on desired element change
            if delta_element > 0:  # Increase element
                # For weights: decrease values if gradient is positive, increase if negative
                # (due to gradient descent's negative relationship)
                direction = torch.randn_like(param.data)
            else:  # Decrease element
                direction = -torch.randn_like(param.data)

            intervention = direction * std_value * magnitude * abs(delta_element)

        elif 'bias' in param_name:
            # For biases, apply more uniform shift
            if delta_element > 0:  # Increase element
                direction = torch.ones_like(param.data)
            else:  # Decrease element
                direction = -torch.ones_like(param.data)

            intervention = direction * magnitude * abs(delta_element)

        else:
            # Generic intervention
            if delta_element > 0:  # Increase element
                direction = torch.randn_like(param.data)
            else:  # Decrease element
                direction = -torch.randn_like(param.data)

            intervention = direction * param.data.std().item() * magnitude * abs(delta_element)

        return intervention

    def create_wuxing_mechanism(self, model: nn.Module,
                                dataloader: Any,
                                criterion: Callable,
                                device: Optional[torch.device] = None):
        """
        Create a WuXingMechanism instance with this adapter's functions

        Args:
            model: PyTorch neural network model
            dataloader: DataLoader with validation data
            criterion: Loss function
            device: Device for computation

        Returns:
            WuXingMechanism instance
        """
        # Import here to avoid circular imports
        from ..core.mechanism import WuXingMechanism

        # Create base mechanism
        mechanism = WuXingMechanism(model, dataloader, criterion, device)

        # Override methods with adapter's implementations
        mechanism._override_measure_water_strength = lambda: self.measure_water_strength(model, dataloader)
        mechanism._override_measure_wood_strength = lambda: self.measure_wood_strength(model)
        mechanism._override_measure_fire_strength = lambda: self.measure_fire_strength(model)
        mechanism._override_measure_earth_strength = lambda: self.measure_earth_strength(model)
        mechanism._override_measure_metal_strength = lambda: self.measure_metal_strength(model)
        mechanism._override_map_param_to_element = lambda param_name: self.map_param_to_element(param_name)
        mechanism._override_calculate_target_state = lambda current_state, desired_outcome: self.calculate_target_state(
            current_state, desired_outcome)
        mechanism._override_design_intervention_vector = lambda param_name, param, target_state, current_state, magnitude: self.design_intervention_vector(param_name,
                                                                                                           param,
                                                                                                           target_state,
                                                                                                           current_state,
                                                                                                           magnitude)

        return mechanism


class CNNAdapter(BaseAdapter):
    """Adapter for Convolutional Neural Networks"""

    def __init__(self, task_type: str = 'classification', data_type: str = 'image'):
        super().__init__(model_type='cnn', task_type=task_type, data_type=data_type)

    def measure_wood_strength(self, model: nn.Module) -> float:
        """Measure Wood element (architecture complexity) for CNNs"""
        # Count convolutional layers and channels
        conv_layers = 0
        max_channels = 0

        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                conv_layers += 1
                max_channels = max(max_channels, module.out_channels)

        # No conv layers found
        if conv_layers == 0:
            return super().measure_wood_strength(model)

        # Normalize layer count
        layer_score = min(1.0, conv_layers / 20)  # Scale to common CNN depths

        # Normalize channel count
        channel_score = min(1.0, np.log(max_channels + 1) / np.log(1024))

        # Check for residual connections
        has_residual = any('residual' in name.lower() or 'resnet' in name.lower() or 'skip' in name.lower()
                           for name, _ in model.named_modules())

        # Calculate architecture complexity
        complexity_score = (0.4 * layer_score + 0.4 * channel_score)
        if has_residual:
            complexity_score += 0.2

        return min(1.0, complexity_score)

    def map_param_to_element(self, param_name: str) -> int:
        """Map CNN parameter names to elements"""
        if 'conv' in param_name:
            if 'weight' in param_name:
                # Early convolutions handle pattern recognition (Wood)
                if any(x in param_name for x in ['conv1', 'conv2', 'conv_1', 'conv_2']):
                    return 1  # Wood
                else:
                    return 2  # Later convolutions transform features (Fire)
            else:
                return 0  # Biases affect data flow (Water)
        elif 'bn' in param_name or 'batch_norm' in param_name:
            return 3  # BatchNorm is regularization (Earth)
        elif 'fc' in param_name or 'linear' in param_name or 'classifier' in param_name:
            if 'weight' in param_name:
                if param_name.endswith('.weight') or 'fc2' in param_name:
                    return 4  # Output layer (Metal)
                else:
                    return 2  # Hidden fully connected (Fire)
            else:
                return 0  # Biases (Water)

        # Default to base adapter implementation
        return super().map_param_to_element(param_name)

    def calculate_target_state(self, current_state: WuXingStateVector,
                               desired_outcome: str) -> WuXingStateVector:
        """Calculate target state for CNNs"""
        water, wood, fire, earth, metal = current_state.state

        if desired_outcome == 'accuracy':
            # For CNNs, focus on increasing Wood (feature extraction) and Metal
            target_water = water * 0.9
            target_wood = min(wood * 1.2, 1.0)  # Enhance pattern recognition
            target_fire = min(fire * 1.1, 1.0)
            target_earth = max(earth * 0.9, 0.3)
            target_metal = min(metal * 1.3, 1.0)  # Enhance classification precision

        elif desired_outcome == 'feature_extraction':
            # Enhance the model's ability to extract good features
            target_water = min(water * 1.1, 1.0)
            target_wood = min(wood * 1.4, 1.0)  # Significantly enhance pattern recognition
            target_fire = fire * 0.9
            target_earth = earth
            target_metal = metal

        else:
            # Use base adapter implementation for other outcomes
            return super().calculate_target_state(current_state, desired_outcome)

        return WuXingStateVector(
            target_water, target_wood, target_fire, target_earth, target_metal
        )


class RNNAdapter(BaseAdapter):
    """Adapter for Recurrent Neural Networks (RNN, LSTM, GRU)"""

    def __init__(self, task_type: str = 'sequence', data_type: str = 'text'):
        super().__init__(model_type='rnn', task_type=task_type, data_type=data_type)

    def measure_wood_strength(self, model: nn.Module) -> float:
        """Measure Wood element (architecture complexity) for RNNs"""
        # Check for recurrent layer types
        has_lstm = any(isinstance(m, nn.LSTM) for m in model.modules())
        has_gru = any(isinstance(m, nn.GRU) for m in model.modules())
        has_rnn = any(isinstance(m, nn.RNN) for m in model.modules())

        # Count recurrent layers
        recurrent_layers = sum(1 for m in model.modules()
                               if isinstance(m, (nn.LSTM, nn.GRU, nn.RNN)))

        # No recurrent layers found
        if recurrent_layers == 0:
            return super().measure_wood_strength(model)

        # Layer complexity score
        complexity_score = 0.0
        if has_lstm:
            complexity_score += 0.6  # LSTM is most complex
        elif has_gru:
            complexity_score += 0.4  # GRU is moderately complex
        elif has_rnn:
            complexity_score += 0.2  # Simple RNN

        # Normalize layer count
        layer_score = min(1.0, recurrent_layers / 5)  # Scale to common RNN depths

        # Get hidden dimensions if available
        hidden_dim = 0
        for m in model.modules():
            if isinstance(m, (nn.LSTM, nn.GRU, nn.RNN)):
                if hasattr(m, 'hidden_size'):
                    hidden_dim = max(hidden_dim, m.hidden_size)

        # Normalize hidden dimension
        hidden_score = min(1.0, np.log(hidden_dim + 1) / np.log(2048)) if hidden_dim > 0 else 0.5

        # Calculate architecture complexity
        wood_strength = 0.4 * complexity_score + 0.3 * layer_score + 0.3 * hidden_score

        return min(1.0, wood_strength)

    def measure_fire_strength(self, model: nn.Module) -> float:
        """Measure Fire element (training intensity) for RNNs"""
        # For RNNs, check for bidirectionality and sequence length handling
        has_bidirectional = False

        for m in model.modules():
            if isinstance(m, (nn.LSTM, nn.GRU, nn.RNN)):
                if hasattr(m, 'bidirectional') and m.bidirectional:
                    has_bidirectional = True
                    break

        # Bidirectional models process sequences more intensively
        bidirectional_score = 0.8 if has_bidirectional else 0.5

        # Default fire score that can be enhanced
        base_fire = super().measure_fire_strength(model)

        # Enhance base score with RNN-specific factors
        return min(1.0, base_fire * 0.6 + bidirectional_score * 0.4)

    def map_param_to_element(self, param_name: str) -> int:
        """Map RNN parameter names to elements"""
        if 'weight_ih' in param_name:
            return 0  # Input-to-hidden weights handle data (Water)
        elif 'weight_hh' in param_name:
            return 1  # Hidden-to-hidden weights handle pattern growth (Wood)
        elif 'bias_ih' in param_name or 'bias_hh' in param_name:
            return 2  # Biases help with transformation (Fire)
        elif 'lstm.cell' in param_name:
            return 3  # Cell state maintenance is stability (Earth)
        elif 'encoder' in param_name or 'embedding' in param_name:
            return 0  # Input encoding (Water)
        elif 'decoder' in param_name or 'output' in param_name:
            return 4  # Output processing (Metal)

        # Default to base adapter implementation
        return super().map_param_to_element(param_name)


class TransformerAdapter(BaseAdapter):
    """Adapter for Transformer-based models"""

    def __init__(self, task_type: str = 'language', data_type: str = 'text'):
        super().__init__(model_type='transformer', task_type=task_type, data_type=data_type)

    def measure_wood_strength(self, model: nn.Module) -> float:
        """Measure Wood element (architecture complexity) for Transformers"""
        # Look for transformer-specific components
        attention_heads = 0
        transformer_layers = 0

        for name, module in model.named_modules():
            # Check for attention modules
            if any(x in name.lower() for x in ['attention', 'self_attn', 'mha']):
                # Try to get number of attention heads
                if hasattr(module, 'num_heads'):
                    attention_heads = max(attention_heads, module.num_heads)
                elif hasattr(module, 'n_head'):
                    attention_heads = max(attention_heads, module.n_head)
                elif hasattr(module, 'num_attention_heads'):
                    attention_heads = max(attention_heads, module.num_attention_heads)

            # Check for transformer layers
            if any(x in name.lower() for x in ['encoder_layer', 'decoder_layer', 'transformer_layer']):
                transformer_layers += 1

        # If no transformer-specific components found
        if attention_heads == 0 and transformer_layers == 0:
            return super().measure_wood_strength(model)

        # If we found attention heads but not layers, estimate layers
        if transformer_layers == 0:
            # Count modules that might be transformer layers
            possible_layers = sum(1 for name, _ in model.named_modules()
                                  if any(x in name.lower() for x in ['layer', 'block', 'encoder', 'decoder']))
            transformer_layers = max(1, possible_layers // 3)  # Rough estimate

        # If we found layers but not heads, estimate heads
        if attention_heads == 0:
            attention_heads = 8  # Common default

        # Calculate scores
        head_score = min(1.0, attention_heads / 32)  # Normalize by common max heads
        layer_score = min(1.0, transformer_layers / 24)  # Normalize by common max layers

        # Calculate architecture complexity
        wood_strength = 0.5 * head_score + 0.5 * layer_score

        return min(1.0, wood_strength)

    def measure_fire_strength(self, model: nn.Module) -> float:
        """Measure Fire element (training intensity) for Transformers"""
        # For transformers, look at FFN dimensions and activation types
        ffn_ratio = 0.0
        count = 0

        for name, module in model.named_modules():
            # Look for feed-forward networks in transformer blocks
            if isinstance(module, nn.Linear) and any(x in name.lower() for x in ['ffn', 'mlp', 'feedforward']):
                if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                    if 'up' in name.lower() or 'expand' in name.lower():
                        # Expansion ratio (typically 4x in transformers)
                        ratio = module.out_features / module.in_features
                        ffn_ratio = max(ffn_ratio, ratio)
                    count += 1

        # If no FFN information found
        if count == 0 or ffn_ratio == 0.0:
            ffn_score = 0.5  # Default
        else:
            ffn_score = min(1.0, ffn_ratio / 8.0)  # Normalize by common max ratio

        # Check for activation functions used in transformer
        gelu_count = sum(1 for m in model.modules() if isinstance(m, nn.GELU))
        relu_count = sum(1 for m in model.modules() if isinstance(m, nn.ReLU))

        # GELU is common in modern transformers and provides stronger gradients
        activation_score = 0.5
        if gelu_count > 0:
            activation_score = 0.8
        elif relu_count > 0:
            activation_score = 0.7

        # Calculate fire strength
        fire_strength = 0.5 * ffn_score + 0.5 * activation_score

        return min(1.0, fire_strength)

    def map_param_to_element(self, param_name: str) -> int:
        """Map Transformer parameter names to elements"""
        if 'embedding' in param_name or 'token' in param_name or 'embed' in param_name:
            return 0  # Token embeddings (Water)
        elif 'attention' in param_name or 'attn' in param_name:
            if 'query' in param_name or 'key' in param_name:
                return 1  # Attention query/key mechanisms (Wood)
            elif 'value' in param_name or 'output' in param_name:
                return 2  # Attention value/output (Fire)
            else:
                return 1  # Other attention components (Wood)
        elif 'ffn' in param_name or 'feed_forward' in param_name or 'mlp' in param_name:
            if 'up' in param_name or 'expand' in param_name:
                return 2  # Expansion (Fire)
            elif 'down' in param_name or 'contract' in param_name:
                return 4  # Contraction (Metal)
            else:
                return 2  # Other FFN (Fire)
        elif 'norm' in param_name or 'ln' in param_name:
            return 3  # Layer norm (Earth)
        elif 'head' in param_name or 'classifier' in param_name or 'output' in param_name:
            return 4  # Output layer (Metal)

        # Default to base adapter implementation
        return super().map_param_to_element(param_name)


def get_adapter(model_type: str, task_type: str = 'generic',
                data_type: str = 'generic') -> BaseAdapter:
    """
    Factory function to get the appropriate adapter

    Args:
        model_type: Type of model architecture
                   ('cnn', 'rnn', 'lstm', 'gru', 'transformer', 'bert', 'gpt', etc.)
        task_type: Type of ML task
                   ('classification', 'regression', 'sequence', 'text-classification', etc.)
        data_type: Type of data
                   ('image', 'text', 'tabular', 'timeseries', etc.)

    Returns:
        Appropriate adapter for the specified model and task
    """
    # Normalize inputs
    model_type = model_type.lower()
    task_type = task_type.lower()
    data_type = data_type.lower()

    # Hugging Face Transformer models
    if any(x in model_type for x in ['bert', 'gpt', 'roberta', 't5', 'bart', 'xlnet', 'distilbert', 'electra']):
        from .huggingface_adapter import HuggingFaceAdapter
        return HuggingFaceAdapter(model_type=model_type, task_type=task_type, data_type=data_type)

    # Generic transformer models
    elif any(x in model_type for x in ['transformer']):
        return TransformerAdapter(task_type=task_type, data_type=data_type)

    # CNN-based models
    elif any(x in model_type for x in ['cnn', 'conv', 'resnet', 'vgg', 'inception', 'efficientnet']):
        return CNNAdapter(task_type=task_type, data_type=data_type)

    # RNN-based models
    elif any(x in model_type for x in ['rnn', 'lstm', 'gru', 'recurrent']):
        return RNNAdapter(task_type=task_type, data_type=data_type)

    # Default adapter
    else:
        return BaseAdapter(model_type=model_type, task_type=task_type, data_type=data_type)