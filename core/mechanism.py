import math

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable, Any

from .state_vector import WuXingStateVector
from ..core import visualization as viz


class WuXingMechanism:
    """
    WuXingMechanism: A framework for identifying and leveraging strategic intervention
    points (机) in neural networks, based on traditional Chinese philosophical concepts.

    This framework integrates the Wu Xing (Five Elements) theory with modern neural
    network optimization to identify critical parameters where minimal intervention
    can yield maximal effects.
    """

    def __init__(self, model: nn.Module, dataloader: Any,
                 criterion: Callable, device: Optional[torch.device] = None):
        """
        Initialize the WuXingMechanism framework.

        Args:
            model: PyTorch neural network model
            dataloader: DataLoader with validation data
            criterion: Loss function
            device: Device to use for computation (CPU or GPU)
        """
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device if device is not None else next(model.parameters()).device

        # Flag to detect dictionary-style dataloaders vs tuple-style
        self.dict_style_dataloader = self._check_dataloader_style()

        # Store hooks and activations
        self.hooks = []
        self.activations = {}
        self.gradients = {}

        # Register hooks to capture activations
        self._register_hooks()

        # Calculate baseline metrics
        self.baseline_loss = self._compute_baseline_loss()

        # Adapter override methods
        self._override_measure_water_strength = None
        self._override_measure_wood_strength = None
        self._override_measure_fire_strength = None
        self._override_measure_earth_strength = None
        self._override_measure_metal_strength = None
        self._override_map_param_to_element = None
        self._override_calculate_target_state = None
        self._override_design_intervention_vector = None

        # Store Wu Xing state
        self.current_state = None
        self._assess_wuxing_state()

    def _check_dataloader_style(self) -> bool:
        """
        Check if dataloader provides dictionary-style batches or tuple-style batches

        Returns:
            True if dictionary-style, False if tuple-style
        """
        # Peek at the first item in the dataloader
        for batch in self.dataloader:
            if isinstance(batch, dict):
                return True
            elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                return False
            # If it's another structure, assume it's custom and needs dictionary handling
            return True

        # Default to False if dataloader is empty
        return False

    def _register_hooks(self):
        """Register hooks to capture activations and gradients"""

        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()

            return hook

        def get_gradient(name):
            def hook(grad):
                self.gradients[name] = grad.detach()

            return hook

        # Clear any existing hooks
        self._clear_hooks()

        # Register forward hooks for activations
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LSTM, nn.GRU)):
                hook = module.register_forward_hook(get_activation(name))
                self.hooks.append(hook)

                # For parameters, register backward hooks
                for param_name, param in module.named_parameters():
                    if param.requires_grad:
                        full_name = f"{name}.{param_name}"
                        hook = param.register_hook(get_gradient(full_name))
                        self.hooks.append(hook)

    def _clear_hooks(self):
        """Clear all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def _compute_baseline_loss(self) -> float:
        """Compute the baseline loss of the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.dataloader:
                if self.dict_style_dataloader:
                    # Dictionary-style batch
                    outputs = self.model(**{k: v.to(self.device) for k, v in batch.items() if k != "labels"})
                    loss = self.criterion(outputs, batch)
                else:
                    # Tuple-style batch with (inputs, targets)
                    inputs, targets = batch
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                num_batches += 1

                # Limit to max 5 batches for efficiency
                if num_batches >= 5:
                    break

        return total_loss / max(1, num_batches)

    def _assess_wuxing_state(self, epoch=None, total_epochs=None):
        """
        Dynamically assess the current Wu Xing state with temporal awareness

        Args:
            epoch: Current epoch number
            total_epochs: Total number of epochs in training

        Returns:
            WuXingStateVector representing current system state
        """
        # Calculate training progress (0 to 1)
        if epoch is not None and total_epochs is not None:
            progress = (epoch - 1) / max(1, total_epochs - 1)
        else:
            progress = None

        # Analyze data quality (Water)
        water_strength = self._measure_water_strength(progress)

        # Analyze architecture complexity (Wood)
        wood_strength = self._measure_wood_strength(progress)

        # Analyze training intensity (Fire)
        fire_strength = self._measure_fire_strength(progress, water_strength, wood_strength)

        # Analyze regularization (Earth)
        earth_strength = self._measure_earth_strength(progress, fire_strength)

        # Analyze evaluation precision (Metal)
        metal_strength = self._measure_metal_strength(progress, earth_strength)

        # Create state vector
        self.current_state = WuXingStateVector(
            water_strength, wood_strength, fire_strength,
            earth_strength, metal_strength
        )

        # Apply generation and conquest influences
        if progress is not None:
            # As training progresses, generation and conquest effects intensify
            cycle_intensity = min(1.0, progress * 2)  # Reaches full strength by 50% of training

            generation_weights = {'generation': 0.1 * cycle_intensity,
                                  'conquest': 0.05 * cycle_intensity,
                                  'balance': 0.02}

            self.current_state = self.current_state.full_cycle_transformation(
                weights=generation_weights, time_step=0.5)

        return self.current_state

    def _measure_water_strength(self, progress=None):
        """Measure Water element (data quality/flow) with temporal awareness"""
        # If adapter override is available, use it
        if self._override_measure_water_strength is not None:
            base_water = self._override_measure_water_strength()
        else:
            # Implement basic water measurement here
            base_water = self._basic_water_measurement()

        if progress is None:
            return base_water

        # Water follows a U-shaped curve during training:
        # - High at beginning (fresh data)
        # - Decreases in middle (data patterns becoming routine)
        # - Increases at end (data mastery and flow)
        temporal_factor = 1.0 - 0.3 * math.sin(progress * math.pi)

        # Recent gradient flow indicates data quality
        gradient_flow = self._measure_gradient_flow_smoothness()

        # Calculate dynamic water strength
        water_strength = base_water * 0.6 + temporal_factor * 0.2 + gradient_flow * 0.2

        return min(1.0, max(0.1, water_strength))

    def _measure_wood_strength(self, progress=None):
        """Measure Wood element (architecture complexity/growth) with temporal awareness"""
        # If adapter override is available, use it
        if self._override_measure_wood_strength is not None:
            base_wood = self._override_measure_wood_strength()
        else:
            # Implement basic wood measurement here
            base_wood = self._basic_wood_measurement()

        if progress is None:
            return base_wood

        # Wood follows an early-peaking curve:
        # - Rises in early training (architecture learning its role)
        # - Peaks in early-middle training
        # - Gradually declines as other elements take over
        temporal_factor = math.sin(progress * math.pi * 0.5)  # Peaks at 50% then declines

        # Feature diversity indicates architecture utilization
        feature_diversity = self._measure_feature_representation_diversity()

        # Calculate dynamic wood strength
        wood_strength = base_wood * 0.6 + temporal_factor * 0.2 + feature_diversity * 0.2

        return min(1.0, max(0.1, wood_strength))

    def _measure_fire_strength(self, progress=None, water=None, wood=None):
        """Measure Fire element (training intensity) with temporal awareness"""
        # If adapter override is available, use it
        if self._override_measure_fire_strength is not None:
            base_fire = self._override_measure_fire_strength()
        else:
            # Implement basic fire measurement here
            base_fire = self._basic_fire_measurement()

        if progress is None:
            return base_fire

        # Fire follows a bell curve centered slightly past the middle:
        # - Low at beginning (warming up)
        # - Peaks after middle (intense transformation)
        # - Decreases toward end (cooling down)
        temporal_factor = math.sin(progress * math.pi * 0.7)  # Peaks around 70% of training

        # Current gradient magnitude indicates training intensity
        gradient_magnitude = self._measure_current_gradient_magnitude()

        # Fire is nourished by Wood and controlled by Water
        element_influence = 0
        if wood is not None and water is not None:
            element_influence = min(1.0, wood * 0.2 - water * 0.1)

        # Calculate dynamic fire strength
        fire_strength = base_fire * 0.5 + temporal_factor * 0.2 + gradient_magnitude * 0.2 + element_influence * 0.1

        return min(1.0, max(0.1, fire_strength))

    def _measure_earth_strength(self, progress=None, fire=None):
        """Measure Earth element (regularization/stability) with temporal awareness"""
        # If adapter override is available, use it
        if self._override_measure_earth_strength is not None:
            base_earth = self._override_measure_earth_strength()
        else:
            # Implement basic earth measurement here
            base_earth = self._basic_earth_measurement()

        if progress is None:
            return base_earth

        # Earth follows a sigmoid curve:
        # - Low at beginning (little stabilization needed)
        # - Rises in middle (stabilizing the learning)
        # - Plateaus toward end (maintaining stability)
        temporal_factor = 1.0 / (1.0 + math.exp(-10 * (progress - 0.5)))  # Sigmoid centered at 50%

        # Weight stability indicates regularization effectiveness
        weight_stability = self._measure_weight_stability()

        # Earth is nourished by Fire
        element_influence = 0
        if fire is not None:
            element_influence = min(1.0, fire * 0.2)

        # Calculate dynamic earth strength
        earth_strength = base_earth * 0.5 + temporal_factor * 0.2 + weight_stability * 0.2 + element_influence * 0.1

        return min(1.0, max(0.1, earth_strength))

    def _measure_metal_strength(self, progress=None, earth=None):
        """Measure Metal element (evaluation precision) with temporal awareness"""
        # If adapter override is available, use it
        if self._override_measure_metal_strength is not None:
            base_metal = self._override_measure_metal_strength()
        else:
            # Implement basic metal measurement here
            base_metal = self._basic_metal_measurement()

        if progress is None:
            return base_metal

        # Metal follows a late-rising curve:
        # - Low at beginning (poor evaluation)
        # - Gradually increases throughout training
        # - Highest at end (refined evaluation)
        temporal_factor = math.pow(progress, 1.5)  # Accelerating curve

        # Decision boundary clarity indicates evaluation precision
        decision_clarity = self._measure_decision_boundary_clarity()

        # Metal is nourished by Earth
        element_influence = 0
        if earth is not None:
            element_influence = min(1.0, earth * 0.2)

        # Calculate dynamic metal strength
        metal_strength = base_metal * 0.5 + temporal_factor * 0.2 + decision_clarity * 0.2 + element_influence * 0.1

        return min(1.0, max(0.1, metal_strength))

    def _map_param_to_element(self, param_name: str) -> int:
        """
        Map a parameter name to the Wu Xing element it primarily affects

        Args:
            param_name: Name of the parameter

        Returns:
            Integer index of the corresponding element (0-4)
        """
        # If adapter override is available, use it
        if self._override_map_param_to_element is not None:
            return self._override_map_param_to_element(param_name)

        # Generic mapping logic
        if 'conv' in param_name or 'linear' in param_name or 'fc' in param_name:
            if 'weight' in param_name:
                return 1  # Wood (architecture)
            else:
                return 2  # Fire (training)
        elif 'bn' in param_name or 'norm' in param_name or 'dropout' in param_name:
            return 3  # Earth (regularization)
        elif 'out' in param_name or 'logits' in param_name or param_name.endswith('bias'):
            return 4  # Metal (evaluation)
        else:
            return 0  # Water (default)

    def _basic_water_measurement(self):
        """Basic measurement of Water element without temporal factors"""
        # Analyze input data statistics
        data_stats = []

        with torch.no_grad():
            for batch in self.dataloader:
                if self.dict_style_dataloader:
                    # Dictionary-style batch
                    inputs = next(iter([v for k, v in batch.items()
                                        if isinstance(v, torch.Tensor) and k != "labels"]), None)
                    if inputs is not None:
                        inputs = inputs.to(self.device)
                else:
                    # Tuple-style batch
                    inputs, _ = batch
                    inputs = inputs.to(self.device)

                # Calculate basic statistics
                if isinstance(inputs, torch.Tensor):
                    flat_inputs = inputs.view(inputs.size(0), -1)
                    if flat_inputs.dtype not in [torch.float, torch.double, torch.half]:
                        flat_inputs = flat_inputs.float()

                    mean_val = flat_inputs.mean().item()
                    std_val = flat_inputs.std().item()
                    data_stats.append((mean_val, std_val))

                # Only check one batch
                break

        # If no statistics, return default
        if not data_stats:
            return 0.5

        # Calculate water strength based on data distribution
        mean_val, std_val = data_stats[0]

        # Normalized data has better flow (water strength)
        is_normalized = abs(mean_val) < 0.1 and 0.1 < std_val < 10

        return 0.7 if is_normalized else 0.4

    def _measure_gradient_flow_smoothness(self):
        """Measure the smoothness of gradient flow as a Water element indicator"""
        # Collect gradient statistics
        grad_stats = []

        for name, param in self.model.parameters():
            if param.grad is not None:
                # Calculate gradient mean and std
                grad_mean = param.grad.abs().mean().item()
                grad_std = param.grad.std().item()
                grad_stats.append((grad_mean, grad_std))

        if not grad_stats:
            return 0.5

        # Calculate coefficient of variation for gradients
        cv_values = [std / mean if mean > 0 else float('inf') for mean, std in grad_stats]
        cv_values = [cv for cv in cv_values if not math.isinf(cv)]

        if not cv_values:
            return 0.5

        # Lower CV indicates smoother gradient flow (better Water)
        avg_cv = sum(cv_values) / len(cv_values)
        smoothness = 1.0 / (1.0 + avg_cv)

        return smoothness

    def _basic_wood_measurement(self):
        """Basic measurement of Wood element without temporal factors"""
        # Count parameters and layers
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_layers = sum(1 for _ in self.model.modules() if isinstance(_, (nn.Conv2d, nn.Linear)))

        # Normalize parameter count with log scaling
        param_score = min(1.0, np.log(total_params + 1) / np.log(10 ** 8))

        # Normalize layer count
        layer_score = min(1.0, np.log(total_layers + 1) / np.log(100))

        # Combine scores
        return 0.7 * param_score + 0.3 * layer_score

    def _measure_feature_representation_diversity(self):
        """Measure diversity of feature representations as a Wood element indicator"""
        # Collect activation statistics from hidden layers
        diversity_scores = []

        # Forward pass to get activations
        self.model.eval()
        with torch.no_grad():
            for batch in self.dataloader:
                if self.dict_style_dataloader:
                    # Process dictionary batch
                    inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                    self.model(**inputs)
                else:
                    # Process tuple batch
                    inputs, _ = batch
                    inputs = inputs.to(self.device)
                    self.model(inputs)

                # Calculate diversity from activations
                for name, activation in self.activations.items():
                    if isinstance(activation, torch.Tensor) and activation.dim() > 1:
                        # Reshape to 2D: (batch_size, features)
                        flat_act = activation.view(activation.size(0), -1)

                        # Calculate correlation matrix between features
                        if flat_act.size(1) > 1:  # Need at least 2 features
                            # Convert to float if needed
                            if flat_act.dtype not in [torch.float, torch.double, torch.half]:
                                flat_act = flat_act.float()

                            # Calculate correlation
                            centered = flat_act - flat_act.mean(dim=0, keepdim=True)
                            cov = centered.t() @ centered
                            norm = torch.diag(1.0 / torch.sqrt(torch.diag(cov) + 1e-10))
                            corr = norm @ cov @ norm

                            # Measure diversity as 1 - average absolute correlation
                            diversity = 1.0 - (torch.abs(corr).sum() - torch.abs(torch.diag(corr)).sum()) / (
                                        corr.numel() - corr.size(0))
                            diversity_scores.append(diversity.item())

                # Only check one batch
                break

        if not diversity_scores:
            return 0.5

        # Average diversity scores
        return sum(diversity_scores) / len(diversity_scores)

    def _basic_fire_measurement(self):
        """Basic measurement of Fire element without temporal factors"""
        # Analyze gradient magnitudes
        grad_magnitudes = []

        for param in self.model.parameters():
            if param.grad is not None:
                grad_magnitudes.append(param.grad.abs().mean().item())

        if not grad_magnitudes:
            return 0.5

        # Normalize gradient magnitude
        avg_grad = np.mean(grad_magnitudes)
        return min(1.0, avg_grad * 10)  # Scale appropriately

    def _measure_current_gradient_magnitude(self):
        """Measure current gradient magnitude as a Fire element indicator"""
        # This is similar to basic fire measurement but focused on recent gradients
        return self._basic_fire_measurement()

    def _basic_earth_measurement(self):
        """Basic measurement of Earth element without temporal factors"""
        # Check for regularization techniques
        has_bn = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
                     for m in self.model.modules())
        has_dropout = any(isinstance(m, nn.Dropout) for m in self.model.modules())

        # Calculate Earth strength
        earth_strength = 0.0
        if has_bn:
            earth_strength += 0.4
        if has_dropout:
            earth_strength += 0.4

        earth_strength += 0.2  # Base level
        return min(1.0, earth_strength)

    def _measure_weight_stability(self):
        """Measure weight stability as an Earth element indicator"""
        # Calculate statistics of weight changes
        stability_scores = []

        for name, param in self.model.parameters():
            if hasattr(param, '_prev_value'):
                # Calculate relative change
                prev = param._prev_value
                current = param.data
                rel_change = torch.norm(current - prev) / (torch.norm(prev) + 1e-10)
                stability = torch.exp(-rel_change * 10).item()  # Higher for smaller changes
                stability_scores.append(stability)

            # Update previous value
            param._prev_value = param.data.clone()

        if not stability_scores:
            return 0.5

        # Average stability scores
        return sum(stability_scores) / len(stability_scores)

    def _basic_metal_measurement(self):
        """Basic measurement of Metal element without temporal factors"""
        # Analyze output confidence
        confidence_scores = []

        self.model.eval()
        with torch.no_grad():
            for batch in self.dataloader:
                if self.dict_style_dataloader:
                    # Process dictionary batch
                    inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                    outputs = self.model(**inputs)
                else:
                    # Process tuple batch
                    inputs, _ = batch
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs)

                # Get logits if outputs is a dictionary
                if isinstance(outputs, dict) and "logits" in outputs:
                    outputs = outputs["logits"]

                # Calculate confidence
                if hasattr(outputs, "dim") and outputs.dim() > 1 and outputs.size(1) > 1:
                    # Apply softmax if needed
                    if outputs.min() < 0 or outputs.max() > 1:
                        outputs = torch.softmax(outputs, dim=1)

                    # Maximum probability as confidence
                    max_probs, _ = outputs.max(dim=1)
                    confidence_scores.extend(max_probs.cpu().numpy().tolist())

                # Only check one batch
                break

        if not confidence_scores:
            return 0.5

        # Average confidence
        avg_confidence = np.mean(confidence_scores)
        return avg_confidence

    def _measure_decision_boundary_clarity(self):
        """Measure decision boundary clarity as a Metal element indicator"""
        # Similar to basic metal measurement but focused on decision margins
        confidence_scores = []
        margin_scores = []

        self.model.eval()
        with torch.no_grad():
            for batch in self.dataloader:
                if self.dict_style_dataloader:
                    # Process dictionary batch
                    inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                    outputs = self.model(**inputs)
                else:
                    # Process tuple batch
                    inputs, _ = batch
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs)

                # Get logits if outputs is a dictionary
                if isinstance(outputs, dict) and "logits" in outputs:
                    outputs = outputs["logits"]

                # Calculate decision margins
                if hasattr(outputs, "dim") and outputs.dim() > 1 and outputs.size(1) > 1:
                    # Get top two probabilities
                    if outputs.min() < 0 or outputs.max() > 1:
                        probs = torch.softmax(outputs, dim=1)
                    else:
                        probs = outputs

                    # Get margin between top two predictions
                    top_values, _ = torch.topk(probs, k=2, dim=1)
                    margins = top_values[:, 0] - top_values[:, 1]
                    margin_scores.extend(margins.cpu().numpy().tolist())

                # Only check one batch
                break

        if not margin_scores:
            return 0.5

        # Average margin
        avg_margin = np.mean(margin_scores)
        return min(1.0, avg_margin * 2)  # Scale appropriately
    def identify_mechanism_points(self, top_k: int = 5) -> Dict[str, Dict]:
        """
        Identify the top mechanism points in the neural network

        Args:
            top_k: Number of top mechanism points to return

        Returns:
            Dictionary of parameter names and their mechanism strengths
        """
        mechanism_points = {}

        # Forward and backward pass to set up gradients
        self.model.train()
        for batch in self.dataloader:
            if self.dict_style_dataloader:
                # Dictionary-style batch
                # Move all tensors to device
                batch_on_device = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                                   for k, v in batch.items()}

                # Extract input and target parts
                input_dict = {k: v for k, v in batch_on_device.items() if k != "labels"}

                self.model.zero_grad()
                outputs = self.model(**input_dict)
                loss = self.criterion(outputs, batch_on_device)
                loss.backward()
            else:
                # Tuple-style batch
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.model.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()

            break

        # Calculate mechanism strength for each parameter
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Skip parameters with no gradient
                if param.grad is None:
                    continue

                # 1. Calculate impact (sensitivity)
                impact = self._calculate_parameter_impact(name, param)

                # 2. Calculate energy (resistance to change)
                energy = self._calculate_parameter_energy(name, param)

                # 3. Calculate positional significance
                position = self._calculate_positional_significance(name)

                # 4. Calculate mechanism strength
                if energy > 0:
                    strength = (impact / energy) * position
                else:
                    strength = impact * position

                mechanism_points[name] = {
                    'strength': float(strength),
                    'impact': float(impact),
                    'energy': float(energy),
                    'position': float(position)
                }

        # Sort and return top-k
        sorted_points = dict(sorted(mechanism_points.items(),
                                    key=lambda x: x[1]['strength'],
                                    reverse=True))

        return {k: sorted_points[k] for k in list(sorted_points.keys())[:top_k]}

    def _calculate_parameter_impact(self, param_name: str, param: torch.Tensor) -> float:
        """
        Calculate the impact of perturbing a parameter

        Args:
            param_name: Name of the parameter
            param: Parameter tensor

        Returns:
            Impact score
        """
        # Store original data
        original_data = param.data.clone()

        # Perturb parameter
        epsilon = 0.01
        param.data += epsilon * torch.randn_like(param.data)

        # Compute new loss
        perturbed_loss = self._compute_baseline_loss()

        # Restore original parameter
        param.data = original_data

        # Calculate impact as relative change in loss
        if self.baseline_loss != 0:
            impact = abs(perturbed_loss - self.baseline_loss) / self.baseline_loss
        else:
            impact = abs(perturbed_loss - self.baseline_loss)

        return float(impact)

    def _calculate_parameter_energy(self, param_name: str, param: torch.Tensor) -> float:
        """
        Calculate the energy (resistance to change) of a parameter

        Args:
            param_name: Name of the parameter
            param: Parameter tensor

        Returns:
            Energy score
        """
        # 1. Calculate rigidity based on gradient magnitude
        if param.grad is not None:
            # Convert grad to float if needed
            grad = param.grad
            if grad.dtype not in [torch.float, torch.double, torch.half]:
                grad = grad.float()

            grad_magnitude = grad.abs().mean().item()
            rigidity = 1.0 / (1.0 + 10 * grad_magnitude)  # Inverse relationship
        else:
            rigidity = 0.8  # Default if no gradient

        # 2. Calculate connectivity based on parameter name and layer position
        if 'weight' in param_name:
            connectivity = 0.8
        elif 'bias' in param_name:
            connectivity = 0.4
        else:
            connectivity = 0.6

        # Layer depth affects connectivity
        depth_indicator = sum(c.isdigit() for c in param_name)
        connectivity *= (1 + 0.1 * depth_indicator)  # Deeper layers have higher connectivity
        connectivity = min(1.0, connectivity)

        # 3. Calculate developmental stage based on parameter values
        # Convert param to float if needed
        p_data = param.data
        if p_data.dtype not in [torch.float, torch.double, torch.half]:
            p_data = p_data.float()

        mean_value = p_data.abs().mean().item()
        variance = p_data.var().item()

        if mean_value < 0.01:
            developmental_stage = 0.2  # Early stage
        elif variance > 0.1:
            developmental_stage = 0.5  # Middle stage
        else:
            developmental_stage = 0.8  # Late stage

        # Combine components with weights
        alpha, beta, gamma = 0.4, 0.3, 0.3
        energy = alpha * rigidity + beta * connectivity + gamma * developmental_stage

        return float(energy)

    def _calculate_positional_significance(self, param_name: str) -> float:
        """
        Calculate the positional significance of a parameter using Lo Shu magic square

        Args:
            param_name: Name of the parameter

        Returns:
            Positional significance score
        """
        # Define Lo Shu magic square
        lo_shu = np.array([
            [4, 9, 2],
            [3, 5, 7],
            [8, 1, 6]
        ])

        # Map parameter to Lo Shu position
        # Extract layer type
        if 'conv' in param_name.lower():
            row = 0  # Convolutional layers in top row
        elif 'linear' in param_name.lower() or 'fc' in param_name.lower():
            row = 1  # Linear layers in middle row
        else:
            row = 2  # Other layers in bottom row

        # Extract layer depth for column
        digits = ''.join(c for c in param_name if c.isdigit())
        layer_num = int(digits) if digits else 0
        col = layer_num % 3

        # Get Lo Shu value and normalize
        lo_shu_value = lo_shu[row, col]
        significance = lo_shu_value / 9.0  # Normalize to [0, 1]

        return float(significance)

    def visualize_mechanism_points(self, mechanism_points: Dict[str, Dict], title: Optional[str] = None):
        """
        Visualize the identified mechanism points

        Args:
            mechanism_points: Dictionary of parameter names and their mechanism info
            title: Optional title for the plot
        """
        fig = viz.plot_mechanism_points(mechanism_points, title)
        return fig

    def calculate_target_state(self, desired_outcome: str) -> WuXingStateVector:
        """
        Calculate target Wu Xing state based on desired outcome

        Args:
            desired_outcome: String indicating the desired outcome

        Returns:
            WuXingStateVector representing target system state
        """
        # If adapter override is available, use it
        if self._override_calculate_target_state is not None:
            return self._override_calculate_target_state(self.current_state, desired_outcome)

        if self.current_state is None:
            self._assess_wuxing_state()

        water, wood, fire, earth, metal = self.current_state.state

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

    def _design_intervention_vector(self, param_name: str, param: torch.Tensor,
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
        # If adapter override is available, use it
        if self._override_design_intervention_vector is not None:
            return self._override_design_intervention_vector(
                param_name, param, target_state, current_state, magnitude
            )

        # Calculate state change vector
        delta_state = target_state.state - current_state.state

        # Determine which element this parameter primarily affects
        element_index = self._map_param_to_element(param_name)
        delta_element = delta_state[element_index]

        # Create intervention based on parameter type
        if 'weight' in param_name:
            # For weights, scale based on existing parameter distribution
            std_value = param.data.std().item()

            # Direction based on desired element change
            if delta_element > 0:  # Increase element
                # For weights: decrease values if gradient is positive, increase if negative
                # (due to gradient descent's negative relationship)
                direction = -torch.sign(param.grad) if param.grad is not None else torch.sign(param.data)
            else:  # Decrease element
                direction = torch.sign(param.grad) if param.grad is not None else -torch.sign(param.data)

            intervention = direction * std_value * magnitude * abs(delta_element)

        elif 'bias' in param_name:
            # For biases, apply more uniform shift
            if delta_element > 0:  # Increase element
                direction = -torch.sign(param.grad) if param.grad is not None else torch.ones_like(param.data)
            else:  # Decrease element
                direction = torch.sign(param.grad) if param.grad is not None else -torch.ones_like(param.data)

            intervention = direction * magnitude * abs(delta_element)

        else:
            # Generic intervention
            if delta_element > 0:  # Increase element
                direction = -torch.sign(param.grad) if param.grad is not None else torch.randn_like(param.data)
            else:  # Decrease element
                direction = torch.sign(param.grad) if param.grad is not None else -torch.randn_like(param.data)

            intervention = direction * magnitude * abs(delta_element)

        return intervention

    def design_intervention(self, target_outcome: str = 'accuracy',
                            intervention_magnitude: float = 0.05) -> Dict:
        """
        Design an optimal intervention to achieve a target outcome

        Args:
            target_outcome: Desired outcome ('accuracy', 'generalization', or 'efficiency')
            intervention_magnitude: Overall magnitude of the intervention

        Returns:
            Dictionary with intervention details
        """
        # 1. Ensure we have current state
        if self.current_state is None:
            self._assess_wuxing_state()

        # 2. Calculate target state based on desired outcome
        target_state = self.calculate_target_state(target_outcome)

        # 3. Identify mechanism points
        mechanism_points = self.identify_mechanism_points(top_k=10)

        # 4. Calculate alignment with target state for each point
        alignments = {}
        for point_name, point_info in mechanism_points.items():
            alignment = self._calculate_alignment(point_name, target_state)
            alignments[point_name] = alignment

        # 5. Find optimal intervention point (highest strength * alignment)
        optimal_point = None
        max_score = 0.0

        for point_name, point_info in mechanism_points.items():
            score = point_info['strength'] * alignments[point_name]
            if score > max_score:
                max_score = score
                optimal_point = {
                    'name': point_name,
                    'mechanism_info': point_info,
                    'alignment': alignments[point_name],
                    'score': score
                }

        if optimal_point is None:
            return {'success': False, 'error': 'No suitable intervention point found'}

        # 6. Get the parameter
        param = None
        for name, p in self.model.named_parameters():
            if name == optimal_point['name']:
                param = p
                break

        if param is None:
            return {'success': False, 'error': f'Parameter {optimal_point["name"]} not found'}

        # 7. Design intervention vector
        intervention_vector = self._design_intervention_vector(
            optimal_point['name'], param, target_state, self.current_state, intervention_magnitude
        )

        # 8. Return intervention plan
        return {
            'success': True,
            'intervention_point': optimal_point['name'],
            'mechanism_strength': optimal_point['mechanism_info']['strength'],
            'alignment': optimal_point['alignment'],
            'score': optimal_point['score'],
            'intervention_vector': intervention_vector,
            'current_state': self.current_state,
            'target_state': target_state
        }

    def _calculate_alignment(self, param_name: str, target_state: WuXingStateVector) -> float:
        """
        Calculate alignment between a parameter and the target state

        Args:
            param_name: Name of the parameter
            target_state: Target WuXingStateVector

        Returns:
            Alignment score between 0 and 1
        """
        # Determine which element this parameter primarily affects
        element_index = self._map_param_to_element(param_name)

        # Calculate direction of desired change
        delta_desired = target_state.state[element_index] - self.current_state.state[element_index]

        # Get parameter gradient direction
        param = None
        for name, p in self.model.named_parameters():
            if name == param_name:
                param = p
                break

        if param is None or param.grad is None:
            return 0.0

        # Calculate gradient direction
        grad_direction = torch.sign(param.grad).float().mean().item()

        # Alignment is positive if grad direction matches desired state change
        if abs(delta_desired) < 1e-6:
            alignment = 0.0
        else:
            # Perfect alignment: opposite signs (negative gradient decreases param)
            alignment = 1.0 if (delta_desired * grad_direction < 0) else 0.0

        return float(alignment)

    def apply_intervention(self, intervention_plan: Dict) -> Dict:
        """
        Apply the designed intervention to the model

        Args:
            intervention_plan: Dictionary with intervention details from design_intervention

        Returns:
            Dictionary with results of intervention
        """
        if not intervention_plan['success']:
            return intervention_plan

        param_name = intervention_plan['intervention_point']
        intervention_vector = intervention_plan['intervention_vector']

        # 1. Evaluate before intervention
        before_loss = self.baseline_loss

        # 2. Apply intervention
        original_param = None
        for name, param in self.model.named_parameters():
            if name == param_name:
                # Store original parameter
                original_param = param.data.clone()

                # Apply intervention
                param.data += intervention_vector
                break

        if original_param is None:
            return {'success': False, 'error': f'Parameter {param_name} not found'}

        # 3. Evaluate after intervention
        after_loss = self._compute_baseline_loss()
        improvement = before_loss - after_loss

        # 4. Decide whether to keep intervention
        if improvement > 0:
            keep_intervention = True
            result = 'improved'
        else:
            # Rollback intervention
            for name, param in self.model.named_parameters():
                if name == param_name:
                    param.data = original_param
                    break
            keep_intervention = False
            result = 'reverted'

        # 5. Return results
        return {
            'success': True,
            'intervention_point': param_name,
            'before_loss': before_loss,
            'after_loss': after_loss,
            'improvement': improvement,
            'intervention_kept': keep_intervention,
            'result': result
        }

    def strategic_optimization(self, target_outcome: str = 'accuracy',
                               iterations: int = 3, magnitude: float = 0.05) -> Dict:
        """
        Perform strategic optimization through multiple interventions

        Args:
            target_outcome: Desired outcome ('accuracy', 'generalization', or 'efficiency')
            iterations: Number of intervention iterations
            magnitude: Overall magnitude of the interventions

        Returns:
            Dictionary with optimization results
        """
        results = {
            'iterations': [],
            'initial_state': self.current_state,
            'initial_loss': self.baseline_loss,
            'improvements': []
        }

        for i in range(iterations):
            # 1. Design intervention
            intervention_plan = self.design_intervention(
                target_outcome=target_outcome,
                intervention_magnitude=magnitude
            )

            if not intervention_plan['success']:
                print(f"Iteration {i + 1}: Failed to design intervention")
                continue

            # 2. Apply intervention
            intervention_result = self.apply_intervention(intervention_plan)

            if not intervention_result['success']:
                print(f"Iteration {i + 1}: Failed to apply intervention")
                continue

            # 3. Re-assess Wu Xing state
            self._assess_wuxing_state()

            # 4. Record results
            improvement = intervention_result.get('improvement', 0)
            results['iterations'].append({
                'iteration': i + 1,
                'intervention_point': intervention_result['intervention_point'],
                'improvement': improvement,
                'kept': intervention_result['intervention_kept'],
                'state': self.current_state
            })

            results['improvements'].append(improvement)

            # 5. Update baseline loss
            self.baseline_loss = self._compute_baseline_loss()

            print(f"Iteration {i + 1}: {'Improved' if improvement > 0 else 'No improvement'}, "
                  f"Loss: {self.baseline_loss:.6f}")

        # Calculate final results
        results['final_state'] = self.current_state
        results['final_loss'] = self.baseline_loss
        results['total_improvement'] = results['initial_loss'] - results['final_loss']
        results['success'] = results['total_improvement'] > 0

        return results

    def visualize_wuxing_evolution(self, optimization_results: Dict):
        """
        Visualize the evolution of Wu Xing elements during optimization

        Args:
            optimization_results: Results from strategic_optimization
        """
        # Extract states from results
        states = [optimization_results['initial_state']]
        for iteration in optimization_results['iterations']:
            if 'state' in iteration:
                states.append(iteration['state'])

        # Extract losses
        losses = [optimization_results['initial_loss']]
        for improvement in optimization_results['improvements']:
            losses.append(losses[-1] - improvement)

        # Visualize evolution
        fig = viz.plot_wuxing_evolution(states, losses)
        return fig

    def visualize_current_state(self, title: Optional[str] = None):
        """
        Visualize the current Wu Xing state of the model

        Args:
            title: Optional title for the plot
        """
        if self.current_state is None:
            self._assess_wuxing_state()

        fig = self.current_state.plot(title=title)
        return fig

    def analyze_empty_full_dynamics(self):
        """
        Analyze Empty-Full dynamics (虛實動態) in the neural network

        Returns:
            Dictionary with Empty-Full dynamics information
        """
        dynamics = {}

        # Forward pass on multiple batches to capture activation variation
        self.model.eval()

        activation_stats = {}

        with torch.no_grad():
            for batch in self.dataloader:
                if self.dict_style_dataloader:
                    # Dictionary-style batch
                    # Move all tensors to device
                    input_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                                  for k, v in batch.items() if k != "labels"}

                    self.model(**input_dict)
                else:
                    # Tuple-style batch
                    inputs, _ = batch
                    inputs = inputs.to(self.device)
                    self.model(inputs)

                # Collect activation statistics
                for name, activation in self.activations.items():
                    if name not in activation_stats:
                        activation_stats[name] = {'values': []}

                    # Convert to float if needed
                    if activation.dtype not in [torch.float, torch.double, torch.half, torch.complex64,
                                                torch.complex128]:
                        activation = activation.float()

                    act_mean = activation.mean().item()
                    activation_stats[name]['values'].append(act_mean)

        # Calculate Empty-Full dynamics
        for name, stats in activation_stats.items():
            if len(stats['values']) > 1:
                mean_activation = np.mean(stats['values'])
                std_activation = np.std(stats['values'])

                # Calculate positional significance (if available)
                position = 0.5
                if name.replace('.', '_') in self.model.named_modules():
                    param_name = next((p for p in self.model.named_parameters()
                                       if name in p), None)
                    if param_name:
                        position = self._calculate_positional_significance(param_name)

                # Empty-Full score: combination of position and activation variability
                empty_full_score = position * (1 + std_activation / (mean_activation + 1e-6))

                # Determine state
                if mean_activation < 0.3 and std_activation > 0.1:
                    state = 'Empty (虛)'  # Low activation with high variability
                elif mean_activation > 0.7 and std_activation < 0.1:
                    state = 'Full (實)'  # High activation with low variability
                else:
                    state = 'Transitional'  # In between states

                dynamics[name] = {
                    'mean_activation': mean_activation,
                    'std_activation': std_activation,
                    'empty_full_score': empty_full_score,
                    'state': state
                }

        # Sort by empty_full_score
        return dict(sorted(dynamics.items(), key=lambda x: x[1]['empty_full_score'], reverse=True))

    def visualize_empty_full_dynamics(self, dynamics=None):
        """
        Visualize Empty-Full dynamics (虛實動態) in the neural network

        Args:
            dynamics: Optional pre-computed dynamics. If None, will compute dynamics.
        """
        if dynamics is None:
            dynamics = self.analyze_empty_full_dynamics()

        fig = viz.plot_empty_full_dynamics(dynamics)
        return fig