"""
WuXing Mechanism Framework
==========================

A framework for identifying and leveraging strategic mechanism points in neural networks
based on Wu Xing (Five Elements) philosophy.

The framework helps identify critical parameters in neural networks where minimal
intervention can produce maximal effects, based on the ancient Chinese concept of
"得机" (dé jī) - finding and seizing the optimal mechanism point.

Example
-------
>>> import torch
>>> from wuxing_mechanism import WuXingMechanism
>>> from wuxing_mechanism.adapters import get_adapter
>>>
>>> # Initialize your model and data
>>> model = YourModel()
>>> dataloader = YourDataLoader()
>>> criterion = YourLossCriterion()
>>>
>>> # Get appropriate adapter
>>> adapter = get_adapter(model_type='cnn', task_type='classification')
>>>
>>> # Create WuXing Mechanism
>>> mechanism = adapter.create_wuxing_mechanism(model, dataloader, criterion)
>>>
>>> # Analyze current state
>>> mechanism.visualize_current_state()
>>>
>>> # Identify mechanism points
>>> mechanism_points = mechanism.identify_mechanism_points(top_k=5)
>>>
>>> # Design and apply intervention
>>> intervention = mechanism.design_intervention(target_outcome='accuracy')
>>> result = mechanism.apply_intervention(intervention)
"""

__version__ = '0.1.0'

# Import core components
from .core.state_vector import WuXingStateVector
from .core.mechanism import WuXingMechanism

# Import adapters
from .adapters.adapters import BaseAdapter, CNNAdapter, RNNAdapter, TransformerAdapter, get_adapter

# Import integration utilities
from .integration.pytorch_integration import WuXingTrainer

# Import utility functions
from .utils.data_utils import DataAnalyzer
from .utils.model_utils import ModelAnalyzer