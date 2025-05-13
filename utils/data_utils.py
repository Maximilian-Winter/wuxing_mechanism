import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any


class DataAnalyzer:
    """
    Utility class for analyzing data characteristics relevant to the Wu Xing framework
    """

    @staticmethod
    def analyze_tensor_dataset(dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Analyze a PyTorch DataLoader with tensor data

        Args:
            dataloader: PyTorch DataLoader

        Returns:
            Dictionary of data characteristics
        """
        stats = {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'missing_ratio': 0.0,
            'is_normalized': False,
            'class_balance': 0.0,
            'sample_count': 0
        }

        # Process a few batches
        sample_count = 0
        feature_means = []
        feature_stds = []
        feature_mins = []
        feature_maxs = []
        label_counts = {}

        for inputs, targets in dataloader:
            if isinstance(inputs, torch.Tensor):
                # Extract batch statistics
                batch_mean = inputs.mean().item()
                batch_std = inputs.std().item()
                batch_min = inputs.min().item()
                batch_max = inputs.max().item()

                feature_means.append(batch_mean)
                feature_stds.append(batch_std)
                feature_mins.append(batch_min)
                feature_maxs.append(batch_max)

                # Count samples
                sample_count += inputs.size(0)

                # Check for missing values (NaN)
                missing_count = torch.isnan(inputs).sum().item()
                stats['missing_ratio'] += missing_count / inputs.numel()

                # Count target classes
                if isinstance(targets, torch.Tensor):
                    for target in targets:
                        label = target.item() if target.numel() == 1 else target.argmax().item()
                        label_counts[label] = label_counts.get(label, 0) + 1

            # Limit to a few batches for efficiency
            if sample_count >= 1000:
                break

        if sample_count > 0:
            # Compute aggregated statistics
            stats['mean'] = np.mean(feature_means)
            stats['std'] = np.mean(feature_stds)
            stats['min'] = np.min(feature_mins)
            stats['max'] = np.max(feature_maxs)
            stats['missing_ratio'] /= sample_count
            stats['sample_count'] = sample_count

            # Check if data appears normalized
            stats['is_normalized'] = (abs(stats['mean']) < 0.1 and
                                      0.9 < stats['std'] < 1.1)

            # Calculate class balance if classification task
            if label_counts:
                total = sum(label_counts.values())
                class_props = [count / total for count in label_counts.values()]
                # Entropy-based measure of balance
                entropy = -sum(p * np.log(p) for p in class_props)
                max_entropy = np.log(len(label_counts))
                stats['class_balance'] = entropy / max_entropy if max_entropy > 0 else 1.0

        return stats

    @staticmethod
    def estimate_water_strength(data_stats: Dict[str, float]) -> float:
        """
        Estimate Water element strength based on data characteristics

        Args:
            data_stats: Dictionary of data statistics from analyze_tensor_dataset

        Returns:
            Water strength value between 0 and 1
        """
        # Initialize with default score
        water_strength = 0.5

        # Sample count contributes to strength (more data = stronger Water)
        sample_score = min(1.0, np.log(data_stats['sample_count'] + 1) / np.log(100000))

        # Normalization improves data quality
        normalization_score = 0.7 if data_stats['is_normalized'] else 0.3

        # Missing values reduce data quality
        missing_penalty = data_stats['missing_ratio'] * 0.5

        # Class balance improves data quality for classification
        balance_score = data_stats['class_balance'] if 'class_balance' in data_stats else 0.5

        # Combine factors
        water_strength = (0.3 * sample_score +
                          0.3 * normalization_score +
                          0.2 * (1.0 - missing_penalty) +
                          0.2 * balance_score)

        return min(1.0, max(0.0, water_strength))