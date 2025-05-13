import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.patches as mpatches


def plot_wuxing_evolution(states: List['WuXingStateVector'],
                          losses: Optional[List[float]] = None,
                          title: Optional[str] = None) -> plt.Figure:
    """
    Visualize the evolution of Wu Xing elements over time

    Args:
        states: List of WuXingStateVector objects representing state evolution
        losses: Optional list of loss values corresponding to each state
        title: Optional title for the plot

    Returns:
        Matplotlib figure containing the plot
    """
    elements = states[0].element_names()
    iterations = range(len(states))

    if losses is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    else:
        fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Wu Xing element evolution
    for i, element in enumerate(elements):
        values = [state.state[i] for state in states]
        ax1.plot(iterations, values, 'o-', label=element)

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Element Strength')
    if title:
        ax1.set_title(title)
    else:
        ax1.set_title('Evolution of Wu Xing Elements')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot loss evolution if provided
    if losses is not None:
        ax2.plot(range(len(losses)), losses, 'o-', color='red')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss')
        ax2.set_title('Loss Evolution')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_mechanism_points(mechanism_points: Dict[str, Dict],
                          title: Optional[str] = None) -> plt.Figure:
    """
    Visualize identified mechanism points

    Args:
        mechanism_points: Dictionary of parameter names and their mechanism info
        title: Optional title for the plot

    Returns:
        Matplotlib figure containing the plot
    """
    if not mechanism_points:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No mechanism points to visualize.",
                ha='center', va='center', fontsize=12)
        return fig

    # Extract data
    names = list(mechanism_points.keys())
    strengths = [info['strength'] for info in mechanism_points.values()]
    impacts = [info['impact'] for info in mechanism_points.values()]
    energies = [info['energy'] for info in mechanism_points.values()]
    positions = [info.get('position', 0.5) for info in mechanism_points.values()]

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot mechanism strengths
    bars = ax1.bar(names, strengths, color='blue', alpha=0.7)
    if title:
        ax1.set_title(title)
    else:
        ax1.set_title('Mechanism Strength by Parameter')
    ax1.set_ylabel('Strength')
    ax1.set_xticklabels(names, rotation=45, ha='right')

    # Add grid
    ax1.grid(True, alpha=0.3)

    # Add values on bars
    for bar, strength in zip(bars, strengths):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{strength:.3f}',
                 ha='center', va='bottom', rotation=0)

    # Plot impact vs energy scatter
    scatter = ax2.scatter(energies, impacts, s=100, c=positions,
                          alpha=0.7, cmap='viridis')

    # Add parameter names as annotations
    for i, name in enumerate(names):
        ax2.annotate(name, (energies[i], impacts[i]),
                     xytext=(5, 5), textcoords='offset points')

    ax2.set_title('Impact vs Energy for Parameters')
    ax2.set_xlabel('Energy (resistance to change)')
    ax2.set_ylabel('Impact (effect of change)')
    ax2.grid(True, alpha=0.3)

    # Add colorbar for positions
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Positional Significance')

    plt.tight_layout()
    return fig


def plot_intervention_effects(model_name: str,
                              intervention_results: List[Dict],
                              metrics: Dict[str, List[float]],
                              title: Optional[str] = None) -> plt.Figure:
    """
    Visualize the effects of interventions on model performance

    Args:
        model_name: Name of the model for the plot title
        intervention_results: List of intervention result dictionaries
        metrics: Dictionary of metrics over time (e.g., loss, accuracy)
        title: Optional title for the plot

    Returns:
        Matplotlib figure containing the plot
    """
    # Extract intervention points
    iterations = [i for i, result in enumerate(intervention_results)
                  if result.get('intervention_kept', False)]
    points = [result.get('intervention_point', 'unknown')
              for result in intervention_results
              if result.get('intervention_kept', False)]

    improvements = [result.get('improvement', 0)
                    for result in intervention_results
                    if result.get('intervention_kept', False)]

    # Create figure
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]

    # Plot each metric
    for i, (metric_name, metric_values) in enumerate(metrics.items()):
        ax = axes[i]
        ax.plot(range(len(metric_values)), metric_values, 'o-', label=metric_name)

        # Mark intervention points
        for iter_idx, point, improvement in zip(iterations, points, improvements):
            if iter_idx < len(metric_values):
                ax.axvline(x=iter_idx, color='red', linestyle='--', alpha=0.5)
                ax.text(iter_idx, min(metric_values) + 0.1 * (max(metric_values) - min(metric_values)),
                        f"{point}\n{improvement:.4f}",
                        rotation=90, ha='right', va='bottom')

        ax.set_xlabel('Iteration')
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)
        ax.legend()

    if title:
        fig.suptitle(title)
    else:
        fig.suptitle(f"Intervention Effects on {model_name}")

    plt.tight_layout()
    return fig


def plot_empty_full_dynamics(dynamics: Dict[str, Dict],
                             title: Optional[str] = None) -> plt.Figure:
    """
    Visualize Empty-Full dynamics (虛實動態) in the neural network

    Args:
        dynamics: Dictionary with Empty-Full dynamics information
        title: Optional title for the plot

    Returns:
        Matplotlib figure containing the plot
    """
    if not dynamics:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No dynamics data to visualize.",
                ha='center', va='center', fontsize=12)
        return fig

    plt.figure(figsize=(12, 8))

    names = list(dynamics.keys())
    means = [info['mean_activation'] for info in dynamics.values()]
    stds = [info['std_activation'] for info in dynamics.values()]
    scores = [info['empty_full_score'] for info in dynamics.values()]
    states = [info['state'] for info in dynamics.values()]

    # Create color mapping for states
    colors = []
    for state in states:
        if state == 'Empty (虛)':
            colors.append('blue')
        elif state == 'Full (實)':
            colors.append('red')
        else:
            colors.append('purple')

    plt.scatter(means, stds, s=100 * np.array(scores), c=colors, alpha=0.7)

    # Add labels for most dynamic positions
    top_indices = np.argsort(scores)[-5:]  # Top 5
    for i in top_indices:
        plt.annotate(names[i],
                     (means[i], stds[i]),
                     xytext=(5, 5), textcoords='offset points')

    plt.xlabel('Mean Activation')
    plt.ylabel('Activation Standard Deviation')
    if title:
        plt.title(title)
    else:
        plt.title('Empty-Full Dynamics (虛實動態) in Neural Network')

    # Add legend
    empty_patch = mpatches.Patch(color='blue', label='Empty (虛)')
    full_patch = mpatches.Patch(color='red', label='Full (實)')
    trans_patch = mpatches.Patch(color='purple', label='Transitional')
    plt.legend(handles=[empty_patch, full_patch, trans_patch])

    plt.grid(True, alpha=0.3)
    fig = plt.gcf()
    return fig