import torch
from matplotlib import pyplot as plt
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import time
import os

from ..core import WuXingMechanism
from ..adapters import get_adapter


class WuXingTrainer:
    """
    Trainer class that integrates WuXingMechanism with PyTorch training loops
    """

    def __init__(self, model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: Callable,
                 optimizer: torch.optim.Optimizer,
                 device: Optional[torch.device] = None,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 model_type: str = 'generic',
                 task_type: str = 'generic',
                 data_type: str = 'generic'):
        """
        Initialize WuXingTrainer

        Args:
            model: PyTorch neural network model
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            criterion: Loss function
            optimizer: Optimizer
            device: Computation device (CPU or GPU)
            scheduler: Learning rate scheduler
            model_type: Type of model architecture
            task_type: Type of task
            data_type: Type of data
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scheduler = scheduler

        # Move model to device
        self.model = self.model.to(self.device)

        # Create WuXingMechanism
        adapter = get_adapter(model_type, task_type, data_type)
        self.mechanism = adapter.create_wuxing_mechanism(model, val_loader, criterion, device)

        # Initialize training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'wuxing_states': [],
            'interventions': []
        }

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update statistics
            total_loss += loss.item()

            # For classification tasks
            if len(target.shape) == 1 or target.shape[1] == 1:
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

            # Print progress
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                      f'({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()

        # Calculate epoch average loss and accuracy
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total if total > 0 else 0

        print(f'Train Epoch: {epoch}, Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')

        return avg_loss, accuracy

    def validate(self) -> Tuple[float, float]:
        """
        Validate the model

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()

                # For classification tasks
                if len(target.shape) == 1 or target.shape[1] == 1:
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()

        val_loss /= len(self.val_loader)
        accuracy = 100. * correct / total if total > 0 else 0

        print(f'Validation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')

        return val_loss, accuracy

    def apply_wuxing_intervention(self, target_outcome: str = 'accuracy',
                                  magnitude: float = 0.05) -> Dict:
        """
        Apply a WuXing intervention to the model

        Args:
            target_outcome: Desired outcome
            magnitude: Intervention magnitude

        Returns:
            Dictionary with intervention results
        """
        # Design intervention
        intervention_plan = self.mechanism.design_intervention(
            target_outcome=target_outcome,
            intervention_magnitude=magnitude
        )

        if not intervention_plan['success']:
            print(f"Failed to design intervention: {intervention_plan.get('error', 'unknown error')}")
            return intervention_plan

        # Apply intervention
        intervention_result = self.mechanism.apply_intervention(intervention_plan)

        if intervention_result['success']:
            if intervention_result['intervention_kept']:
                print(f"Intervention applied to {intervention_result['intervention_point']}")
                print(f"Loss change: {intervention_result['improvement']:.6f}")
            else:
                print("Intervention did not improve performance and was reverted")
        else:
            print(f"Failed to apply intervention: {intervention_result.get('error', 'unknown error')}")

        return intervention_result

    def train(self, epochs: int, wuxing_interventions: bool = True,
              intervention_frequency: int = 3,
              target_outcome: str = 'accuracy',
              intervention_magnitude: float = 0.05,
              save_path: Optional[str] = None) -> Dict:
        """
        Train the model with optional WuXing interventions

        Args:
            epochs: Number of epochs to train
            wuxing_interventions: Whether to apply WuXing interventions
            intervention_frequency: Apply intervention every N epochs
            target_outcome: Desired outcome for interventions
            intervention_magnitude: Magnitude of interventions
            save_path: Path to save best model

        Returns:
            Dictionary with training history
        """
        best_val_loss = float('inf')
        start_time = time.time()

        # Initialize with current Wu Xing state
        self.history['wuxing_states'].append(self.mechanism.current_state)

        for epoch in range(1, epochs + 1):
            # Train epoch
            train_loss, train_acc = self.train_epoch(epoch)

            # Validate
            val_loss, val_acc = self.validate()

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)

            # Save best model
            if val_loss < best_val_loss and save_path is not None:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f'Saved best model to {save_path}')

            # Apply WuXing intervention
            if wuxing_interventions and epoch % intervention_frequency == 0:
                print("\nPerforming WuXing intervention...")

                # Re-assess Wu Xing state
                self.mechanism._assess_wuxing_state()
                self.history['wuxing_states'].append(self.mechanism.current_state)

                # Apply intervention
                intervention_result = self.apply_wuxing_intervention(
                    target_outcome=target_outcome,
                    magnitude=intervention_magnitude
                )

                self.history['interventions'].append(intervention_result)

        # Final Wu Xing state
        self.mechanism._assess_wuxing_state()
        self.history['wuxing_states'].append(self.mechanism.current_state)

        # Calculate training time
        training_time = time.time() - start_time

        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Best validation loss: {best_val_loss:.4f}")

        return self.history

    def visualize_training_history(self, show_interventions: bool = True):
        """
        Visualize training history

        Args:
            show_interventions: Whether to mark intervention points
        """
        import matplotlib.pyplot as plt

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot training and validation loss
        epochs = range(1, len(self.history['train_loss']) + 1)
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Training loss')
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validation loss')

        # Mark intervention points
        if show_interventions:
            intervention_epochs = []
            for i, epoch in enumerate(epochs):
                if i < len(self.history['interventions']):
                    intervention = self.history['interventions'][i]
                    if intervention and intervention.get('intervention_kept', False):
                        intervention_epochs.append(epoch)
                        ax1.axvline(x=epoch, color='g', linestyle='--', alpha=0.5)
                        ax1.text(epoch, min(self.history['train_loss']),
                                 f"{intervention['intervention_point']}",
                                 rotation=90, ha='right')

        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot training and validation accuracy
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Training accuracy')
        ax2.plot(epochs, self.history['val_acc'], 'r-', label='Validation accuracy')

        # Mark intervention points
        if show_interventions:
            for epoch in intervention_epochs:
                ax2.axvline(x=epoch, color='g', linestyle='--', alpha=0.5)

        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return fig

    def visualize_wuxing_evolution(self):
        """
        Visualize the evolution of Wu Xing elements during training
        """
        if len(self.history['wuxing_states']) <= 1:
            print("Not enough Wu Xing states to visualize evolution")
            return None

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot Wu Xing element evolution
        elements = self.history['wuxing_states'][0].element_names()
        epochs = range(len(self.history['wuxing_states']))

        for i, element in enumerate(elements):
            values = [state.state[i] for state in self.history['wuxing_states']]
            ax1.plot(epochs, values, 'o-', label=element)

        ax1.set_xlabel('Intervention Points')
        ax1.set_ylabel('Element Strength')
        ax1.set_title('Evolution of Wu Xing Elements During Training')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot validation loss
        if self.history['val_loss']:
            epochs_loss = range(1, len(self.history['val_loss']) + 1)
            ax2.plot(epochs_loss, self.history['val_loss'], 'r-', label='Validation Loss')

            # Mark intervention points
            intervention_markers = []
            marker_epochs = []
            for i, epoch in enumerate(epochs_loss):
                if i < len(self.history['interventions']):
                    intervention = self.history['interventions'][i]
                    if intervention and intervention.get('intervention_kept', False):
                        intervention_markers.append(self.history['val_loss'][i])
                        marker_epochs.append(epoch)

            if intervention_markers:
                ax2.scatter(marker_epochs, intervention_markers, c='g', s=100,
                            label='Interventions', zorder=5)

            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Loss')
            ax2.set_title('Validation Loss with Intervention Points')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return fig

    def visualize_current_state(self):
        """
        Visualize the current Wu Xing state of the model
        """
        return self.mechanism.visualize_current_state(
            title=f"Current Wu Xing State"
        )

    def save_visualization(self, fig, filename: str):
        """
        Save a visualization figure to file

        Args:
            fig: Matplotlib figure
            filename: Output filename
        """
        if fig is not None:
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {filename}")

    def export_history(self, filename: str):
        """
        Export training history to file

        Args:
            filename: Output filename
        """
        import json

        # Create a serializable version of history
        export_history = {
            'train_loss': self.history['train_loss'],
            'val_loss': self.history['val_loss'],
            'train_acc': self.history['train_acc'],
            'val_acc': self.history['val_acc'],
            'wuxing_states': [
                [float(x) for x in state.state] for state in self.history['wuxing_states']
            ],
            'interventions': []
        }

        # Process interventions
        for intervention in self.history['interventions']:
            if intervention:
                export_intervention = {
                    'intervention_point': intervention.get('intervention_point', ''),
                    'improvement': float(intervention.get('improvement', 0)),
                    'kept': bool(intervention.get('intervention_kept', False))
                }
                export_history['interventions'].append(export_intervention)

        # Save to file
        with open(filename, 'w') as f:
            json.dump(export_history, f, indent=2)

        print(f"Exported training history to {filename}")