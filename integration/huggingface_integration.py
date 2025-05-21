import torch
from torch import nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import os
import time
import logging

from ..core import WuXingMechanism
from ..adapters import HuggingFaceAdapter
import matplotlib.pyplot as plt


class HuggingFaceTrainer:
    """
    Trainer class that integrates WuXingMechanism with Hugging Face's training pipeline
    for transformers models.
    """

    def __init__(self, model, tokenizer, train_dataset, eval_dataset,
                 data_collator=None, compute_metrics=None,
                 model_type: str = 'bert', task_type: str = 'text-classification',
                 device: Optional[torch.device] = None):
        """
        Initialize HuggingFaceTrainer

        Args:
            model: Hugging Face transformer model
            tokenizer: Tokenizer for the model
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            data_collator: Function to create batches
            compute_metrics: Function to compute evaluation metrics
            model_type: Specific type of transformer
            task_type: Type of NLP task
            device: Device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.compute_metrics = compute_metrics
        self.model_type = model_type
        self.task_type = task_type
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move model to device
        self.model = self.model.to(self.device)

        # Setup logger
        self.logger = logging.getLogger(__name__)

        # Determine loss function based on task
        self.criterion = self._get_criterion()

        # Create dataloaders
        from torch.utils.data import DataLoader
        if self.data_collator is None:
            from transformers import default_data_collator
            self.data_collator = default_data_collator

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=8,  # Default small batch size
            shuffle=True,
            collate_fn=self.data_collator
        )

        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=8,  # Default small batch size
            collate_fn=self.data_collator
        )

        # Create WuXingMechanism
        adapter = HuggingFaceAdapter(model_type=model_type, task_type=task_type, data_type='text')

        # Convert samples for mechanism analysis
        analysis_samples = []
        for i, batch in enumerate(self.eval_dataloader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            analysis_samples.append(batch)
            if i >= 2:  # Limit to a few batches
                break

        self.mechanism = adapter.create_wuxing_mechanism(model, analysis_samples, self.criterion, device)

        # Initialize training history
        self.history = {
            'train_loss': [],
            'eval_loss': [],
            'metrics': {},
            'wuxing_states': [],
            'interventions': []
        }

    def _get_criterion(self) -> Callable:
        """
        Determine appropriate loss function based on task

        Returns:
            Loss function
        """
        # Check model's config for num_labels
        config = getattr(self.model, 'config', None)

        if config is not None:
            num_labels = getattr(config, 'num_labels', None)
            problem_type = getattr(config, 'problem_type', None)

            if num_labels is not None:
                if num_labels == 1:
                    # Regression task
                    return nn.MSELoss()
                elif problem_type == 'multi_label_classification':
                    # Multi-label classification
                    return nn.BCEWithLogitsLoss()
                else:
                    # Single-label classification (most common)
                    return nn.CrossEntropyLoss()

        # Default for language modeling
        if 'lm' in self.task_type or self.task_type == 'causal-lm' or self.task_type == 'masked-lm':
            return nn.CrossEntropyLoss()

        # Default fallback
        return nn.CrossEntropyLoss()

    def train(self, num_epochs: int = 3, learning_rate: float = 5e-5,
              wuxing_interventions: bool = True,
              intervention_frequency: int = 1,
              target_outcome: str = 'accuracy',
              intervention_magnitude: float = 0.02,
              output_dir: Optional[str] = None) -> Dict:
        """
        Train the model with optional WuXing interventions

        Args:
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            wuxing_interventions: Whether to apply WuXing interventions
            intervention_frequency: Apply intervention every N epochs
            target_outcome: Desired outcome for interventions
            intervention_magnitude: Magnitude of interventions
            output_dir: Directory to save outputs

        Returns:
            Training history
        """
        # Create optimizer
        from transformers import AdamW, get_linear_schedule_with_warmup

        # Prepare optimizer and schedule
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01,
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

        # Total steps for scheduler
        total_steps = len(self.train_dataloader) * num_epochs
        warmup_steps = int(0.1 * total_steps)  # 10% warmup

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        # Initialize best metric for saving best model
        best_metric = float('-inf')
        best_model_path = None

        # Create output directory if needed
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Store current Wu Xing state
        self.history['wuxing_states'].append(self.mechanism.current_state)

        # Training loop
        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            self.logger.info(f"Starting epoch {epoch}/{num_epochs}")

            # Training phase
            self.model.train()
            total_train_loss = 0

            for step, batch in enumerate(self.train_dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else self.criterion(outputs.logits, batch['labels'])

                # Backward pass
                loss.backward()

                # Update parameters
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Update statistics
                total_train_loss += loss.item()

                # Log progress
                if step % 50 == 0:
                    self.logger.info(f"Epoch {epoch}, Step {step}: Loss = {loss.item():.4f}")

            # Calculate average training loss
            avg_train_loss = total_train_loss / len(self.train_dataloader)
            self.history['train_loss'].append(avg_train_loss)

            self.logger.info(f"Epoch {epoch}: Average training loss = {avg_train_loss:.4f}")

            # Evaluation phase
            self.model.eval()
            total_eval_loss = 0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in self.eval_dataloader:
                    # Move batch to device
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}

                    # Forward pass
                    outputs = self.model(**batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else self.criterion(outputs.logits, batch['labels'])

                    # Update statistics
                    total_eval_loss += loss.item()

                    # Store predictions and labels for metrics
                    if self.compute_metrics is not None:
                        if hasattr(outputs, 'logits'):
                            logits = outputs.logits

                            # Get predictions based on task
                            if logits.size(-1) == 1:  # Regression
                                preds = logits.squeeze(-1)
                            else:  # Classification
                                preds = logits.argmax(dim=-1)

                            all_preds.append(preds.cpu().numpy())
                            all_labels.append(batch['labels'].cpu().numpy())

            # Calculate average evaluation loss
            avg_eval_loss = total_eval_loss / len(self.eval_dataloader)
            self.history['eval_loss'].append(avg_eval_loss)

            self.logger.info(f"Epoch {epoch}: Evaluation loss = {avg_eval_loss:.4f}")

            # Compute metrics if available
            metrics = {}
            if self.compute_metrics is not None and all_preds and all_labels:
                # Concatenate predictions and labels
                all_preds = np.concatenate(all_preds, axis=0)
                all_labels = np.concatenate(all_labels, axis=0)

                # Compute metrics
                metrics = self.compute_metrics((all_preds, all_labels))

                # Store metrics in history
                for metric_name, metric_value in metrics.items():
                    if metric_name not in self.history['metrics']:
                        self.history['metrics'][metric_name] = []
                    self.history['metrics'][metric_name].append(metric_value)

                # Log metrics
                metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
                self.logger.info(f"Epoch {epoch}: Metrics: {metrics_str}")

                # Save best model
                if output_dir:
                    # Use first metric as criterion for best model
                    main_metric_name = list(metrics.keys())[0]
                    main_metric_value = metrics[main_metric_name]

                    if main_metric_value > best_metric:
                        best_metric = main_metric_value
                        best_model_path = os.path.join(output_dir, f"best_model")

                        # Save model and tokenizer
                        self.model.save_pretrained(best_model_path)
                        self.tokenizer.save_pretrained(best_model_path)

                        self.logger.info(f"New best model saved with {main_metric_name} = {main_metric_value:.4f}")

            # Apply WuXing intervention if enabled
            if wuxing_interventions and epoch % intervention_frequency == 0:
                self.logger.info("Applying WuXing intervention...")

                # Reassess current state
                current_elements = self.mechanism._assess_wuxing_state(epoch=epoch, total_epochs=num_epochs)
                print(current_elements)
                self.history['wuxing_states'].append(self.mechanism.current_state)

                # Design and apply intervention
                intervention_plan = self.mechanism.design_intervention(
                    target_outcome=target_outcome,
                    intervention_magnitude=intervention_magnitude
                )

                if intervention_plan['success']:
                    intervention_result = self.mechanism.apply_intervention(intervention_plan)

                    if intervention_result['success']:
                        if intervention_result['intervention_kept']:
                            self.logger.info(f"Intervention applied to {intervention_result['intervention_point']}")
                            self.logger.info(f"Loss improvement: {intervention_result['improvement']:.6f}")
                        else:
                            self.logger.info("Intervention did not improve performance and was reverted")

                        self.history['interventions'].append(intervention_result)
                    else:
                        self.logger.warning(
                            f"Failed to apply intervention: {intervention_result.get('error', 'unknown error')}")
                        self.history['interventions'].append(None)
                else:
                    self.logger.warning(
                        f"Failed to design intervention: {intervention_plan.get('error', 'unknown error')}")
                    self.history['interventions'].append(None)

        # Final state assessment
        self.mechanism._assess_wuxing_state(epoch=epoch, total_epochs=num_epochs)
        self.history['wuxing_states'].append(self.mechanism.current_state)
        # Save final model if requested
        if output_dir:
            final_model_path = os.path.join(output_dir, "final_model")
            self.model.save_pretrained(final_model_path)
            self.tokenizer.save_pretrained(final_model_path)
            self.logger.info(f"Final model saved to {final_model_path}")

            # Save training history
            import json

            # Create serializable version of history
            export_history = {
                'train_loss': self.history['train_loss'],
                'eval_loss': self.history['eval_loss'],
                'metrics': self.history['metrics'],
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
                else:
                    export_history['interventions'].append(None)

            # Save to file
            with open(os.path.join(output_dir, "training_history.json"), 'w') as f:
                json.dump(export_history, f, indent=2)

        # Calculate training time
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")

        return self.history

    def visualize_training_history(self, show_interventions: bool = True):
        """
        Visualize training history

        Args:
            show_interventions: Whether to mark intervention points
        """
        # Create figure
        num_metrics = len(self.history['metrics']) + 1  # +1 for loss
        fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 5 * num_metrics))

        if num_metrics == 1:
            axes = [axes]

        # Plot training and evaluation loss
        epochs = range(1, len(self.history['train_loss']) + 1)
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Training loss')
        axes[0].plot(epochs, self.history['eval_loss'], 'r-', label='Evaluation loss')

        # Mark intervention points
        if show_interventions:
            intervention_epochs = []
            for i, epoch in enumerate(epochs):
                if i < len(self.history['interventions']):
                    intervention = self.history['interventions'][i]
                    if intervention and intervention.get('intervention_kept', False):
                        intervention_epochs.append(epoch)
                        axes[0].axvline(x=epoch, color='g', linestyle='--', alpha=0.5)
                        axes[0].text(epoch, min(self.history['train_loss']),
                                     f"{intervention['intervention_point']}",
                                     rotation=90, ha='right')

        axes[0].set_title('Training and Evaluation Loss')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot metrics
        for i, (metric_name, metric_values) in enumerate(self.history['metrics'].items(), 1):
            axes[i].plot(epochs, metric_values, 'g-', label=metric_name)

            # Mark intervention points
            if show_interventions:
                for epoch in intervention_epochs:
                    axes[i].axvline(x=epoch, color='g', linestyle='--', alpha=0.5)

            axes[i].set_title(f'{metric_name} Over Time')
            axes[i].set_xlabel('Epochs')
            axes[i].set_ylabel(metric_name)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return fig

    def visualize_wuxing_evolution(self):
        """
        Visualize the evolution of Wu Xing elements during training
        """
        if len(self.history['wuxing_states']) <= 1:
            self.logger.warning("Not enough Wu Xing states to visualize evolution")
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

        # Plot evaluation loss
        if self.history['eval_loss']:
            epochs_loss = range(1, len(self.history['eval_loss']) + 1)
            ax2.plot(epochs_loss, self.history['eval_loss'], 'r-', label='Evaluation Loss')

            # Mark intervention points
            intervention_markers = []
            marker_epochs = []
            for i, epoch in enumerate(epochs_loss):
                if i < len(self.history['interventions']):
                    intervention = self.history['interventions'][i]
                    if intervention and intervention.get('intervention_kept', False):
                        intervention_markers.append(self.history['eval_loss'][i])
                        marker_epochs.append(epoch)

            if intervention_markers:
                ax2.scatter(marker_epochs, intervention_markers, c='g', s=100,
                            label='Interventions', zorder=5)

            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Loss')
            ax2.set_title('Evaluation Loss with Intervention Points')
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
            title=f"Current Wu Xing State of {self.model_type.upper()} Model"
        )