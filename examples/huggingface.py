"""
Example script for using WuXingMechanism with Hugging Face Transformers
"""

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from wuxing_mechanism import WuXingMechanism
from wuxing_mechanism.adapters import HuggingFaceAdapter
from wuxing_mechanism.integration import HuggingFaceTrainer


def load_data_and_model(model_name="distilbert-base-uncased", max_length=128):
    """
    Load dataset, tokenizer, and model

    Args:
        model_name: Name of the pre-trained model
        max_length: Maximum sequence length

    Returns:
        Tuple of (model, tokenizer, train_dataset, eval_dataset, data_collator)
    """
    # Load dataset (SST-2 for sentiment analysis)
    dataset = load_dataset("glue", "sst2")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    # Tokenize the datasets
    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length" if max_length else False,
            truncation=True,
            max_length=max_length,
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Sample datasets for faster example
    train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    eval_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(200))

    # Create data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return model, tokenizer, train_dataset, eval_dataset, data_collator


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics

    Args:
        eval_pred: Tuple of (predictions, labels)

    Returns:
        Dictionary of metrics
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1) if predictions.shape[-1] > 1 else predictions

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    accuracy = accuracy_score(labels, predictions)

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def direct_intervention_example():
    """
    Example of direct intervention on a Hugging Face model
    """
    print("\nDirect Intervention Example")
    print("---------------------------")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and data
    model, tokenizer, train_dataset, eval_dataset, data_collator = load_data_and_model()
    model = model.to(device)

    # Create sample batches for analysis
    from torch.utils.data import DataLoader
    sample_loader = DataLoader(
        eval_dataset, batch_size=8, collate_fn=data_collator
    )

    analysis_batches = []
    for i, batch in enumerate(sample_loader):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        analysis_batches.append(batch)
        if i >= 2:
            break

    # Get adapter and create mechanism
    adapter = HuggingFaceAdapter(model_type='distilbert', task_type='text-classification')
    criterion = torch.nn.CrossEntropyLoss()

    mechanism = adapter.create_wuxing_mechanism(model, analysis_batches, criterion, device)

    # Analyze current state
    print("\nInitial Wu Xing State:")
    mechanism.visualize_current_state()
    plt.show()

    # Identify mechanism points
    print("\nIdentifying Mechanism Points...")
    mechanism_points = mechanism.identify_mechanism_points(top_k=5)

    print("\nTop 5 Mechanism Points:")
    for name, info in mechanism_points.items():
        print(f"{name}: Strength={info['strength']:.4f}, Impact={info['impact']:.4f}, Energy={info['energy']:.4f}")

    # Visualize mechanism points
    mechanism.visualize_mechanism_points(mechanism_points)
    plt.show()

    # Design and apply intervention
    print("\nDesigning and applying intervention...")
    intervention = mechanism.design_intervention(target_outcome='accuracy', intervention_magnitude=0.02)

    if intervention['success']:
        print(f"Designed intervention for parameter: {intervention['intervention_point']}")

        result = mechanism.apply_intervention(intervention)

        if result['success']:
            if result['intervention_kept']:
                print(f"Intervention applied successfully to {result['intervention_point']}")
                print(f"Loss improvement: {result['improvement']:.6f}")
            else:
                print("Intervention did not improve performance and was reverted")
        else:
            print(f"Failed to apply intervention: {result.get('error', 'unknown error')}")
    else:
        print(f"Failed to design intervention: {intervention.get('error', 'unknown error')}")

    # Final state
    print("\nFinal Wu Xing State:")
    mechanism.visualize_current_state()
    plt.show()

    return model, mechanism


def trainer_example():
    """
    Example using the HuggingFaceTrainer
    """
    print("\nHuggingFace Trainer Example")
    print("--------------------------")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and data
    model, tokenizer, train_dataset, eval_dataset, data_collator = load_data_and_model()

    # Create HuggingFaceTrainer
    trainer = HuggingFaceTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        model_type='distilbert',
        task_type='text-classification',
        device=device
    )

    # Initial state visualization
    print("\nInitial Wu Xing State:")
    trainer.visualize_current_state()
    plt.show()

    # Train with WuXing interventions
    print("\nTraining with WuXing interventions...")
    history = trainer.train(
        num_epochs=3,
        learning_rate=2e-5,
        wuxing_interventions=True,
        intervention_frequency=1,
        target_outcome='accuracy',
        intervention_magnitude=0.02,
        output_dir='huggingface_output'
    )

    # Visualize training history
    print("\nTraining History:")
    trainer.visualize_training_history()

    # Visualize Wu Xing evolution
    print("\nWu Xing Evolution:")
    trainer.visualize_wuxing_evolution()

    return trainer, model, history


def main():
    """Main function to run examples"""
    # Create output directory
    os.makedirs('huggingface_output', exist_ok=True)

    # Run direct intervention example
    model, mechanism = direct_intervention_example()

    # Run trainer example
    trainer, model, history = trainer_example()

    print("\nExamples completed!")


if __name__ == "__main__":
    main()