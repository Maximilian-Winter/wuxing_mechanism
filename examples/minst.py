"""
Example script for using WuXingMechanism with MNIST dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from wuxing_mechanism import WuXingMechanism, WuXingStateVector
from wuxing_mechanism.adapters import get_adapter
from wuxing_mechanism.integration import WuXingTrainer


# Define CNN for MNIST
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def load_mnist_data(batch_size=64):
    """Load MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def visualize_mnist_sample(train_loader):
    """Visualize a sample of MNIST data"""
    # Get a batch of training data
    examples = iter(train_loader)
    example_data, example_targets = next(examples)

    # Create a grid of sample images
    fig = plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title(f"Label: {example_targets[i]}")
        plt.xticks([])
        plt.yticks([])
    plt.suptitle("MNIST Sample Images")
    plt.show()

    return fig


def example_with_wuxing_intervention():
    """Example of using WuXingMechanism for direct intervention"""
    print("WuXingMechanism Framework - MNIST Example")
    print("------------------------------------------")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("\nLoading MNIST dataset...")
    batch_size = 64
    train_loader, test_loader = load_mnist_data(batch_size)

    # Create and train a model
    print("\nCreating and training initial model...")
    model = MNIST_CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train for a few epochs to get a decent starting point
    initial_epochs = 3
    for epoch in range(1, initial_epochs + 1):
        # Training
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        # Testing
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
              f' ({100. * correct / len(test_loader.dataset):.2f}%)')

    # Initialize WuXingMechanism
    print("\nInitializing WuXingMechanism framework...")
    # Create a smaller dataset for mechanism analysis (for efficiency)
    analysis_loader = [(next(iter(train_loader))) for _ in range(3)]

    # Get adapter for CNNs
    adapter = get_adapter(model_type='cnn', task_type='classification', data_type='image')

    # Create WuXingMechanism
    mechanism = adapter.create_wuxing_mechanism(model, analysis_loader, criterion, device)

    # Assess current state
    print("\nAssessing current Wu Xing state...")
    current_state = mechanism.visualize_current_state()
    plt.show()

    # Identify mechanism points
    print("\nIdentifying mechanism points...")
    mechanism_points = mechanism.identify_mechanism_points(top_k=5)

    print("\nTop 5 Mechanism Points:")
    for name, info in mechanism_points.items():
        print(f"{name}: Strength={info['strength']:.4f}, Impact={info['impact']:.4f}, Energy={info['energy']:.4f}")

    # Visualize mechanism points
    mechanism.visualize_mechanism_points(mechanism_points)
    plt.show()

    # Perform strategic optimization
    print("\nPerforming strategic optimization...")
    optimization_results = mechanism.strategic_optimization(
        target_outcome='accuracy', iterations=5, magnitude=0.05
    )

    # Visualize evolution
    mechanism.visualize_wuxing_evolution(optimization_results)
    plt.show()

    # Final state
    print("\nFinal Wu Xing State:")
    mechanism.visualize_current_state()
    plt.show()

    # Final test
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    final_accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Final test results: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
          f' ({final_accuracy:.2f}%)')

    return model, mechanism, optimization_results


def example_with_wuxing_trainer():
    """Example of using WuXingTrainer for integrated training"""
    print("WuXingTrainer - MNIST Example")
    print("-----------------------------")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("\nLoading MNIST dataset...")
    batch_size = 64
    train_loader, test_loader = load_mnist_data(batch_size)

    # Create model, criterion, and optimizer
    model = MNIST_CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create WuXingTrainer
    print("\nCreating WuXingTrainer...")
    trainer = WuXingTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        model_type='cnn',
        task_type='classification',
        data_type='image'
    )

    # Train with WuXing interventions
    print("\nTraining with WuXing interventions...")
    history = trainer.train(
        epochs=8,
        wuxing_interventions=True,
        intervention_frequency=2,
        target_outcome='accuracy',
        intervention_magnitude=0.05,
        save_path='mnist_wuxing_model.pt'
    )

    # Visualize training history
    print("\nVisualizing training history...")
    trainer.visualize_training_history()

    # Visualize Wu Xing evolution
    print("\nVisualizing Wu Xing evolution...")
    trainer.visualize_wuxing_evolution()

    # Export results
    print("\nExporting results...")
    trainer.export_history('mnist_wuxing_history.json')

    return trainer, model, history


def main():
    """Main function to run the examples"""
    # Create output directory
    os.makedirs('output', exist_ok=True)

    # Visualize MNIST samples
    train_loader, _ = load_mnist_data()
    sample_fig = visualize_mnist_sample(train_loader)
    sample_fig.savefig('output/mnist_samples.png')

    # Run direct intervention example
    print("\n\nRunning example with direct WuXing intervention...\n")
    model1, mechanism, opt_results = example_with_wuxing_intervention()

    # Run trainer example
    print("\n\nRunning example with WuXingTrainer...\n")
    trainer, model2, history = example_with_wuxing_trainer()

    print("\nExamples completed. Results saved to 'output' directory.")


if __name__ == "__main__":
    main()