import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn

from m6a_powernet.dataloader import RNAData
from m6a_powernet.model import M6APowerNet

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run m6A PowerNet with the specified configuration.")
    parser.add_argument('--num_epochs', type=int, required=True, help="Number of times to iterate the training dataset for training.")
    parser.add_argument('--learning_rate', type=float, required=True, help="Learning rate of the optimiser for training")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset file.")
    parser.add_argument('--labels_path', type=str, required=True, help="Path to the labels file.")
    parser.add_argument('--checkpoints_directory', type=str, required=True, help="Path to where the checkpoints will be saved.")
    args = parser.parse_args()

    return args

def start(args: argparse.Namespace):
    model = M6APowerNet()
    criterion = nn.BCELoss()
    optimiser = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Load dataset
    dataset = RNAData(dataset_path=args.dataset_path, label_path=args.labels_path, seed=888)
    train_dataloader = dataset.set_train_mode().data_loader()

    # Set model to training mode
    model.train()

    # Training
    for epoch in range(args.num_epochs):
        running_loss = 0.0
        steps_taken = 0
        for inputs, label in train_dataloader:
            steps_taken += 1
            if steps_taken % 100 == 0:
                print(f"Completed {steps_taken} steps in epoch {epoch + 1}")

            # Clear model gradients
            optimiser.zero_grad()

            # DataLoader adds a batch dimension, but we do not support batch size more than 1 at the moment.
            inputs = list(map(lambda x: x.squeeze(0), inputs))
            label = label.squeeze(0)

            output = model(inputs)

            # Calculate loss
            loss = criterion(output, label)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimiser.step()
            
            # Accumulate the loss
            running_loss += loss.item()
        
        # Print the average loss after each epoch
        epoch_loss = running_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}/{args.num_epochs}, Loss: {epoch_loss:.4f}')
        model_save_path = os.path.join(args.checkpoints_directory, f"m6a_powernet_model_epoch{epoch + 1}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f'Model saved to {model_save_path}')

    print('Training finished.')

if __name__ == "__main__":
    args = parse_arguments()
    start(args)
    