import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from m6a_powernet.dataloader import RNAData
from m6a_powernet.model import M6APowerNet

import wandb
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Suppress specific warning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

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
    parser.add_argument('--warmup_epochs', type=int, help="Number of warm-up epochs to gradually increase the learning rate.")
    args = parser.parse_args()

    return args

# python -m m6a_powernet.training --dataset_path data/dataset0.json.gz --labels_path data/data.info.labelled --checkpoints_directory m6a_powernet/checkpoints --num_epochs 100 --learning_rate 0.0005
def start(args: argparse.Namespace):
    wandb.init(project="m6a-powernet-training", name="First run", config={
        "Total Epochs": args.num_epochs,
        "Max Learning Rate": args.learning_rate,
        "Warmup Epochs": args.warmup_epochs,
        "Dataset Path": args.dataset_path,
        "Labels Path": args.labels_path
    })

    model = M6APowerNet()

    criterion = nn.BCELoss()
    optimiser = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Cosine Annealing LR scheduler, after the warm-up period
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=(args.num_epochs - args.warmup_epochs))

    # Load dataset
    dataset = RNAData(dataset_path=args.dataset_path, label_path=args.labels_path, seed=888)

    train_dataloader = dataset.set_train_mode().data_loader()

    model.train()

    # Training
    for epoch in range(args.num_epochs):
        
        # Linear warm-up for the first `warmup_epochs` epochs
        if epoch < args.warmup_epochs:
            warmup_lr = args.learning_rate * (epoch + 1) / args.warmup_epochs
            for param_group in optimiser.param_groups:
                param_group['lr'] = warmup_lr
        else:
            # Use Cosine Annealing LR scheduler after warm-up
            cosine_scheduler.step()
            warmup_lr = optimiser.param_groups[0]['lr']  # Get the updated learning rate

        total_loss = 0
        with tqdm(train_dataloader, unit="step") as tepoch:
            tepoch.set_description(f"Epoch [{epoch + 1}/{args.num_epochs}], Learning Rate: {optimiser.param_groups[0]['lr']}")

            for inputs, label in tepoch:

                # Clear model gradients
                optimiser.zero_grad()

                # DataLoader adds a batch dimension, but we do not support batch size more than 1 at the moment.
                inputs = [x.squeeze(0) for x in inputs]
                label = label.squeeze(0)

                output = model(inputs)

                # Calculate loss
                loss = criterion(output, label)
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                
                # Update weights
                optimiser.step()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f'Epoch [{epoch + 1}/{args.num_epochs}] - Train Loss: {avg_train_loss}')

        metrics = {
            "Epoch": epoch + 1,
            "Train Loss": avg_train_loss,
            "Learning Rate": warmup_lr
        }
        val_metrics = get_validation(model, dataset, criterion)
        metrics.update(val_metrics)
        wandb.log(metrics)

        model_save_path = os.path.join(args.checkpoints_directory, f"m6a_powernet_model_epoch{epoch + 1}.pth")
        torch.save(model.state_dict(), model_save_path)
        # print(f'Model saved to {model_save_path}')

    print('Training finished.')
    wandb.finish()

def get_validation(model, dataset, criterion):
    model.eval()
    validation_dataloader = dataset.set_validation_mode().data_loader()

    running_val_loss = 0.0
    outputs = []
    labels = []
    with torch.no_grad():
        for inputs, label in validation_dataloader:
            inputs = [x.squeeze(0) for x in inputs]
            label = label.squeeze(0)

            output = model(inputs)

            val_loss = criterion(output, label)
            running_val_loss += val_loss.item()

            outputs.append(output.item())
            labels.append(label.item())
    
    avg_val_loss = running_val_loss / len(validation_dataloader)

    thresholded_outputs = [1 if output >= 0.5 else 0 for output in outputs]

    # Calculate precision, recall, F1 score
    precision = precision_score(labels, thresholded_outputs)
    recall = recall_score(labels, thresholded_outputs)
    f1 = f1_score(labels, thresholded_outputs)

    # Calculate ROC AUC and PR AUC
    roc_auc = roc_auc_score(labels, outputs)
    pr_auc = average_precision_score(labels, outputs)

    print(f'Validation Loss: {avg_val_loss} | Precision: {precision} | Recall: {recall} | F1: {f1} | ROC AUC: {roc_auc} | PR_AUC: {pr_auc}')

    val_metrics = {
        "Validation Loss": avg_val_loss,
        "Validation Precision": precision,
        "Validation Recall": recall,
        "Validation F1 Score": f1,
        "Validation ROC AUC": roc_auc,
        "Validation PR AUC": pr_auc
    }

    # Setting model and dataset back to train mode for next training.
    model.train()
    dataset.set_train_mode()

    return val_metrics

if __name__ == "__main__":
    args = parse_arguments()
    start(args)
    