import argparse
import torch

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from m6a_powernet.dataloader import RNAData
from m6a_powernet.model import M6APowerNet

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate m6A PowerNet.")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset file.")
    parser.add_argument('--labels_path', type=str, required=True, help="Path to the labels file.")
    parser.add_argument('--checkpoint_directory', type=str, required=True, help="Path to the weights of the model for evaluation.")
    args = parser.parse_args()

    return args

def start(args: argparse.Namespace):
    model = M6APowerNet()
    model.load_state_dict(torch.load(args.checkpoint_directory))

    # Set the model to evaluation mode
    model.eval()

    # Load dataset
    dataset = RNAData(dataset_path=args.dataset_path, label_path=args.labels_path, seed=888)
    test_dataloader = dataset.set_test_mode().data_loader()

    # Evaluation
    data_evaluated = 0
    outputs = []
    labels = []
    for inputs, label in test_dataloader:
        data_evaluated += 1
        if data_evaluated % 100 == 0:
            print(f"Completed {data_evaluated} points of evaluation.")

        # DataLoader adds a batch dimension, but we do not support batch size more than 1 at the moment.
        inputs = list(map(lambda x: x.squeeze(0), inputs))
        label = label.squeeze(0)

        output = torch.round(model(inputs))
        
        outputs.append(output.item())
        labels.append(label.item())
    
    # Convert predicted probabilities to binary predictions (threshold = 0.5)
    thresholded_outputs = [1 if output >= 0.5 else 0 for output in outputs]

    # Calculate precision, recall, F1 score
    precision = precision_score(labels, thresholded_outputs)
    recall = recall_score(labels, thresholded_outputs)
    f1 = f1_score(labels, thresholded_outputs)

    # Calculate ROC AUC and PR AUC
    roc_auc = roc_auc_score(labels, outputs)
    pr_auc = average_precision_score(labels, outputs)

    # Output the results
    print('='*80)
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'ROC AUC: {roc_auc:.4f}')
    print(f'PR AUC: {pr_auc:.4f}')
    print('='*80)

    print('Evaluation finished.')

if __name__ == "__main__":
    args = parse_arguments()
    start(args)
    