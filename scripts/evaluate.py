# scripts/evaluate.py

import sys
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from collections import defaultdict
import random

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.args import read_args
from dataloaders.data_loader import PRELUDEDataset
from dataloaders.data_generator import DataGenerator
from dataloaders.feature_loader import FeatureLoader
from models.tools import HetAgg
from scripts.train import LinkPredictionDataset # Re-use the dataset class

def evaluate_model(model, dataloader, generator, device, dataset):
    """
    Evaluates the model on a given dataloader and returns a dictionary of metrics.
    Calculates ROC-AUC, F1-Score, and Mean Reciprocal Rank (MRR).
    """
    model.eval()
    all_preds = []
    all_labels = []
    # For MRR, we group predictions by the cell to rank drugs for that cell
    preds_by_cell = defaultdict(list)

    with torch.no_grad():
        for u_gids_batch, v_gids_batch, labels_batch in dataloader:
            u_lids = [dataset.nodes['type_map'][gid.item()][1] for gid in u_gids_batch]
            v_lids = [dataset.nodes['type_map'][gid.item()][1] for gid in v_gids_batch]
            
            u_type = dataset.nodes['type_map'][u_gids_batch[0].item()][0]
            drug_type_id = dataset.node_name2type['drug']

            # Ensure drug_lids and cell_lids are correctly assigned
            if u_type == drug_type_id:
                drug_lids, cell_lids = u_lids, v_lids
                cell_gids = v_gids_batch.numpy()
            else:
                drug_lids, cell_lids = v_lids, u_lids
                cell_gids = u_gids_batch.numpy()
            
            preds = model.link_prediction_forward(drug_lids, cell_lids, generator)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

            # Group predictions by cell ID for MRR
            batch_preds = preds.cpu().numpy()
            batch_labels = labels_batch.cpu().numpy()
            for i in range(len(cell_gids)):
                cell_id = cell_gids[i]
                preds_by_cell[cell_id].append((batch_preds[i], batch_labels[i]))

    if len(all_labels) < 2 or len(np.unique(all_labels)) < 2:
        return {"ROC-AUC": 0.0, "F1-Score": 0.0, "MRR": 0.0}

    roc_auc = roc_auc_score(all_labels, all_preds)
    f1 = f1_score(np.array(all_labels), np.array(all_preds) > 0.5)

    # Calculate Mean Reciprocal Rank (MRR)
    reciprocal_ranks = []
    for cell_id, predictions in preds_by_cell.items():
        # Only consider cells that have at least one true positive link in the test set
        if any(label == 1.0 for score, label in predictions):
            # Sort predictions for this cell by score, descending
            predictions.sort(key=lambda x: x[0], reverse=True)
            # Find the rank of the first true positive drug
            for rank, (score, label) in enumerate(predictions):
                if label == 1.0:
                    reciprocal_ranks.append(1.0 / (rank + 1))
                    break
    
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    return {"ROC-AUC": roc_auc, "F1-Score": f1, "MRR": mrr}


def main():
    args = read_args()
    
    if not args.load_path:
        print("Error: Must provide a path to a trained model checkpoint using --load_path")
        return

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = PRELUDEDataset(args.data_dir)
    feature_loader = FeatureLoader(dataset, device)
    generator = DataGenerator(args.data_dir).load_train_neighbors(os.path.join(args.data_dir, "train_neighbors.txt"))
    
    model = HetAgg(args, dataset, feature_loader, device).to(device)
    model.setup_link_prediction(drug_type_name="drug", cell_type_name="cell")
    
    print(f"\nLoading trained model weights from: {args.load_path}")
    model.load_state_dict(torch.load(args.load_path, map_location=device, weights_only=True))
    
    # --- Prepare Test DataLoader using pre-defined splits ---
    print("\nLoading pre-split and GMM-filtered test data...")
    
    test_pos = dataset.links['test_pos']
    test_neg = dataset.links['test_neg']
    
    print(f"  > Evaluating on {len(test_pos)} positive and {len(test_neg)} negative test pairs.")
    test_dataset = LinkPredictionDataset(test_pos, test_neg)
    test_loader = DataLoader(test_dataset, batch_size=args.mini_batch_s)

    # --- Evaluate ---
    print("\nRunning final evaluation on the test set...")
    metrics = evaluate_model(model, test_loader, generator, device, dataset)

    print("\n--- Final Test Set Performance ---")
    print(f"  ROC-AUC:  {metrics['ROC-AUC']:.4f}")
    print(f"  F1-Score: {metrics['F1-Score']:.4f}")
    print(f"  MRR:      {metrics['MRR']:.4f}")
    print("----------------------------------")

if __name__ == "__main__":
    main()
