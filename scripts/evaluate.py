# scripts/evaluate.py

import sys
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from collections import defaultdict

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
    preds_by_head = defaultdict(list) # For MRR calculation

    with torch.no_grad():
        for u_gids_batch, v_gids_batch, labels_batch in dataloader:
            u_lids = [dataset.nodes['type_map'][gid.item()][1] for gid in u_gids_batch]
            v_lids = [dataset.nodes['type_map'][gid.item()][1] for gid in v_gids_batch]
            
            u_type = dataset.nodes['type_map'][u_gids_batch[0].item()][0]
            drug_type_id = dataset.node_name2type['drug']

            if u_type == drug_type_id:
                drug_lids, cell_lids = u_lids, v_lids
                head_gids = u_gids_batch.numpy()
            else:
                drug_lids, cell_lids = v_lids, u_lids
                head_gids = v_gids_batch.numpy()
            
            preds = model.link_prediction_forward(drug_lids, cell_lids, generator)
            
            # Store for overall metrics
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

            # Group by head node for MRR
            batch_preds = preds.cpu().numpy()
            batch_labels = labels_batch.cpu().numpy()
            for i in range(len(head_gids)):
                head_id = head_gids[i]
                preds_by_head[head_id].append((batch_preds[i], batch_labels[i]))

    if len(all_labels) < 2 or len(np.unique(all_labels)) < 2:
        return {"ROC-AUC": 0.0, "F1-Score": 0.0, "MRR": 0.0}

    # Calculate ROC-AUC and F1-Score
    roc_auc = roc_auc_score(all_labels, all_preds)
    f1 = f1_score(np.array(all_labels), np.array(all_preds) > 0.5)

    # Calculate Mean Reciprocal Rank (MRR)
    reciprocal_ranks = []
    for head_id, predictions in preds_by_head.items():
        if any(label == 1.0 for score, label in predictions):
            predictions.sort(key=lambda x: x[0], reverse=True)
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

    # --- Setup ---
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Components ---
    dataset = PRELUDEDataset(args.data_dir)
    feature_loader = FeatureLoader(dataset, device)
    generator = DataGenerator(args.data_dir).load_train_neighbors(os.path.join(args.data_dir, "train_neighbors.txt"))
    
    # --- Initialize Model and Load State ---
    model = HetAgg(args, dataset, feature_loader, device).to(device)
    model.setup_link_prediction(drug_type_name="drug", cell_type_name="cell")
    
    print(f"\nLoading trained model weights from: {args.load_path}")
    model.load_state_dict(torch.load(args.load_path, map_location=device))
    
    # --- Prepare Test DataLoader ---
    lp_relation_type = 0 # Assume drug-cell is type 0
    generator.build_edge_set()
    valid_cell_gids = dataset.valid_cell_ids

    # Load positive links from the dedicated test file
    test_pos = [(e[0], e[1]) for e in dataset.links['test'][lp_relation_type]]
    test_pos = [pair for pair in test_pos if pair[0] in valid_cell_gids]

    # Sample negative links and filter for valid cells
    all_test_neg = generator.sample_negative_pairs(lp_relation_type, num_samples=len(test_pos))
    test_neg = [pair for pair in all_test_neg if pair[0] in valid_cell_gids]

    print(f"Evaluating on {len(test_pos)} positive and {len(test_neg)} negative test pairs.")
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