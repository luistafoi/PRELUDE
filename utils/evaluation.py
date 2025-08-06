# utils/evaluation.py

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from collections import defaultdict

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
            # The "head" node for MRR is the cell
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
        if any(label == 1.0 for score, label in predictions):
            predictions.sort(key=lambda x: x[0], reverse=True)
            for rank, (score, label) in enumerate(predictions):
                if label == 1.0:
                    reciprocal_ranks.append(1.0 / (rank + 1))
                    break
    
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    return {"ROC-AUC": roc_auc, "F1-Score": f1, "MRR": mrr}