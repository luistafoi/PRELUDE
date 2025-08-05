# scripts/train.py

import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from sklearn.metrics import roc_auc_score
from tqdm import tqdm 
import csv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.args import read_args
from dataloaders.data_loader import PRELUDEDataset
from dataloaders.data_generator import DataGenerator
from dataloaders.feature_loader import FeatureLoader
from models.tools import HetAgg

# --- Helper: Memory-Efficient Dataset for Link Prediction ---
class LinkPredictionDataset(Dataset):
    """Custom Dataset for link prediction to handle large numbers of links."""
    def __init__(self, pos_links, neg_links):
        u_nodes = [p[0] for p in pos_links] + [n[0] for n in neg_links]
        v_nodes = [p[1] for p in pos_links] + [n[1] for n in neg_links]
        labels = [1.0] * len(pos_links) + [0.0] * len(neg_links)

        self.u_nodes = np.array(u_nodes, dtype=np.int64)
        self.v_nodes = np.array(v_nodes, dtype=np.int64)
        self.labels = np.array(labels, dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.u_nodes[idx], self.v_nodes[idx], self.labels[idx]

# --- Helper: Evaluation Function ---
def evaluate(model, dataloader, generator, device, dataset):
    """Evaluates the model on a given dataloader and returns ROC-AUC."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for u_gids_batch, v_gids_batch, labels_batch in dataloader:
            u_lids = [dataset.nodes['type_map'][gid.item()][1] for gid in u_gids_batch]
            v_lids = [dataset.nodes['type_map'][gid.item()][1] for gid in v_gids_batch]
            
            u_type = dataset.nodes['type_map'][u_gids_batch[0].item()][0]
            drug_type_id = dataset.node_name2type['drug']

            if u_type == drug_type_id:
                drug_lids, cell_lids = u_lids, v_lids
            else:
                drug_lids, cell_lids = v_lids, u_lids
            
            preds = model.link_prediction_forward(drug_lids, cell_lids, generator)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
            
    if len(all_labels) == 0 or len(np.unique(all_labels)) < 2:
        return 0.0
        
    return roc_auc_score(all_labels, all_preds)

# --- Main Training Function ---
def main():
    args = read_args()

    # --- Setup ---
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Components ---
    NEIGHBOR_FILE = os.path.join(args.data_dir, "train_neighbors.txt")
    if not os.path.exists(NEIGHBOR_FILE):
        print(f"Error: Neighbor file not found at '{NEIGHBOR_FILE}'")
        print("Please run 'python scripts/generate_neighbors.py' first.")
        return

    dataset = PRELUDEDataset(args.data_dir)
    feature_loader = FeatureLoader(dataset, device)
    generator = DataGenerator(args.data_dir).load_train_neighbors(NEIGHBOR_FILE)

    model = HetAgg(args, dataset, feature_loader, device).to(device)
    model.setup_link_prediction(drug_type_name="drug", cell_type_name="cell")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- Prepare DataLoaders using pre-defined splits ---
    print("\nLoading pre-split and GMM-filtered training/validation data...")
    
    train_pos = dataset.links['train_pos']
    train_neg = dataset.links['train_neg']
    train_dataset = LinkPredictionDataset(train_pos, train_neg)
    train_loader = DataLoader(train_dataset, batch_size=args.mini_batch_s, shuffle=True)
    print(f"  > Created training loader with {len(train_pos)} positive and {len(train_neg)} negative pairs.")
    
    valid_pos = dataset.links['valid_pos']
    valid_neg = dataset.links['valid_neg']
    valid_dataset = LinkPredictionDataset(valid_pos, valid_neg)
    valid_loader = DataLoader(valid_dataset, batch_size=args.mini_batch_s)
    print(f"  > Created validation loader with {len(valid_pos)} positive and {len(valid_neg)} negative pairs.")

    # --- Training Loop ---
    best_valid_auc = 0.0
    patience_counter = 0
    save_path = os.path.join(args.save_dir, f"{args.model_name}.pth")
    log_path = os.path.join(args.save_dir, f"{args.model_name}_log.csv")

    with open(log_path, 'w', newline='') as log_file:
        csv_writer = csv.writer(log_file)
        csv_writer.writerow(['epoch', 'lp_loss', 'rw_loss', 'val_auc'])

        print("\n--- Starting Model Training ---")

        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            model.train()
            total_lp_loss = 0
            total_rw_loss = 0

            # Curriculum learning for node isolation
            max_isolation_ratio = 0.2
            current_isolation_ratio = max_isolation_ratio * (epoch / (args.epochs -1)) if args.epochs > 1 else 0
            
            # Curriculum learning for LP loss weight
            current_lambda = 1.0
            if args.use_lp_curriculum:
                max_lambda = args.lp_loss_lambda
                current_lambda = max_lambda * (epoch / (args.epochs - 1)) if args.epochs > 1 else max_lambda

            # Phase 1: Supervised Link Prediction
            lp_iterator = tqdm(train_loader, desc="Phase 1: Link Prediction")
            for u_gids, v_gids, labels in lp_iterator:
                optimizer.zero_grad()

                u_lids = [dataset.nodes['type_map'][gid.item()][1] for gid in u_gids]
                v_lids = [dataset.nodes['type_map'][gid.item()][1] for gid in v_gids]
                labels = labels.to(device)

                u_type = dataset.nodes['type_map'][u_gids[0].item()][0]
                drug_type_id = dataset.node_name2type['drug']

                if u_type == drug_type_id:
                    drug_lids, cell_lids = u_lids, v_lids
                else:
                    drug_lids, cell_lids = v_lids, u_lids

                loss = model.link_prediction_loss(drug_lids, cell_lids, labels, generator, current_isolation_ratio)
                
                weighted_loss = current_lambda * loss
                weighted_loss.backward()
                
                optimizer.step()
                total_lp_loss += loss.item()
                lp_iterator.set_postfix({"Loss": total_lp_loss / (lp_iterator.n + 1)})

            # Phase 2: Self-Supervised Random Walk
            if args.use_rw_loss:
                rw_pairs = generator.generate_rw_triples(walk_length=args.walk_length, window_size=args.window_size, num_walks=args.num_walks)

                if rw_pairs:
                    all_node_ids = list(dataset.id2node.keys())
                    rw_batch = [(c, p, random.choice(all_node_ids)) for c, p in rw_pairs]

                    rw_iterator = tqdm(range(0, len(rw_batch), args.mini_batch_s), desc="Phase 2: Random Walk")
                    for i in rw_iterator:
                        optimizer.zero_grad()
                        batch = rw_batch[i : i + args.mini_batch_s]
                        if not batch: continue
                        loss_rw = model.self_supervised_rw_loss(batch, generator)
                        loss_rw.backward()
                        optimizer.step()
                        total_rw_loss += loss_rw.item()
                        rw_iterator.set_postfix({"Loss": total_rw_loss / (rw_iterator.n + 1)})

            avg_lp_loss = total_lp_loss / len(train_loader) if train_loader else 0
            avg_rw_loss = 0
            if args.use_rw_loss and rw_pairs:
                num_rw_batches = (len(rw_batch) + args.mini_batch_s - 1) // args.mini_batch_s
                avg_rw_loss = total_rw_loss / num_rw_batches if num_rw_batches > 0 else 0

            # --- Validation & Early Stopping ---
            valid_auc_epoch = np.nan
            if (epoch + 1) % args.val_freq == 0:
                valid_auc_epoch = evaluate(model, valid_loader, generator, device, dataset)
                print(f"  Validation AUC: {valid_auc_epoch:.4f}")

                if valid_auc_epoch > best_valid_auc:
                    best_valid_auc = valid_auc_epoch
                    patience_counter = 0
                    torch.save(model.state_dict(), save_path)
                    print(f"  ✨ New best model saved to {save_path} (AUC: {best_valid_auc:.4f})")
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        print(f"  Stopping early as validation AUC has not improved for {patience_counter} checks.")
                        csv_writer.writerow([epoch + 1, avg_lp_loss, avg_rw_loss, valid_auc_epoch])
                        break

            csv_writer.writerow([epoch + 1, avg_lp_loss, avg_rw_loss, valid_auc_epoch])

    print("\n--- Training Complete ---")
    print(f"Best validation AUC: {best_valid_auc:.4f}")

if __name__ == "__main__":
    main()
