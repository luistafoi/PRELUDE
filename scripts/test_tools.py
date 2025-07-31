import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.args import read_args
from dataloaders.data_loader import PRELUDEDataset
from dataloaders.data_generator import DataGenerator
from models.tools import HetAgg

# Load args
args = read_args()
args.use_vae_encoder = True
args.use_skip_connection = True
args.use_node_isolation = True
args.use_concat_bilinear = True
args.use_rw_loss = True
args.embed_d = 256

# Load data
dataset = PRELUDEDataset("data/processed")
generator = DataGenerator("data/processed")

# After loading dataset but before model initialization
print(f"Cell feature dimension: {dataset.cell_features_raw.shape[1]}")
print(f"VAE expected dimension: {args.vae_dims.split(',')[0]}")
assert dataset.cell_features_raw.shape[1] == int(args.vae_dims.split(',')[0]), \
       f"VAE dimension mismatch! {dataset.cell_features_raw.shape[1]} vs {int(args.vae_dims.split(',')[0])}"

# Print relation types for verification
print("\nRelation types in dataset:")
for rt, meta in dataset.info['link.dat'].items():
    print(f"Type {rt}: {meta[3]} edges ({meta[0]}→{meta[1]})")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
print("\nInitializing HetAgg model...")
model = HetAgg(args, dataset, device).to(device)
model.setup_link_prediction(drug_type_name="drug", cell_type_name="cell")

# --- Filter for valid cells with VAE features ---
valid_cell_ids = dataset.valid_cell_ids
valid_cell_names = [dataset.id2node[gid] for gid in valid_cell_ids]
print(f"\nNumber of valid cells: {len(valid_cell_names)}")
print(f"Sample valid cell names: {valid_cell_names[:5]}")

# Get filtered link prediction samples - USE RELATION TYPE 0 FOR CELL-DRUG
all_pos = generator.get_positive_pairs(0)  # Correct relation type for cell-drug
generator.build_edge_set()
all_neg = generator.sample_negative_pairs(0, num_samples=100)  # Same type

# Print sample pairs for verification
print(f"\nFirst 5 cell-drug pairs (global IDs): {all_pos[:5]}")
print(f"First 5 cell names in pairs: {[dataset.id2node[p[0]] for p in all_pos[:5]]}")
print(f"First 5 drug names in pairs: {[dataset.id2node[p[1]] for p in all_pos[:5]]}")

# Convert global IDs to cell names for filtering
def global_id_to_name(global_id):
    return dataset.id2node.get(global_id, "UNKNOWN")

filtered_pos = [
    pair for pair in all_pos 
    if global_id_to_name(pair[0]) in valid_cell_names
]

filtered_neg = [
    pair for pair in all_neg 
    if global_id_to_name(pair[0]) in valid_cell_names
]

# Add after generating filtered_pos/filtered_neg
print(f"\nTotal positive pairs: {len(all_pos)}")
print(f"Valid positive pairs: {len(filtered_pos)}")
print(f"Total negative pairs: {len(all_neg)}")
print(f"Valid negative pairs: {len(filtered_neg)}")

if not filtered_pos or not filtered_neg:
    print("\nWarning: Using first positive/negative pairs without filtering")
    filtered_pos = all_pos[:2]
    filtered_neg = all_neg[:2]
    print(f"Using filtered_pos: {filtered_pos}")
    print(f"Using filtered_neg: {filtered_neg}")

pos_pair = filtered_pos[0]
neg_pair = filtered_neg[0]

# Format input tensors
cell_ids = torch.tensor([pos_pair[0], neg_pair[0]], dtype=torch.long).to(device)
drug_ids = torch.tensor([pos_pair[1], neg_pair[1]], dtype=torch.long).to(device)
labels = torch.tensor([1, 0], dtype=torch.float).to(device)

# --- Forward pass + loss ---
print("\n🔁 Running link prediction forward pass...")
scores = model.link_prediction_forward(drug_ids, cell_ids)
print(f"Scores: {scores.detach().cpu().numpy()}")

print("\n🎯 Running link prediction loss...")
loss = model.link_prediction_loss(drug_ids, cell_ids, labels, isolation_ratio=0.2)
print(f"Link Prediction Loss: {loss.item():.4f}")

# --- Optional: RW loss ---
if args.use_rw_loss:
    print("\n🔀 Running self-supervised RW loss test...")
    rw_triples = generator.generate_rw_triples(walk_length=10, window_size=2, num_walks=2)
    if rw_triples:
        rw_loss = model.self_supervised_rw_loss([(t[0], t[1], t[1]) for t in rw_triples[:10]])  # dummy neg=pos
        print(f"RW Loss: {rw_loss.item():.4f}")
    else:
        print("No RW triples generated.")