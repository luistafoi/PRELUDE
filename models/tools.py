# models/tools.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
from collections import defaultdict

from dataloaders.data_loader import PRELUDEDataset
from dataloaders.feature_loader import FeatureLoader
from models.layers import RnnGnnLayer
from scripts.cell_vae import CellLineVAE

class HetAgg(nn.Module):
    def __init__(self, args, dataset: PRELUDEDataset, feature_loader: FeatureLoader, device):
        super(HetAgg, self).__init__()
        self.args = args
        self.device = device
        self.embed_d = args.embed_d
        self.dataset = dataset
        self.feature_loader = feature_loader
        
        self.node_types = sorted(self.dataset.nodes['count'].keys())
        cell_type_id = self.dataset.node_name2type['cell']
        drug_type_id = self.dataset.node_name2type['drug']
        gene_type_id = self.dataset.node_name2type['gene']
        
        # --- 1. Feature Projection Layers (Instead of nn.Embedding) ---
        self.feat_proj = nn.ModuleDict()
        
        # For drugs and genes, we use a Linear layer to project pre-trained features
        drug_feat_dim = self.feature_loader.drug_features.shape[1]
        self.feat_proj[str(drug_type_id)] = nn.Linear(drug_feat_dim, self.embed_d).to(device)
        
        gene_feat_dim = self.feature_loader.gene_features.shape[1]
        self.feat_proj[str(gene_type_id)] = nn.Linear(gene_feat_dim, self.embed_d).to(device)

        # For cells, we use the VAE encoder
        if args.use_vae_encoder:
            print("INFO: Initializing VAE encoder for cell features.")
            vae_dims = list(map(int, args.vae_dims.split(',')))
            full_vae = CellLineVAE(vae_dims).to(device)
            if os.path.exists(args.vae_checkpoint):
                full_vae.load_state_dict(torch.load(args.vae_checkpoint, weights_only=True))
                print(f"Loaded VAE weights from: {args.vae_checkpoint}")
            else:
                print(f"VAE weights not found. Using random init.")
            self.cell_encoder = full_vae.encoder
            self.cell_encoder.eval()
            for param in self.cell_encoder.parameters():
                param.requires_grad = False
        else: # Fallback to a simple embedding if not using VAE
            num_cells = self.dataset.nodes['count'][cell_type_id]
            self.feat_proj[str(cell_type_id)] = nn.Embedding(num_cells, self.embed_d).to(device)

        # --- 2. GNN Layers (Restored from old model) ---
        self.gnn_layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.gnn_layers.append(
                RnnGnnLayer(self.embed_d, self.embed_d, self.node_types)
            )

        # --- 3. Link Prediction Head ---
        self.lp_bilinear = None # To be initialized by setup_link_prediction
        self.drug_type_name = None
        self.cell_type_name = None

    def setup_link_prediction(self, drug_type_name, cell_type_name):
        self.drug_type_name = drug_type_name
        self.cell_type_name = cell_type_name
        factor = 2 if self.args.use_skip_connection else 1
        self.lp_bilinear = nn.Bilinear(self.embed_d * factor, self.embed_d * factor, 1).to(self.device)

    def conteng_agg(self, local_id_batch, node_type):
        """Gets the initial node features (before message passing)."""
        if not local_id_batch:
            return torch.empty(0, self.embed_d, device=self.device)
            
        cell_type_id = self.dataset.node_name2type['cell']

        if self.args.use_vae_encoder and node_type == cell_type_id:
            feature_indices = [self.dataset.cell_local_id_to_feature_idx[i] for i in local_id_batch]
            raw_features = self.dataset.cell_features_raw[feature_indices].to(self.device)
            return self.cell_encoder(raw_features)
        
        elif node_type == self.dataset.node_name2type['drug']:
            raw_features = self.feature_loader.drug_features[local_id_batch]
            return self.feat_proj[str(node_type)](raw_features)
            
        elif node_type == self.dataset.node_name2type['gene']:
            raw_features = self.feature_loader.gene_features[local_id_batch]
            return self.feat_proj[str(node_type)](raw_features)

        else: # Fallback for cells if not using VAE, or other types
            local_indices_tensor = torch.LongTensor(local_id_batch).to(self.device)
            return self.feat_proj[str(node_type)](local_indices_tensor)

    def node_het_agg(self, id_batch_local, node_type, data_generator):
        """
        Performs full GNN message passing using the pre-generated neighbor list.
        This is the corrected implementation, aligned with the original HetGNN logic.
        """
        if data_generator.train_neighbors is None:
            raise RuntimeError("Training neighbors not loaded in DataGenerator. Call load_train_neighbors() first.")

        # Start with initial features
        current_embeds = self.conteng_agg(id_batch_local, node_type)
        node_type_name = self.dataset.node_name2type[node_type]
        
        for layer in self.gnn_layers:
            neigh_embeds_by_type = defaultdict(list)
            
            # 1. Gather pre-sampled neighbors for each node in the batch
            for i, local_id in enumerate(id_batch_local):
                center_node_str = f"{node_type_name}{local_id}"
                # Look up the neighbor strings (e.g., ["gene123", "drug45"])
                neighbor_strings = data_generator.train_neighbors.get(center_node_str, [])
                
                # Parse and group neighbors by their type
                parsed_neighbors = defaultdict(list)
                for neigh_str in neighbor_strings:
                    for nt_id, nt_name in self.dataset.node_name2type.items():
                        if neigh_str.startswith(nt_name):
                            try:
                                neigh_local_id = int(neigh_str[len(nt_name):])
                                parsed_neighbors[nt_id].append(neigh_local_id)
                                break
                            except ValueError:
                                continue # Skip malformed strings
                
                # 2. Pad or sample to a fixed size for each neighbor type
                for nt in self.node_types:
                    # NOTE: This should be configured in args, hardcoding for now
                    num_samples = 10 
                    
                    neigh_list = parsed_neighbors.get(nt, [])
                    if len(neigh_list) > num_samples:
                        neigh_list = random.sample(neigh_list, num_samples)
                    else:
                        # Pad with random nodes of the same type
                        num_to_pad = num_samples - len(neigh_list)
                        num_nodes_of_type = self.dataset.nodes['count'][nt]
                        padding = [random.randint(0, num_nodes_of_type - 1) for _ in range(num_to_pad)]
                        neigh_list.extend(padding)
                    
                    neigh_embeds_by_type[nt].append(neigh_list)
            
            # 3. Fetch features and pass to the GNN layer
            aggregated_neighbors = {}
            for nt, batched_neigh_ids in neigh_embeds_by_type.items():
                # Flatten to fetch all features at once
                flat_ids = [item for sublist in batched_neigh_ids for item in sublist]
                if flat_ids:
                    neigh_feats = self.conteng_agg(flat_ids, nt)
                    # Reshape for RNN: (batch_size, num_samples, dim)
                    aggregated_neighbors[nt] = neigh_feats.view(len(id_batch_local), -1, self.embed_d)
                else:
                    aggregated_neighbors[nt] = torch.zeros(len(id_batch_local), num_samples, self.embed_d, device=self.device)

            current_embeds = layer(current_embeds, aggregated_neighbors)
        
        return current_embeds

    def get_combined_embedding(self, id_batch_local, node_type, data_generator):
        """Concatenates initial features with final GNN embeddings for skip connections."""
        initial_embeds = self.conteng_agg(id_batch_local, node_type)
        final_embeds = self.node_het_agg(id_batch_local, node_type, data_generator)
        
        if self.args.use_skip_connection:
            return torch.cat([initial_embeds, final_embeds], dim=1)
        else:
            return final_embeds

    def link_prediction_loss(self, drug_indices_local, cell_indices_local, labels, data_generator, isolation_ratio=0.0):
        drug_type_id = self.dataset.node_name2type[self.drug_type_name]
        cell_type_id = self.dataset.node_name2type[self.cell_type_name]

        # Get drug embeddings (always uses GNN)
        drug_embeds = self.get_combined_embedding(drug_indices_local, drug_type_id, data_generator)

        # Handle cell embeddings with potential isolation
        if self.args.use_node_isolation and isolation_ratio > 0:
            initial_cell_embeds = self.conteng_agg(cell_indices_local, cell_type_id)
            final_cell_embeds = torch.zeros_like(initial_cell_embeds)
            
            should_isolate = torch.rand(len(cell_indices_local), device=self.device) < isolation_ratio
            graph_connected_mask = ~should_isolate

            if graph_connected_mask.any():
                connected_ids = [cell_indices_local[i] for i, connected in enumerate(graph_connected_mask) if connected]
                final_cell_embeds[graph_connected_mask] = self.node_het_agg(connected_ids, cell_type_id, data_generator)
            
            if should_isolate.any():
                final_cell_embeds[should_isolate] = initial_cell_embeds[should_isolate]

            cell_embeds = torch.cat([initial_cell_embeds, final_cell_embeds], dim=1) if self.args.use_skip_connection else final_cell_embeds
        else:
            cell_embeds = self.get_combined_embedding(cell_indices_local, cell_type_id, data_generator)
            
        scores = self.lp_bilinear(drug_embeds, cell_embeds).squeeze(-1)
        return F.binary_cross_entropy_with_logits(scores, labels.float())

    def link_prediction_forward(self, drug_indices_local, cell_indices_local, data_generator):
        drug_type_id = self.dataset.node_name2type[self.drug_type_name]
        cell_type_id = self.dataset.node_name2type[self.cell_type_name]
        
        drug_embeds = self.get_combined_embedding(drug_indices_local, drug_type_id, data_generator)
        cell_embeds = self.get_combined_embedding(cell_indices_local, cell_type_id, data_generator)
        
        scores = self.lp_bilinear(drug_embeds, cell_embeds).squeeze(-1)
        return torch.sigmoid(scores)