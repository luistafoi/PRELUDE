# dataloaders/feature_loader.py

import os
import torch
import pandas as pd
import numpy as np
import pickle

class FeatureLoader:
    """
    Handles the loading of pre-computed node features for drugs and genes.
    It creates feature tensors where the row index corresponds to the local ID
    of a node, ensuring perfect alignment for the GNN.
    """
    def __init__(self, dataset, device):
        print("Initializing FeatureLoader...")
        self.dataset = dataset
        self.device = device
        
        self.drug_features = self._load_drug_features().to(self.device)
        self.gene_features = self._load_gene_features().to(self.device)
        print("FeatureLoader initialization complete.")

    def _load_drug_features(self):
        """
        Loads drug embeddings and aligns them into a tensor by local ID.
        """
        drug_type_id = self.dataset.node_name2type["drug"]
        num_drugs = self.dataset.nodes['count'][drug_type_id]
        
        # 1. Load the raw drug embeddings CSV
        path = "data/embeddings/drugs_with_embeddings.csv"
        df = pd.read_csv(path)
        
        # 2. Create a mapping from DRUG NAME -> feature vector
        drug_name_to_feat = {
            row['Drug'].upper(): np.array(row[1:].values.astype(np.float32))
            for _, row in df.iterrows()
        }
        
        # Assume a consistent feature dimension from the first entry
        feature_dim = len(next(iter(drug_name_to_feat.values())))
        
        # 3. Create the final tensor, ordered by local ID
        aligned_drug_feats = torch.zeros(num_drugs, feature_dim, dtype=torch.float32)
        
        # 4. Populate the tensor
        for global_id, (ntype, local_id) in self.dataset.nodes['type_map'].items():
            if ntype == drug_type_id:
                node_name = self.dataset.id2node[global_id].upper()
                feature_vec = drug_name_to_feat.get(node_name)
                if feature_vec is not None:
                    aligned_drug_feats[local_id] = torch.from_numpy(feature_vec)
                else:
                    # Optional: Log if a drug in the graph has no feature
                    pass 
                    
        print(f"  > Loaded aligned drug features tensor: {aligned_drug_feats.shape}")
        return aligned_drug_feats

    def _load_gene_features(self):
        """
        Loads gene embeddings and aligns them into a tensor by local ID.
        """
        gene_type_id = self.dataset.node_name2type["gene"]
        num_genes = self.dataset.nodes['count'][gene_type_id]

        # 1. Load the raw gene embeddings pickle
        path = "data/embeddings/gene_embeddings_esm_by_symbol.pkl"
        with open(path, "rb") as f:
            gene_name_to_feat = pickle.load(f)

        # Assume a consistent feature dimension
        feature_dim = len(next(iter(gene_name_to_feat.values())))

        # 2. Create the final tensor
        aligned_gene_feats = torch.zeros(num_genes, feature_dim, dtype=torch.float32)

        # 3. Populate the tensor
        for global_id, (ntype, local_id) in self.dataset.nodes['type_map'].items():
            if ntype == gene_type_id:
                node_name = self.dataset.id2node[global_id]
                feature_vec = gene_name_to_feat.get(node_name)
                if feature_vec is not None:
                    aligned_gene_feats[local_id] = torch.from_numpy(np.array(feature_vec))
        
        print(f"  > Loaded aligned gene features tensor: {aligned_gene_feats.shape}")
        return aligned_gene_feats