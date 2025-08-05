# dataloaders/data_loader.py

import os
import json
import pandas as pd
import numpy as np
import torch
from collections import defaultdict

class PRELUDEDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.info = self._load_info()
        
        # This dictionary will now hold the graph structure and the pre-split links
        self.links = {
            'graph': defaultdict(list), # For GNN message passing (from train.dat)
            'train_pos': [],
            'train_neg': [],
            'valid_pos': [],
            'valid_neg': [],
            'test_pos': [],
            'test_neg': []
        }

        self._load_nodes()

        # Create the node mapping dictionaries first
        type_counts = defaultdict(int)
        type_map = {}
        local_type_counters = defaultdict(int)
        for nid in sorted(self.node_types.keys()):
            ntype = self.node_types[nid]
            local_id = local_type_counters[ntype]
            type_map[nid] = (ntype, local_id)
            local_type_counters[ntype] += 1
            type_counts[ntype] += 1
        self.nodes = {
            'count': dict(type_counts),
            'type_map': dict(type_map)
        }

        # Load all the different link files
        self._load_graph_links()
        self._load_lp_splits()

        # Load cell features
        self.cell_features_raw = None
        self.cell_global_id_to_feature_idx = None
        self._load_cell_features()

        self.node_name2type = {"cell": 0, "drug": 1, "gene": 2}
        self.node_type2name = {v: k for k, v in self.node_name2type.items()}
        
        self.local_to_global_map = {ntype: {} for ntype in self.nodes['count']}
        for global_id, (ntype, local_id) in self.nodes['type_map'].items():
            self.local_to_global_map[ntype][local_id] = global_id

    def _load_info(self):
        path = os.path.join(self.data_dir, 'info.dat')
        if not os.path.exists(path): return {}
        with open(path, 'r') as f:
            return json.load(f)

    def _load_nodes(self):
        path = os.path.join(self.data_dir, 'node.dat')
        self.node2id = {}
        self.id2node = {}
        self.node_types = {}
        with open(path, 'r') as f:
            for line in f:
                nid, name, ntype = line.strip().split('\t')
                nid = int(nid)
                ntype = int(ntype)
                self.node2id[name] = nid
                self.id2node[nid] = name
                self.node_types[nid] = ntype

    def _load_graph_links(self):
        """Loads the structural links (from train.dat) for GNN message passing."""
        path = os.path.join(self.data_dir, 'train.dat')
        if not os.path.exists(path): return
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 4: continue
                src, tgt, ltype, weight = parts
                self.links['graph'][int(ltype)].append((int(src), int(tgt), float(weight)))

    def _load_lp_splits(self):
        """Loads all pre-split positive and negative link files."""
        print("Loading pre-split link prediction data...")
        split_names = ['train_pos', 'train_neg', 'valid_pos', 'valid_neg', 'test_pos', 'test_neg']
        for split in split_names:
            path = os.path.join(self.data_dir, f"{split}.dat")
            if os.path.exists(path):
                with open(path, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            self.links[split].append((int(parts[0]), int(parts[1])))
            print(f"  > Loaded {len(self.links[split])} links for {split}")

    def _load_cell_features(self):
        print("\nLoading VAE-compatible raw cell expression features...")
        EXPRESSION_FILE = 'data/embeddings/OmicsExpressionProteinCodingGenesTPMLogp1.csv'

        node_cells = {
            name.upper(): nid for nid, name in self.id2node.items()
            if self.node_types[nid] == 0
        }

        df_expr = pd.read_csv(EXPRESSION_FILE)
        expr_ids = df_expr.iloc[:, 0].astype(str).str.upper().tolist()
        expr_id_to_idx = {dep_id: i for i, dep_id in enumerate(expr_ids)}

        common_cells = sorted(set(node_cells.keys()).intersection(expr_id_to_idx.keys()))
        print(f"  > Matched {len(common_cells)} cell lines with expression data.")

        idxs = [expr_id_to_idx[cid] for cid in common_cells]
        expression_array = df_expr.iloc[idxs, 1:].astype(np.float32).to_numpy()

        self.cell_features_raw = torch.tensor(expression_array, dtype=torch.float32)
        
        self.cell_global_id_to_feature_idx = {
            self.node2id[cid.upper()]: i for i, cid in enumerate(common_cells)
        }
        
        self.valid_cell_ids = set(self.cell_global_id_to_feature_idx.keys())
        self.valid_cell_local_ids = [
            self.nodes['type_map'][gid][1] for gid in self.valid_cell_ids
        ]

        print(f"  > Feature shape: {self.cell_features_raw.shape}")
        print("Cell feature loading complete.\n")

    def summary(self):
        """Prints a detailed summary of the loaded dataset."""
        print("--- PRELUDE Dataset Summary ---")
        print(f"\nNodes:")
        for ntype_id, count in sorted(self.nodes['count'].items()):
            ntype_name = self.node_type2name.get(ntype_id, f"Type {ntype_id}")
            print(f"  - {ntype_name.capitalize()} (Type {ntype_id}): {count} nodes")

        print(f"\nStructural Links (for GNN message passing from train.dat):")
        total_graph_links = 0
        for ltype, edges in sorted(self.links['graph'].items()):
            # Get type names from info.dat if available
            type_info = self.info.get('link.dat', {}).get(str(ltype), ["", "", f"Type {ltype}"])
            print(f"  - {type_info[2]}: {len(edges)} edges")
            total_graph_links += len(edges)
        print(f"  > Total structural links for GNN: {total_graph_links}")

        print(f"\nLink Prediction Pairs (GMM Labeled):")
        print(f"  - Training Set:   {len(self.links['train_pos'])} positive, {len(self.links['train_neg'])} negative")
        print(f"  - Validation Set: {len(self.links['valid_pos'])} positive, {len(self.links['valid_neg'])} negative")
        print(f"  - Test Set:       {len(self.links['test_pos'])} positive, {len(self.links['test_neg'])} negative")
        print("---------------------------------")