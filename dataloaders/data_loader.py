# dataloaders/data_loader.py

import os
import json
import pandas as pd
import numpy as np
import torch
from collections import defaultdict
import random

class PRELUDEDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.info = self._load_info()
        self.links = {
            'train': defaultdict(list), # Changed from 'data'
            'valid': defaultdict(list), # Added validation set
            'test': defaultdict(list)
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

        self._load_links()
        self._load_test_links()

        # Load cell features (now that self.nodes exists)
        self.cell_features_raw = None
        self.cell_global_id_to_feature_idx = None # Renamed for clarity
        self._load_cell_features()

        self.node_name2type = {"cell": 0, "drug": 1, "gene": 2}
        self.node_type2name = {v: k for k, v in self.node_name2type.items()}
        
        # Create a helper map for converting local IDs to global IDs
        self.local_to_global_map = {ntype: {} for ntype in self.nodes['count']}
        for global_id, (ntype, local_id) in self.nodes['type_map'].items():
            self.local_to_global_map[ntype][local_id] = global_id

    def _load_info(self):
        path = os.path.join(self.data_dir, 'info.dat')
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

    def _load_links(self, val_split_ratio=0.1):
        """
        Loads links and splits them into training and validation sets.
        """
        path = os.path.join(self.data_dir, 'link.dat')
        
        # First, load all links grouped by type
        all_links = defaultdict(list)
        with open(path, 'r') as f:
            for line in f:
                src, tgt, ltype, weight = line.strip().split('\t')
                all_links[int(ltype)].append((int(src), int(tgt), float(weight)))
        
        # Now, split each type into train and validation
        print("\nSplitting links into training and validation sets...")
        for ltype, edges in all_links.items():
            random.shuffle(edges)
            split_idx = int(len(edges) * (1 - val_split_ratio))
            self.links['train'][ltype] = edges[:split_idx]
            self.links['valid'][ltype] = edges[split_idx:]
            print(f"  - Link Type {ltype}: {len(self.links['train'][ltype])} train, {len(self.links['valid'][ltype])} valid")


    def _load_test_links(self):
        path = os.path.join(self.data_dir, 'link.dat.test')
        if not os.path.exists(path):
            return
        with open(path, 'r') as f:
            for line in f:
                src, tgt, ltype, weight = line.strip().split('\t')
                self.links['test'][int(ltype)].append((int(src), int(tgt), float(weight)))

    def _load_cell_features(self):
        print("\nLoading VAE-compatible raw cell expression features...")
        EXPRESSION_FILE = 'data/embeddings/OmicsExpressionProteinCodingGenesTPMLogp1.csv'

        node_cells = {
            name.upper(): nid for nid, name in self.id2node.items()
            if self.node_types[nid] == 0  # 0 = cell
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
        print(f"🧬 Nodes: {len(self.node2id)}")
        for ntype_id_str, meta in self.info['node.dat'].items():
            ntype_id = int(ntype_id_str)
            label = self.node_type2name.get(ntype_id, meta[0])
            count = self.nodes['count'].get(ntype_id, meta[1])
            print(f"  • Type {ntype_id} ({label}): {count}")
        print(f"\n🔗 Links:")
        for ltype_id_str, meta in self.info['link.dat'].items():
            ltype_id = int(ltype_id_str)
            num_edges = len(self.links['data'].get(ltype_id, []))
            print(f"  • Type {ltype_id}: {meta[2]} — {num_edges} edges")
        if self.links['test']:
            print(f"\n🧪 Test Links:")
            for ltype_id, edges in self.links['test'].items():
                print(f"  • Type {ltype_id}: {len(edges)} edges")