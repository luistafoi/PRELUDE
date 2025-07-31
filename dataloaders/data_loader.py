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
        self.node_id_map = {}
        self.node_type_map = {}
        self.links = {
            'data': defaultdict(list),  # main graph
            'test': defaultdict(list)   # optional test links
        }

        self._load_nodes()
        self._load_links()
        self._load_test_links()

        # --- NEW: Load VAE Cell Features ---
        self.cell_features_raw = None
        self.cell_local_id_to_feature_idx = None
        self.cell_list_ordered = None
        self._load_cell_features()

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
            'count': dict(type_counts),       # dict: {node_type: count}
            'type_map': dict(type_map)        # dict: {global_id: (node_type, local_id)}
        }
        
        self.node_name2type = {
            "cell": 0,
            "drug": 1,
            "gene": 2
        }

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

    def _load_links(self):
        path = os.path.join(self.data_dir, 'link.dat')
        with open(path, 'r') as f:
            for line in f:
                src, tgt, ltype, weight = line.strip().split('\t')
                self.links['data'][int(ltype)].append((int(src), int(tgt), float(weight)))

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
        # Replace path with non-batch-corrected data
        EXPRESSION_FILE = '/data/luis/PRELUDE/data/embeddings/OmicsExpressionProteinCodingGenesTPMLogp1.csv'

        # Step 1: Extract cell names from node.dat (these are depmap_id now)
        node_cells = {
            name.upper(): nid for nid, name in self.id2node.items()
            if self.node_types[nid] == 0  # 0 = cell
        }

        # Step 2: Load expression data
        df_expr = pd.read_csv(EXPRESSION_FILE)
        expr_ids = df_expr.iloc[:, 0].astype(str).str.upper().tolist()
        expr_id_to_idx = {dep_id: i for i, dep_id in enumerate(expr_ids)}

        # Step 3: Find common cell lines between node.dat and expression data
        common_cells = sorted(set(node_cells.keys()).intersection(expr_id_to_idx.keys()))
        print(f"  > Matched {len(common_cells)} cell lines with expression data.")

        # Step 4: Extract matching rows and build tensor
        idxs = [expr_id_to_idx[cid] for cid in common_cells]
        df_expr_filtered = df_expr.iloc[idxs, 1:].astype(np.float32)
        expression_array = df_expr_filtered.to_numpy()

        # Step 5: Store tensor and nodeID -> featureIndex map
        self.cell_features_raw = torch.tensor(expression_array, dtype=torch.float32)
        self.cell_local_id_to_feature_idx = {
            self.node2id[cid]: i for i, cid in enumerate(common_cells)
        }
        self.cell_list_ordered = common_cells
        self.valid_cell_ids = set(self.cell_local_id_to_feature_idx.keys())

        print(f"  > Feature shape: {self.cell_features_raw.shape}")
        print("Cell feature loading complete.\n")

    def summary(self):
        print(f"Nodes: {len(self.node2id)}")
        for ntype_id, (label, count) in self.info['node.dat'].items():
            print(f"  • Type {ntype_id} ({label}): {count}")
        print(f"\n Links:")
        for ltype_id, meta in self.info['link.dat'].items():
            print(f"  • Type {ltype_id}: {meta[2]} — {meta[3]} edges")
        if self.links['test']:
            print(f"\n Test Links:")
            for ltype_id, edges in self.links['test'].items():
                print(f"  • Type {ltype_id}: {len(edges)} edges")