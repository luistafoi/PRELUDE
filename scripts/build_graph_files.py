# scripts/build_graph_files.py

"""
Constructs the final graph files (node.dat, link.dat, info.dat) and a node
mapping file, following the proven single-pass logic from the source notebook.
"""
import pandas as pd
import json
from collections import defaultdict
from tqdm import tqdm
import os
import argparse

def main(args):
    # --- Configuration: Define input files and node type mappings ---
    link_paths = {
        "cell-drug": args.cell_drug_file,
        "gene-cell": os.path.join(args.raw_dir, "link_gene_cell.txt"),
        "gene-drug": os.path.join(args.raw_dir, "link_gene_drug.txt"),
        "gene-gene": os.path.join(args.raw_dir, "link_gene_gene.txt")
    }

    # Maps file key to (source_type_id, target_type_id, relation_name)
    type_map = {
        "cell-drug": (0, 1, "cell-drug_interaction"),
        "gene-cell": (0, 2, "cell-gene_expression"),
        "gene-drug": (2, 1, "gene-targeted_by_drug"),
        "gene-gene": (2, 2, "gene-interacts_gene")
    }
    
    # --- Initialization (as per the notebook) ---
    node2id = {}
    node_types = {}  # maps node name to its type id
    next_id = 0

    edges = []
    link_info = {}
    link_type_counter = 0

    # --- Process all link files in a single pass ---
    print("--- Building graph files from raw sources ---")
    for name, path in link_paths.items():
        if not os.path.exists(path):
            print(f"  > Warning: File not found for {name} at {path}. Skipping.")
            continue

        print(f"  > Processing {name}...")
        df = pd.read_csv(path, sep="\t", header=None, names=["src", "tgt", "type", "weight"])
        src_type, tgt_type, relation_name = type_map[name]

        # Clean strings
        df["src"] = df["src"].astype(str).str.strip()
        df["tgt"] = df["tgt"].astype(str).str.strip()

        # Uppercase drug names only
        if src_type == 1:
            df["src"] = df["src"].str.upper()
        if tgt_type == 1:
            df["tgt"] = df["tgt"].str.upper()

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  - {name}"):
            src, tgt = row["src"], row["tgt"]

            # Assign IDs and types if nodes are new
            for node, ntype in [(src, src_type), (tgt, tgt_type)]:
                if node not in node2id:
                    node2id[node] = next_id
                    node_types[node] = ntype
                    next_id += 1

            # Translate names to IDs for the edge
            src_id = node2id[src]
            tgt_id = node2id[tgt]
            weight = row["weight"]

            edges.append(f"{src_id}\t{tgt_id}\t{link_type_counter}\t{weight}")
        
        # Store metadata for info.dat
        type_names = {0: "cell", 1: "drug", 2: "gene"}
        link_info[str(link_type_counter)] = [ 
            type_names[src_type],
            type_names[tgt_type],
            relation_name,
            len(df)
        ]
        link_type_counter += 1

    # --- Create output directory if it doesn't exist ---
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Save node.dat ---
    node_dat_path = os.path.join(args.output_dir, "node.dat")
    with open(node_dat_path, "w") as f:
        sorted_nodes = sorted(node2id.items(), key=lambda item: item[1])
        for node, nid in sorted_nodes:
            f.write(f"{nid}\t{node}\t{node_types[node]}\n")

    # --- Save node_mappings.json ---
    map_path = os.path.join(args.output_dir, "node_mappings.json")
    with open(map_path, 'w') as f:
        json.dump(node2id, f, indent=4)

    # --- Save link.dat ---
    link_dat_path = os.path.join(args.output_dir, "link.dat")
    with open(link_dat_path, "w") as f:
        f.write("\n".join(edges))

    # --- Save info.dat ---
    type_counts = defaultdict(int)
    for ntype in node_types.values():
        type_counts[ntype] += 1
        
    info = {
        "dataset": "PRELUDE_CellDrugGene_Network",
        "node.dat": {
            "0": ["cell", type_counts[0]],
            "1": ["drug", type_counts[1]],
            "2": ["gene", type_counts[2]]
        },
        "link.dat": link_info
    }
    info_dat_path = os.path.join(args.output_dir, "info.dat")
    with open(info_dat_path, "w") as f:
        json.dump(info, f, indent=4)

    # --- Summary ---
    print("\n--- Summary ---")
    print(f"  > Wrote {len(node2id)} nodes to {node_dat_path}")
    print(f"  > Wrote mapping for {len(node2id)} nodes to {map_path}")
    print(f"  > Wrote {len(edges)} links to {link_dat_path}")
    print(f"  > Wrote metadata to {info_dat_path}")
    print("\nGraph construction complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build graph .dat files from raw link files, following the proven notebook logic.")
    parser.add_argument('--cell-drug-file', default='data/raw/link_cell_drug_labeled.txt',
                        help='Path to the GMM-labeled cell-drug link file.')
    parser.add_argument('--raw-dir', default='data/raw',
                        help='Directory containing other raw link files (gene-gene, etc.).')
    parser.add_argument('--output-dir', default='data/processed',
                        help='Directory to save the output .dat files.')
    
    args = parser.parse_args()
    main(args)
