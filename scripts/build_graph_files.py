# scripts/build_graph_files.py

"""
Graph Construction Script for the PRELUDE Project.

This script reads a cleaned and unified file of interactions and converts
it into the .dat graph format required by the model. It creates a `node.dat`
file with unique IDs for all entities (cells, drugs, genes) and a `link.dat`
file that represents the connections between them.

Example usage:
    python scripts/build_graph_files.py \
        --interactions-file data/processed/cleaned_interactions.csv \
        --output-dir data/processed
"""

import pandas as pd
import argparse
import os

def create_node_mappings(df: pd.DataFrame) -> tuple:
    """Creates unique integer IDs for all nodes."""
    print("Creating node mappings...")
    cells = pd.unique(df['cell_line_name'])
    drugs = pd.unique(df[df['interaction_type'] == 'cell_drug']['target_name'])
    genes = pd.unique(df[df['interaction_type'] == 'cell_gene']['target_name'])
    
    node_to_id = {}
    node_to_type = {}
    
    # Assign IDs and types: cell=0, drug=1, gene=2
    for node in cells:
        if node not in node_to_id:
            node_to_id[node] = len(node_to_id)
            node_to_type[node] = 0
    for node in drugs:
        if node not in node_to_id:
            node_to_id[node] = len(node_to_id)
            node_to_type[node] = 1
    for node in genes:
        if node not in node_to_id:
            node_to_id[node] = len(node_to_id)
            node_to_type[node] = 2
            
    print(f"  > Mapped {len(node_to_id)} unique nodes.")
    return node_to_id, node_to_type

def write_node_dat(node_to_id: dict, node_to_type: dict, output_path: str):
    """Writes the node.dat file."""
    print(f"Writing node data to {output_path}...")
    with open(output_path, 'w') as f:
        # Sort by ID for consistency
        sorted_nodes = sorted(node_to_id.items(), key=lambda item: item[1])
        for name, uid in sorted_nodes:
            ntype = node_to_type[name]
            f.write(f"{uid}\t{name}\t{ntype}\n")

def write_link_dat(df: pd.DataFrame, node_to_id: dict, output_path: str):
    """Writes the link.dat file."""
    print(f"Writing link data to {output_path}...")
    # Define link types based on the interaction pairs
    link_type_map = {
        ('cell', 'drug'): 0,
        ('cell', 'gene'): 1
        # Add other types like ('gene', 'drug'): 2, etc. if they exist
    }
    
    with open(output_path, 'w') as f:
        for _, row in df.iterrows():
            source_name = row['cell_line_name']
            target_name = row['target_name']
            
            source_id = node_to_id.get(source_name)
            target_id = node_to_id.get(target_name)
            
            if source_id is None or target_id is None:
                continue

            # Determine link type (e.g., cell_drug -> (cell, drug))
            interaction = tuple(row['interaction_type'].split('_'))
            link_type = link_type_map.get(interaction)
            
            if link_type is not None:
                f.write(f"{source_id}\t{target_id}\t{link_type}\t{row['weight']}\n")

def main(args):
    """Main execution function."""
    df_interactions = pd.read_csv(args.interactions_file)
    
    node_to_id, node_to_type = create_node_mappings(df_interactions)
    
    node_dat_path = os.path.join(args.output_dir, 'node.dat')
    link_dat_path = os.path.join(args.output_dir, 'link.dat')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    write_node_dat(node_to_id, node_to_type, node_dat_path)
    write_link_dat(df_interactions, node_to_id, link_dat_path)
    
    print("\nGraph construction complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build graph .dat files from cleaned interactions.")
    parser.add_argument('--interactions-file', required=True, help='Path to the cleaned interactions CSV file.')
    parser.add_argument('--output-dir', default='data/processed', help='Directory to save the output .dat files.')
    
    args = parser.parse_args()
    main(args)