# scripts/verify_splits.py

import pandas as pd
import json
import random
import os

# --- Configuration ---
PROCESSED_DIR = "data/processed"
RAW_DATA_FILE = "/data/luis/HetGNN Data Processing/Repurposing_Public_23Q2_LMFI_NORMALIZED_with_DrugNames.csv"

NODE_MAP_FILE = os.path.join(PROCESSED_DIR, "node_mappings.json")
VALID_POS_FILE = os.path.join(PROCESSED_DIR, "valid_pos.dat")
VALID_NEG_FILE = os.path.join(PROCESSED_DIR, "valid_neg.dat")

NUM_SAMPLES_TO_CHECK = 10

# --- Main Verification Script ---
if __name__ == "__main__":
    print("--- Verifying the GMM-based data splits ---")

    try:
        # --- Step 1: Load and Prepare Data ---
        print(" > Loading and preparing data...")
        df_raw = pd.read_csv(RAW_DATA_FILE)
        
        # Apply the same cleaning as the build script
        df_raw['depmap_id'] = df_raw['row_id'].str.split('::').str[0].str.strip()
        df_raw['name_upper'] = df_raw['name'].astype(str).str.strip().str.upper()

        with open(NODE_MAP_FILE, 'r') as f:
            node_to_id = json.load(f)
        df_map = pd.DataFrame(node_to_id.items(), columns=['name', 'id'])
        df_map['name_upper'] = df_map['name'].astype(str).str.upper()
        
        df_pos = pd.read_csv(VALID_POS_FILE, sep='\t', header=None, names=['src_id', 'tgt_id', 'type', 'label'])
        df_neg = pd.read_csv(VALID_NEG_FILE, sep='\t', header=None, names=['src_id', 'tgt_id', 'type', 'label'])

        # --- Step 2: Verify Positive Links ---
        print(f"\n--- Checking {NUM_SAMPLES_TO_CHECK} random samples from the POSITIVE set ---")
        sample_pos = df_pos.sample(min(NUM_SAMPLES_TO_CHECK, len(df_pos)))
        for _, row in sample_pos.iterrows():
            cell_id, drug_id = row['src_id'], row['tgt_id']
            cell_name = df_map.loc[df_map['id'] == cell_id, 'name'].iloc[0]
            drug_name_upper = df_map.loc[df_map['id'] == drug_id, 'name_upper'].iloc[0]
            
            original_interaction = df_raw[
                (df_raw['depmap_id'] == cell_name) & (df_raw['name_upper'] == drug_name_upper)
            ]
            
            if not original_interaction.empty:
                original_score = original_interaction['LMFI.normalized'].iloc[0]
                print(f"  - Link: Cell '{cell_name}' <-> Drug '{df_map.loc[df_map['id'] == drug_id, 'name'].iloc[0]}'")
                print(f"    Original LMFI Score: {original_score:.4f} (Expected: Low for positive)")
            else:
                print(f"  - Could not find original interaction for Cell ID {cell_id} / Drug ID {drug_id}")

        # --- Step 3: Verify Negative Links ---
        print(f"\n--- Checking {NUM_SAMPLES_TO_CHECK} random samples from the NEGATIVE set ---")
        sample_neg = df_neg.sample(min(NUM_SAMPLES_TO_CHECK, len(df_neg)))
        for _, row in sample_neg.iterrows():
            cell_id, drug_id = row['src_id'], row['tgt_id']
            
            cell_name = df_map.loc[df_map['id'] == cell_id, 'name'].iloc[0]
            drug_name_upper = df_map.loc[df_map['id'] == drug_id, 'name_upper'].iloc[0]
            
            original_interaction = df_raw[
                (df_raw['depmap_id'] == cell_name) & (df_raw['name_upper'] == drug_name_upper)
            ]
            
            if not original_interaction.empty:
                original_score = original_interaction['LMFI.normalized'].iloc[0]
                print(f"  - Link: Cell '{cell_name}' <-> Drug '{df_map.loc[df_map['id'] == drug_id, 'name'].iloc[0]}'")
                print(f"    Original LMFI Score: {original_score:.4f} (Expected: High for negative)")
            else:
                print(f"  - Could not find original interaction for Cell ID {cell_id} / Drug ID {drug_id}")
            
    except FileNotFoundError as e:
        print(f"\nError: A required file was not found: {e.filename}")

    print("\nVerification complete.")
