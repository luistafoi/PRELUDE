# scripts/create_splits.py

import os
import random
import pandas as pd
import json

# --- Configuration ---
PROCESSED_DIR = "data/processed"
LINK_DAT_FILE = os.path.join(PROCESSED_DIR, "link.dat")

# Output files
TRAIN_POS_FILE = os.path.join(PROCESSED_DIR, "train_pos.dat")
TRAIN_NEG_FILE = os.path.join(PROCESSED_DIR, "train_neg.dat")
VALID_POS_FILE = os.path.join(PROCESSED_DIR, "valid_pos.dat")
VALID_NEG_FILE = os.path.join(PROCESSED_DIR, "valid_neg.dat")
TEST_POS_FILE = os.path.join(PROCESSED_DIR, "test_pos.dat")
TEST_NEG_FILE = os.path.join(PROCESSED_DIR, "test_neg.dat")
FINAL_TRAIN_GRAPH_FILE = os.path.join(PROCESSED_DIR, "train.dat")

# Ratios
VALID_RATIO = 0.1
TEST_RATIO = 0.1
CELL_DRUG_LINK_TYPE = 0

# --- Main Script ---
if __name__ == "__main__":
    print("--- Starting Final Data Splitting from ID-based files ---")

    print(f"Reading processed links from: {LINK_DAT_FILE}")
    df_links = pd.read_csv(
        LINK_DAT_FILE, 
        sep='\t', 
        header=None, 
        names=['src_id', 'tgt_id', 'type', 'label'],
        dtype={'src_id': int, 'tgt_id': int, 'type': int, 'label': float} 
    )

    df_dc_links = df_links[df_links['type'] == CELL_DRUG_LINK_TYPE].copy()
    other_links = df_links[df_links['type'] != CELL_DRUG_LINK_TYPE]

    positive_links = df_dc_links[df_dc_links['label'] == 1.0]
    negative_links = df_dc_links[df_dc_links['label'] == 0.0]
    
    print(f"  > Found {len(positive_links)} Positive and {len(negative_links)} Negative cell-drug links.")

    pos_links_str = []
    if not positive_links.empty:
        pos_links_str = positive_links.to_csv(header=False, index=False, sep='\t').strip().split('\n')

    neg_links_str = []
    if not negative_links.empty:
        neg_links_str = negative_links.to_csv(header=False, index=False, sep='\t').strip().split('\n')

    random.shuffle(pos_links_str)
    random.shuffle(neg_links_str)

    # --- Split Positive and Negative Links into Train/Valid/Test ---
    print("\n--- Splitting links into training, validation, and test sets ---")
    
    # Split positives
    pos_test_end = int(len(pos_links_str) * TEST_RATIO)
    pos_valid_end = pos_test_end + int(len(pos_links_str) * VALID_RATIO)
    test_pos = pos_links_str[:pos_test_end]
    valid_pos = pos_links_str[pos_test_end:pos_valid_end]
    train_pos = pos_links_str[pos_valid_end:]

    # --- START OF FIX: Corrected slicing logic for negatives ---
    neg_test_end = int(len(neg_links_str) * TEST_RATIO)
    neg_valid_end = neg_test_end + int(len(neg_links_str) * VALID_RATIO)
    test_neg = neg_links_str[:neg_test_end]
    valid_neg = neg_links_str[neg_test_end:neg_valid_end] # This was the line with the likely error
    train_neg = neg_links_str[neg_valid_end:]
    # --- END OF FIX ---

    # --- Write the Split Files ---
    with open(TRAIN_POS_FILE, 'w') as f: f.write('\n'.join(train_pos))
    with open(TRAIN_NEG_FILE, 'w') as f: f.write('\n'.join(train_neg))
    print(f"  > Wrote {len(train_pos)} pos / {len(train_neg)} neg links to training files.")

    with open(VALID_POS_FILE, 'w') as f: f.write('\n'.join(valid_pos))
    with open(VALID_NEG_FILE, 'w') as f: f.write('\n'.join(valid_neg))
    print(f"  > Wrote {len(valid_pos)} pos / {len(valid_neg)} neg links to validation files.")

    with open(TEST_POS_FILE, 'w') as f: f.write('\n'.join(test_pos))
    with open(TEST_NEG_FILE, 'w') as f: f.write('\n'.join(test_neg))
    print(f"  > Wrote {len(test_pos)} pos / {len(test_neg)} neg links to test files.")

    # --- Create the Final `train.dat` for Graph Construction ---
    print("\n--- Creating final train.dat for graph structure ---")
    
    other_links_str = []
    if not other_links.empty:
        other_links_str = other_links.to_csv(header=False, index=False, sep='\t').strip().split('\n')
    
    final_train_graph_content = train_pos + other_links_str
    
    with open(FINAL_TRAIN_GRAPH_FILE, 'w') as f:
        f.write('\n'.join(final_train_graph_content))
    print(f"  > Final train.dat created with {len(final_train_graph_content)} total structural links.")
    
    print("\n Data curation and splitting complete.")
