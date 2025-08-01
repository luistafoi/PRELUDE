# scripts/create_splits.py

import os
import random
from collections import defaultdict

# --- Configuration ---
# 1. This script now reads your main link file.
SOURCE_LINK_DAT = "data/processed/link.dat" 

# 2. These are the output files.
OUTPUT_DIR = "data/processed"
NEW_TRAIN_FILE = os.path.join(OUTPUT_DIR, "link.dat")
TEST_FILE = os.path.join(OUTPUT_DIR, "link.dat.test")

# 3. Set the test set split ratio for the drug-cell links.
TEST_SPLIT_RATIO = 0.1 # 10% of drug-cell links will be used for testing
DRUG_CELL_LINK_TYPE = '0' # The type ID for drug-cell links as a string

# --- Main Script ---
if __name__ == "__main__":
    print(f"Reading all links from: {SOURCE_LINK_DAT}")
    
    if not os.path.exists(SOURCE_LINK_DAT):
        print(f"Error: Source file not found at {SOURCE_LINK_DAT}. Aborting.")
        exit()

    # Read the file and separate drug-cell links from all other links
    drug_cell_links = []
    other_links = []
    with open(SOURCE_LINK_DAT, 'r') as f:
        for line in f:
            # Assumes format: src\ttgt\tltype\tweight
            parts = line.strip().split('\t')
            if len(parts) == 4 and parts[2] == DRUG_CELL_LINK_TYPE:
                drug_cell_links.append(line)
            else:
                other_links.append(line)
    
    print(f"  > Found {len(drug_cell_links)} drug-cell links to split.")
    print(f"  > Found {len(other_links)} other links to keep for training.")

    # Shuffle and split only the drug-cell links
    random.shuffle(drug_cell_links)
    split_idx = int(len(drug_cell_links) * (1 - TEST_SPLIT_RATIO))
    
    train_drug_cell_links = drug_cell_links[:split_idx]
    test_drug_cell_links = drug_cell_links[split_idx:]
    
    # Write the new training file (link.dat)
    # It contains the training portion of drug-cell links + ALL other links
    with open(NEW_TRAIN_FILE, 'w') as f:
        f.writelines(train_drug_cell_links)
        f.writelines(other_links)
    print(f"Wrote {len(train_drug_cell_links)} drug-cell links and {len(other_links)} other links to {NEW_TRAIN_FILE}")

    # Write the test file (link.dat.test)
    with open(TEST_FILE, 'w') as f:
        f.writelines(test_drug_cell_links)
    print(f"Wrote {len(test_drug_cell_links)} drug-cell links to {TEST_FILE}")

    print("\nData splitting complete.")