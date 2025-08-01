# PRELUDE: A Graph Neural Network for Inductive Drug Response Prediction

PRELUDE (Predictive Link-learning for Unseen Drug Efficacy) is a graph neural network framework designed to predict drug-cell line interactions. By constructing a heterogeneous graph of cells, drugs, and genes, PRELUDE leverages deep learning to model complex biological relationships and predict drug efficacy, with a focus on generalizing to unseen entities (inductive learning).

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---
## Features
This framework incorporates several modern GNN techniques for robust link prediction:
* **Heterogeneous Graph Network**: Models cells, drugs, and genes, and the multi-relational links between them.
* **Inductive Capability**: Designed to generate predictions for nodes not seen during training.
* **Pre-trained Feature Integration**: Initializes nodes with rich, pre-trained embeddings:
    * **Cells**: A Variational Autoencoder (VAE) compresses high-dimensional gene expression data into dense embeddings.
    * **Drugs**: Uses embeddings from MoleculeSTM.
    * **Genes**: Uses embeddings from ESM3.
* **Hybrid Training Objective**: Combines a supervised link prediction loss with an optional self-supervised random walk loss to learn both specific interactions and general graph topology.
* **Advanced Architectural Options**: Includes configurable options like skip connections and curriculum learning (node isolation).

---
## Project Structure

The repository is organized into the following key directories:
```
.
├── checkpoints/      # Stores saved model weights
├── config/           # Contains command-line argument definitions
├── data/             # Contains all raw, processed, and embedding data
├── dataloaders/      # Scripts for loading graph data and features
├── models/           # GNN model and layer definitions
├── notebooks/        # Jupyter notebooks for exploration and data processing
└── scripts/          # Executable scripts for training, evaluation, plotting, etc.
```

---
## Installation and Setup

Follow these steps to set up the environment and run the project.

1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/luistafoi/PRELUDE.git](https://github.com/luistafoi/PRELUDE.git)
    cd PRELUDE
    ```

2.  **Download LFS Data**:
    This project uses Git LFS to manage large data files. Ensure Git LFS is installed, then run:
    ```bash
    git lfs pull
    ```

3.  **Create and Activate Conda Environment**:
    ```bash
    # Create the environment
    conda create --name prelude_env python=3.11 -y

    # Activate the environment
    conda activate prelude_env
    ```

4.  **Install Dependencies**:
    Install PyTorch with CUDA support, followed by the remaining packages.
    ```bash
    # Install PyTorch (example for CUDA 12.1)
    pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

    # Install other requirements
    pip install -r requirements.txt
    ```

---
## Usage

The project follows a three-step workflow: data preprocessing, model training, and final evaluation.

### 1. Preprocessing (One-Time Setup)

The data must be split into training, validation, and test sets, and the neighbor list for the GNN must be generated.

```bash
# Step 1.1: Create train/valid/test splits from your complete link.dat
python scripts/create_splits.py

# Step 1.2: Generate the neighbor list for the GNN from the training graph
python scripts/generate_neighbors.py
```

### 2. Model Training

Run the main training script. You can enable or disable features using command-line flags. The best performing model will be saved in the `checkpoints/` directory.

```bash
# Example: Train the full model on GPU 2 with a batch size of 1024
python scripts/train.py --gpu 2 --mini_batch_s 1024 --use_vae_encoder --use_node_isolation --use_skip_connection --use_rw_loss
```

### 3. Final Evaluation

Once a model is trained, use the `evaluate.py` script to get its performance on the held-out test set.

```bash
# Use the same feature flags you trained with
python scripts/evaluate.py --gpu 2 --load_path checkpoints/prelude_model.pth --use_skip_connection
```

### 4. Visualizing Results

To plot the training and validation curves from a training run, use the `plot_results.py` script.

```bash
python scripts/plot_results.py checkpoints/prelude_model_log.csv
```

---
## Contributing

Contributions to the project are welcome. Please follow standard fork-and-pull-request workflows.

---
## License

This project is licensed under the Apache 2.0 License - see the `LICENSE` file for details.
