import argparse

def read_args():
    parser = argparse.ArgumentParser()
    
    # --- Optimizer & Learning ---
    parser.add_argument('--weight_decay', type=float, default=0, help='Adam optimizer weight decay')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--batch_s', type=int, default=20000, help='batch size')
    parser.add_argument('--mini_batch_s', type=int, default=200, help='mini batch size')
    parser.add_argument('--train_iter_n', type=int, default=310, help='max number of training iteration')
    parser.add_argument('--save_model_freq', type=float, default=10, help='number of iterations to save model')

    # --- GNN Architecture ---
    parser.add_argument('--embed_d', type=int, default=128, help='embedding dimension')
    parser.add_argument('--n_layers', type=int, default=2, help='number of GNN layers')

    # --- Random Walk ---
    parser.add_argument('--walk_n', type=int, default=10, help='number of walk per root node')
    parser.add_argument('--walk_L', type=int, default=30, help='length of each walk')
    parser.add_argument('--window', type=int, default=5, help='window size for relation extraction')

    # --- Paths and Modes ---
    parser.add_argument('--data', type=str, default='amazon', help='select data path')
    parser.add_argument('--model_path', type=str, default='../model_save/', help='path to save model')
    parser.add_argument("--checkpoint", default='', type=str)
    parser.add_argument("--random_seed", default=10, type=int)
    parser.add_argument('--train_test_label', type=int, default=0, help='0: train, 1: test, 2: generate eval negatives')
    parser.add_argument("--cuda", default=0, type=int)
    parser.add_argument('--feat_type', default=0, type=int, help='0: ID embedding; 1: load from dataloader')

    # --- Model Feature Toggles ---
    parser.add_argument('--use_skip_connection', action='store_true', help='Enable skip connections in final embedding')
    parser.add_argument('--use_node_isolation', action='store_true', help='Enable curriculum learning via node isolation')
    parser.add_argument('--use_vae_encoder', action='store_true', help='Use VAE encoder for cell features')
    parser.add_argument('--use_concat_bilinear', action='store_true', help='Use bilinear layer on concatenated embeddings')
    parser.add_argument('--use_multitask_loss', action='store_true', help='Use multi-task loss with learned uncertainties')
    parser.add_argument('--use_rw_loss', action='store_true', help='Enable self-supervised RW skip-gram loss')

    # --- VAE-related ---
    parser.add_argument('--vae_checkpoint', type=str, default='data/embeddings/cell_vae_weights.pth',
                        help='Path to trained VAE weights')
    parser.add_argument('--vae_dims', type=str, default='19193,10000,5000,1000,500,256',
                        help='Comma-separated dims for VAE layers (used if use_vae_encoder=True)')

    return parser.parse_args()