from models import load_encoder, ContrastiveNetwork
from dataloader import load_dataset
from utils.engine import evaluate
from utils.tsne_generator import tsne_generator
from utils.save_cluster import save_cluster
from utils.dataset_fn import collate_fn_test

import torch
from torch.utils.data import DataLoader

import argparse
import os, sys
import time

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    
    # GPU
    parser.add_argument("--use-cuda", action="store_true")
    
    # Dataset
    parser.add_argument("--dataset", type=str, default='image') # 'image', 'mask', 'seq'
    
    # Model 
    parser.add_argument("--encoder", type=str, default='ResNet') # Encoder (ResNet, LSTMNet)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument("--instance-temperature", type=float, default=0.5)
    parser.add_argument("--cluster-temperature", type=float, default=1.0)
    
    # Hyperparameters
    parser.add_argument("--batch-size", type=int, default=32)
    
    # Load
    parser.add_argument("--weights-filename", type=str)
    
    # Save
    parser.add_argument("--tsne", action="store_true") # Save T-SNE Visualization
    parser.add_argument("--extr", action='store_true') # Extract Original Image Files
    
    return parser

def print_setup(device, args):
    print("=======================[Settings]========================")
    print(f"\n  [GPU]")
    print(f"  |-[device]: {device}")
    print(f"\n  [MODEL]")
    print(f"  |-[encoder]: {args.encoder}")
    print(f"  |-[feature dim]: {args.feature_dim}")
    print(f"  |-[instance temperature]: {args.instance_temperature}")
    print(f"  |-[cluster temperature]: {args.cluster_temperature}")
    print(f"  |-[n clusters]: {args.n_clusters}")
    print(f"\n  [DATA]")
    print(f"  |-[dataset]: {args.dataset}")
    print(f"\n  [HYPERPARAMETERS]")
    print(f"  |-[batch size]: {args.batch_size}")
    print(f"\n  [LOAD]")
    print(f"  |-[weights filename]: {args.weights_filename}")
    print("\n=======================================================")
    
    print("Proceed? [Y/N]: ", end="")
    proceed = input().lower()
    
    if proceed == 'n':
        sys.exit()
        
def main(args):
    device = 'cpu'
    if args.use_cuda and torch.cuda.is_available():
        device = 'cuda'
        
    print_setup(device, args)
    
    # Load Model
    
    encoder = load_encoder(args.encoder, args.dataset)
    model = ContrastiveNetwork(encoder, 
                               feature_dim=args.feature_dim,
                               class_num=args.n_clusters).to(device)
    ckpt = torch.load(os.path.join('saved/weights', args.weights_filename),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    print(f"It was trained {ckpt['epochs']} EPOCHS")
    
    # Load Dataset
    
    ds = load_dataset(dataset=args.dataset, mode='test', data_dir="data/crop_0_diff_3/crop_0_diff_3", encoder=args.encoder)
    dl = DataLoader(ds, 
                    shuffle=False, 
                    batch_size=args.batch_size)
    
    # Evaluate
    
    start_time = int(time.time())
    cluster_assignments, instance_vectors, data_paths = evaluate(model, dl, device)
    elapsed_time = int(time.time() - start_time)
    print(f"Evaluate Time: {elapsed_time//60:02d}m {elapsed_time%60:02d}s", end="\n\n")
    
    if args.tsne == True:
        start_time = int(time.time())
        tsne_generator(instance_vectors, cluster_assignments, args.dataset, args.n_clusters)
        elapsed_time = int(time.time() - start_time)
        print(f"t-SNE Time: {elapsed_time//60:02d}m {elapsed_time%60:02d}s", end="\n\n")
    
    if args.extr == True:
        start_time = int(time.time())
        save_cluster(cluster_assignments, data_paths, args.dataset, args.n_clusters)
        elapsed_time = int(time.time() - start_time)
        print(f"Clustering files Time: {elapsed_time//60:02d}m {elapsed_time%60:02d}s", end="\n\n")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate Contrastive Clustering', parents=[get_args_parser()])
    
    args = parser.parse_args()
    
    main(args)