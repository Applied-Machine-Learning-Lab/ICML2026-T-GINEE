#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Assuming 'tgiee' module contains the previously defined classes
# Ensure tgiee.py is present in the python path
try:
    from tgiee import ScalableTGMEE, train_large_scale
except ImportError:
    # Fallback or placeholder if running independently without the module installed
    pass

def main():
    # 1. Configuration
    file_path = 'stackoverflow_edges.npy'
    
    print("Checking data...")
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    try:
        edges = np.load(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Dynamic detection of graph dimensions
    n_nodes = int(edges[:, :2].max() + 1)
    n_layers = int(edges[:, 2].max() + 1)
    
    print(f"Detected Nodes: {n_nodes}, Layers: {n_layers}")
    print(f"Total Edges: {len(edges)}")
    
    # Use CPU primarily for memory reasons on large graphs unless high-VRAM GPU is available
    device = torch.device("cpu") 
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Creating DataLoader...")
    # Time-based or random split (90/10)
    split_idx = int(len(edges) * 0.9)
    train_edges = edges[:split_idx]
    val_edges = edges[split_idx:]
    
    # Wrap in TensorDataset for robust DataLoader handling
    train_dataset = TensorDataset(torch.from_numpy(train_edges))
    val_dataset = TensorDataset(torch.from_numpy(val_edges))
    
    # Large batch size to optimize CPU throughput
    train_loader = DataLoader(train_dataset, batch_size=50000, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=50000, shuffle=False)

    # 3. Model Initialization
    print("Initializing Million-Scale Model...")
    model = ScalableTGMEE(n_nodes=n_nodes, n_layers=n_layers, embedding_dim=32).to(device)
    
    # 4. Training
    print("Start Training (Scalability Demonstration)...")
    model, history = train_large_scale(
        model, 
        train_loader, 
        val_loader, 
        epochs=20, 
        lr=0.01, 
        device=device
    )
    print("Training complete.")

    # ==========================================
    # 5. Evaluation (AUC Calculation)
    # ==========================================
    print("\nStarting Final Evaluation (Calculating AUC)...")
    model.eval()
    
    all_preds = []
    all_labels = []
    
    # Disable gradient calculation for inference efficiency
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            # Unwrap batch from TensorDataset (returns a list/tuple)
            batch_data = batch[0]
            
            # Get positive samples: u, v, k
            u = batch_data[:, 0].long().to(device)
            v = batch_data[:, 1].long().to(device)
            k = batch_data[:, 2].long().to(device)
            
            # 1. Predict positive sample scores
            # Note: ScalableTGMEE.forward(u, v) returns [Batch, n_layers]
            # We gather the probability for the specific existing layer 'k'
            preds_pos_all = model(u, v)
            pos_scores = preds_pos_all.gather(1, k.unsqueeze(1)).squeeze()
            
            # 2. Generate negative samples (Random corruption of v)
            # Assumption: Collision probability is negligible in million-node graphs
            neg_v = torch.randint(0, n_nodes, (len(u),), device=device)
            
            # Predict negative sample scores
            preds_neg_all = model(u, neg_v)
            neg_scores = preds_neg_all.gather(1, k.unsqueeze(1)).squeeze()
            
            # 3. Collect results
            # Positive samples = 1, Negative samples = 0
            all_preds.append(pos_scores.cpu().numpy())
            all_labels.append(np.ones(len(pos_scores)))
            
            all_preds.append(neg_scores.cpu().numpy())
            all_labels.append(np.zeros(len(neg_scores)))

    # Concatenate results
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    print("Computing ROC_AUC Score...")
    try:
        # Handle potential NaNs if model diverges
        if np.isnan(all_preds).any():
            all_preds = np.nan_to_num(all_preds)
            
        final_auc = roc_auc_score(all_labels, all_preds)
        print("="*50)
        print(f"FINAL RESULT >> StackOverflow ({n_nodes} Nodes) AUC: {final_auc:.4f}")
        print("="*50)
    except Exception as e:
        print(f"Error computing AUC: {e}")

if __name__ == "__main__":
    main()