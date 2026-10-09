#!/usr/bin/env python
# coding: utf-8

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

# ==========================================
# Loss Functions
# ==========================================

class BatchGEELoss(nn.Module):
    def __init__(self, regularization_weight=0.1):
        super(BatchGEELoss, self).__init__()
        self.reg_weight = regularization_weight
        self.bce = nn.BCELoss()

    def forward(self, prob, target, W_matrix):
        """
        Compute Batch-wise GEE Loss.
        prob: [Batch, M]
        target: [Batch, M]
        W_matrix: [M, M] Current global working covariance matrix
        """
        # 1. Base Prediction Error (BCE Loss)
        bce_loss = self.bce(prob, target)
        
        # 2. GEE Regularization Term (Capturing inter-layer correlation)
        # Formula: sum_ij (r_ij^T * Sigma_ij^-1 * r_ij)
        
        # Calculate Gamma^-1/2 (Standardization factor)
        variance = prob * (1 - prob) + 1e-6
        gamma_inv_sqrt = 1.0 / torch.sqrt(variance) # [Batch, M]
        
        # Calculate residuals r
        residual = target - prob # [Batch, M]
        
        # Scale residuals: r_tilde = r * gamma^-1/2
        r_tilde = residual * gamma_inv_sqrt # [Batch, M]
        
        # Calculate Inverse of W (M x M)
        # Add small perturbation for numerical stability
        M = W_matrix.size(0)
        W_inv = torch.inverse(W_matrix + 1e-4 * torch.eye(M, device=W_matrix.device))
        
        # Calculate quadratic form: r_tilde @ W_inv @ r_tilde^T
        # Efficiently calculate diagonal parts
        temp = torch.matmul(r_tilde, W_inv)
        gee_contribution = torch.sum(temp * r_tilde, dim=1).mean()
        
        total_loss = bce_loss + self.reg_weight * gee_contribution
        return total_loss, bce_loss, gee_contribution

class TGMEELoss(nn.Module):
    """
    T-GMEE loss function, combining cross-entropy loss and generalized estimating equation loss.
    """
    def __init__(self, regularization_weight=0.1):
        super(TGMEELoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.reg_weight = regularization_weight
    
    def forward(self, pred_probs, true_labels, model, adjacency_tensor):
        # Standard cross-entropy loss
        bce = self.bce_loss(pred_probs, true_labels)
        
        # Generalized estimating equation regularization loss
        gee_loss = model.compute_gee_loss(adjacency_tensor)
        
        # Combined loss
        total_loss = bce + self.reg_weight * gee_loss
        
        return total_loss, bce, gee_loss

# ==========================================
# Models
# ==========================================

class ScalableTGMEE(nn.Module):
    def __init__(self, n_nodes, n_layers, embedding_dim=32, link_fn='logit'):
        super(ScalableTGMEE, self).__init__()
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.link_fn = link_fn
        
        # Parameter definitions
        self.alpha = nn.Parameter(torch.randn(n_nodes, embedding_dim) * 0.01)
        self.beta = nn.Parameter(torch.randn(n_layers, embedding_dim) * 0.01)
        
        # Working covariance matrix W (M x M)
        self.register_buffer('W', torch.eye(n_layers))
        # Momentum for W updates
        self.momentum = 0.9

    def forward(self, idx_i, idx_j):
        """
        Batch forward pass.
        Input: 
            idx_i: [Batch_Size] Source node indices
            idx_j: [Batch_Size] Target node indices
        Output:
            prob: [Batch_Size, n_layers] Predicted probabilities
        """
        # 1. Retrieve embeddings for current batch
        alpha_i = self.alpha[idx_i]
        alpha_j = self.alpha[idx_j]
        
        # 2. Node interaction (Hadamard product for CP decomposition)
        node_interaction = alpha_i * alpha_j
        
        # 3. Projection to layers
        theta = torch.matmul(node_interaction, self.beta.T)
        
        # 4. Link function
        if self.link_fn == 'logit':
            prob = torch.sigmoid(theta)
        elif self.link_fn == 'identity':
            prob = theta
        else: # modified_logit
            s = 0.99
            prob = s * torch.sigmoid(theta)
            
        return prob

    def update_working_covariance_batch(self, prob_batch, label_batch):
        """
        Update working covariance matrix W based on current batch (Approximation).
        """
        with torch.no_grad():
            residual = label_batch - prob_batch
            
            # Standardize residuals (Pearson Residuals)
            variance = prob_batch * (1 - prob_batch) + 1e-6
            std_dev = torch.sqrt(variance)
            standardized_res = residual / std_dev
            
            # Calculate correlation matrix for batch R = 1/B * (Res^T @ Res)
            batch_W = torch.matmul(standardized_res.T, standardized_res) / prob_batch.size(0)
            
            # Momentum update
            self.W = self.momentum * self.W + (1 - self.momentum) * batch_W

class TGMEE(nn.Module):
    def __init__(self, n_nodes, n_layers, embedding_dim=32, link_fn='logit'):
        """
        Tensor Generalized Multilayer Graph Estimating Equation model.
        """
        super(TGMEE, self).__init__()
        
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        
        self.alpha = nn.Parameter(torch.randn(n_nodes, embedding_dim) * 0.1)
        self.beta = nn.Parameter(torch.randn(n_layers, embedding_dim) * 0.1)
        self.link_fn = link_fn
        self.W = nn.Parameter(torch.eye(n_layers), requires_grad=False)
    
    def get_gamma(self):
        alpha_vec = self.alpha.view(-1)
        beta_vec = self.beta.view(-1)
        return torch.cat([alpha_vec, beta_vec])
    
    def forward(self, edges=None):
        if edges is None:
            theta = self.compute_full_theta()
            if self.link_fn == 'identity':
                prob = theta
            elif self.link_fn == 'logit':
                prob = torch.sigmoid(theta)
            return prob
        else:
            i_indices, j_indices, k_indices = zip(*edges)
            
            i_embeddings = self.alpha[list(i_indices)]
            j_embeddings = self.alpha[list(j_indices)]
            k_embeddings = self.beta[list(k_indices)]
            
            theta_values = torch.sum(i_embeddings * j_embeddings * k_embeddings, dim=1)
            
            if self.link_fn == 'identity':
                prob = theta_values
            elif self.link_fn == 'logit':
                prob = torch.sigmoid(theta_values)
            
            return prob
    
    def compute_full_theta(self):
        theta = torch.zeros(self.n_nodes, self.n_nodes, self.n_layers, device=self.alpha.device)
        
        for r in range(self.embedding_dim):
            alpha_r = self.alpha[:, r].unsqueeze(1)
            beta_r = self.beta[:, r]
            alpha_outer = torch.matmul(alpha_r, alpha_r.transpose(0, 1))
            
            for m in range(self.n_layers):
                theta[:, :, m] += alpha_outer * beta_r[m]
        
        return theta
    
    def update_working_covariance(self, adjacency_tensor, theta_tensor):
        if self.link_fn == 'identity':
            P = theta_tensor
        elif self.link_fn == 'logit':
            P = torch.sigmoid(theta_tensor)
        
        residuals = []
        
        n = self.n_nodes
        # Optimization needed for large graphs, this part is computationally intensive
        for i in range(n):
            for j in range(i+1, n):
                P_ij = P[i, j, :]
                A_ij = adjacency_tensor[i, j, :]
                
                gamma_diag = P_ij * (1 - P_ij) + 1e-8
                gamma_inv_sqrt_ij = 1.0 / torch.sqrt(gamma_diag)
                residual = (A_ij - P_ij)
                scaled_residual = residual * gamma_inv_sqrt_ij
                
                residuals.append(scaled_residual)
        
        residuals = torch.stack(residuals)
        W_new = torch.matmul(residuals.T, residuals) * (2.0 / (n * (n + 1)))
        
        alpha = 0.9
        self.W.data = alpha * self.W.data + (1 - alpha) * W_new
    
    def compute_gee_loss(self, adjacency_tensor):
        theta = self.compute_full_theta()
        
        if self.link_fn == 'identity':
            P = theta
        elif self.link_fn == 'logit':
            P = torch.sigmoid(theta)
        
        loss = 0
        n_pairs = 0
        
        # Computing weighted residuals
        for i in range(self.n_nodes):
            for j in range(i+1, self.n_nodes):
                P_ij = P[i, j, :]
                A_ij = adjacency_tensor[i, j, :]
                
                Gamma_ij = torch.diag(P_ij * (1 - P_ij) + 1e-8)
                Sigma_ij_inv = torch.inverse(torch.matmul(torch.matmul(
                    torch.sqrt(Gamma_ij), self.W), torch.sqrt(Gamma_ij)
                ) + 1e-8 * torch.eye(self.n_layers, device=self.alpha.device))
                
                residual = (A_ij - P_ij)
                weighted_residual = torch.matmul(torch.matmul(residual, Sigma_ij_inv), residual)
                loss += weighted_residual
                n_pairs += 1
        
        gee_loss = loss / n_pairs
        return gee_loss

# ==========================================
# Datasets
# ==========================================

class SparseGraphDataset(Dataset):
    """
    Sparse dataset for large-scale graphs.
    Stores only positive samples (edges).
    """
    def __init__(self, file_path, n_nodes, mode='train', train_ratio=0.8):
        # Load edge list [E, 3] -> (u, v, layer)
        self.edge_list = np.load(file_path)
        self.n_nodes = n_nodes
        self.mode = mode
        
        n_edges = len(self.edge_list)
        indices = np.arange(n_edges)
        
        np.random.seed(42)
        np.random.shuffle(indices)
        
        split = int(n_edges * train_ratio)
        
        if mode == 'train':
            self.indices = indices[:split]
        else:
            self.indices = indices[split:]
            
        self.active_edges = self.edge_list[self.indices]

    def __len__(self):
        return len(self.active_edges)

    def __getitem__(self, idx):
        # Return positive sample: u, v, k
        return self.active_edges[idx]

class LargeScaleGraphDataset(Dataset):
    """
    Dataset optimized for large graphs (100k+ nodes).
    Features:
    1. Only stores positive samples.
    2. Dynamic negative sampling during training.
    """
    def __init__(self, tensor_path, mode='train', neg_ratio=1.0):
        print(f"Loading large scale data from {tensor_path}...")
        full_adj = np.load(tensor_path)
        
        self.n_nodes = full_adj.shape[0]
        self.n_layers = full_adj.shape[2]
        self.mode = mode
        self.neg_ratio = neg_ratio
        
        # Extract positive samples
        rows, cols, layers = np.where(full_adj > 0)
        
        # Filter upper triangle (u < v)
        mask = rows < cols
        self.pos_rows = rows[mask]
        self.pos_cols = cols[mask]
        self.pos_layers = layers[mask]
        
        self.n_pos = len(self.pos_rows)
        print(f"Dataset Mode: {mode} | Positive Samples: {self.n_pos} | Nodes: {self.n_nodes}")
        
        indices = np.arange(self.n_pos)
        np.random.shuffle(indices)
        split = int(0.8 * self.n_pos)
        
        if mode == 'train':
            self.indices = indices[:split]
        else:
            self.indices = indices[split:]
            
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        u = self.pos_rows[real_idx]
        v = self.pos_cols[real_idx]
        k = self.pos_layers[real_idx]
        
        if self.mode == 'train':
            # Dynamic negative sampling
            neg_u = np.random.randint(0, self.n_nodes)
            neg_v = np.random.randint(0, self.n_nodes)
            return np.array([u, v, k]), np.array([neg_u, neg_v, k])
        else:
            return np.array([u, v, k]), 1.0

class MultilayerGraphDataset(Dataset):
    def __init__(self, tensor_path, train_ratio=0.8, val_ratio=0.1, mode='train', random_seed=42):
        """
        Standard multilayer graph dataset for dense tensors.
        """
        self.adjacency_tensor = np.load(tensor_path)
        self.adjacency_tensor = torch.FloatTensor(self.adjacency_tensor)
        
        self.n_nodes = self.adjacency_tensor.shape[0]
        self.n_layers = self.adjacency_tensor.shape[2]
        
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        indices = []
        for i in range(self.n_nodes):
            for j in range(i+1, self.n_nodes):
                for k in range(self.n_layers):
                    indices.append((i, j, k))
        
        np.random.shuffle(indices)
        
        n_samples = len(indices)
        train_size = int(train_ratio * n_samples)
        val_size = int(val_ratio * n_samples)
        
        if mode == 'train':
            self.indices = indices[:train_size]
        elif mode == 'val':
            self.indices = indices[train_size:train_size+val_size]
        else:
            self.indices = indices[train_size+val_size:]
            
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        i, j, k = self.indices[idx]
        edge_exists = self.adjacency_tensor[i, j, k]
        return (i, j, k), edge_exists

# ==========================================
# Training & Evaluation Flows
# ==========================================

def train_large_scale(model, train_loader, val_loader, epochs=20, lr=0.01, device='cuda'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = BatchGEELoss(regularization_weight=0.1)
    
    history = {'train_loss': [], 'val_auc': []}
    
    print(f"Start training on device: {device}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for batch in pbar:
            # 1. Get positive samples [B, 3] -> u, v, k
            pos_u = batch[:, 0].long().to(device)
            pos_v = batch[:, 1].long().to(device)
            
            batch_size = pos_u.size(0)
            
            # 2. Dynamic Negative Sampling
            neg_u = torch.randint(0, model.n_nodes, (batch_size,), device=device)
            neg_v = torch.randint(0, model.n_nodes, (batch_size,), device=device)
            
            # 3. Concatenate
            u_all = torch.cat([pos_u, neg_u])
            v_all = torch.cat([pos_v, neg_v])
            
            # 4. Forward Pass
            preds_all_layers = model(u_all, v_all)
            
            # 5. Build Targets
            # Get layer index k
            pos_k = batch[:, 2].long().to(device)
            
            # Initialize target matrix [2B, n_layers]
            targets = torch.zeros_like(preds_all_layers)
            
            # Set target=1 for positive samples at layer k
            indices = torch.arange(batch_size, device=device)
            targets[indices, pos_k] = 1.0 
            
            # Calculate Loss
            loss, _, _ = criterion(preds_all_layers, targets, model.W)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Periodic W update
            if np.random.rand() < 0.05:
                model.update_working_covariance_batch(preds_all_layers, targets)

        avg_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        
        # Validation
        if (epoch + 1) % 5 == 0:
            val_auc, _, _ = evaluate_sparse(model, val_loader, device)
            history['val_auc'].append(val_auc)
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}")
        else:
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
            
    return model, history

def evaluate_sparse(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            # Positive samples
            pos_u = batch[:, 0].long().to(device)
            pos_v = batch[:, 1].long().to(device)
            pos_k = batch[:, 2].long().to(device)
            
            # Negative samples (1:1 for testing)
            neg_u = torch.randint(0, model.n_nodes, (pos_u.size(0),), device=device)
            neg_v = torch.randint(0, model.n_nodes, (pos_u.size(0),), device=device)
            
            u_all = torch.cat([pos_u, neg_u])
            v_all = torch.cat([pos_v, neg_v])
            
            # Predict [2B, M]
            preds = model(u_all, v_all)
            
            # Fine-grained evaluation: Check existence at layer k
            k_all = torch.cat([pos_k, pos_k]) 
            
            # Gather probability for specific layer
            final_preds = preds.gather(1, k_all.unsqueeze(1)).squeeze()
            
            # Labels
            labels = torch.cat([torch.ones(pos_u.size(0)), torch.zeros(pos_u.size(0))]).to(device)
            
            all_preds.append(final_preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    
    return roc_auc_score(y_true, y_pred), [], []

def train_scalable_tgmee(model, adj_tensor, epochs=50, batch_size=1024, lr=0.01, device='cuda'):
    """
    Scalable training loop alternative.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = BatchGEELoss(regularization_weight=0.1)
    
    print("Preprocessing positive sample indices...")
    any_edge_mask = np.any(adj_tensor > 0, axis=2)
    rows, cols = np.where(any_edge_mask)
    mask = rows < cols
    pos_rows, pos_cols = rows[mask], cols[mask]
    
    num_pos = len(pos_rows)
    print(f"Positive samples: {num_pos}")
    
    model.to(device)
    model.train()
    
    full_adj = torch.FloatTensor(adj_tensor) 
    
    for epoch in range(epochs):
        total_loss = 0
        perm = torch.randperm(num_pos)
        num_batches = (num_pos + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            batch_u = torch.from_numpy(pos_rows[idx]).long().to(device)
            batch_v = torch.from_numpy(pos_cols[idx]).long().to(device)
            
            targets_pos = full_adj[batch_u.cpu(), batch_v.cpu()].to(device)
            
            curr_batch_size = batch_u.size(0)
            neg_u = torch.randint(0, model.n_nodes, (curr_batch_size,), device=device)
            neg_v = torch.randint(0, model.n_nodes, (curr_batch_size,), device=device)
            
            targets_neg = torch.zeros_like(targets_pos)
            
            batch_u_all = torch.cat([batch_u, neg_u])
            batch_v_all = torch.cat([batch_v, neg_v])
            targets_all = torch.cat([targets_pos, targets_neg])
            
            optimizer.zero_grad()
            probs = model(batch_u_all, batch_v_all)
            
            loss, bce, gee = criterion(probs, targets_all, model.W)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                model.update_working_covariance_batch(probs, targets_all)
                
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        
    return model

def train_tgmee(model, train_loader, val_loader, adjacency_tensor, epochs=100, lr=0.01, weight_decay=1e-5, reg_weight=0.1, device="cuda"):
    """
    Standard training loop for TGMEE.
    """
    model = model.to(device)
    adjacency_tensor = adjacency_tensor.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = TGMEELoss(regularization_weight=reg_weight)
    
    history = {
        'train_loss': [], 'train_bce': [], 'train_gee': [],
        'val_loss': [], 'val_auc': []
    }
    
    with torch.no_grad():
        theta = model.compute_full_theta()
        model.update_working_covariance(adjacency_tensor, theta)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_bce = 0
        epoch_gee = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (edges, labels) in enumerate(pbar):
            edges = [(i.item(), j.item(), k.item()) for i, j, k in zip(*edges)]
            labels = labels.float().to(device)
            
            pred_probs = model(edges)
            loss, bce, gee = criterion(pred_probs, labels, model, adjacency_tensor)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_bce += bce.item()
            epoch_gee += gee.item()
            
            pbar.set_postfix({
                'train_loss': epoch_loss / (batch_idx + 1),
                'bce': epoch_bce / (batch_idx + 1),
                'gee': epoch_gee / (batch_idx + 1)
            })
        
        history['train_loss'].append(epoch_loss / len(train_loader))
        history['train_bce'].append(epoch_bce / len(train_loader))
        history['train_gee'].append(epoch_gee / len(train_loader))
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for edges, labels in val_loader:
                edges = [(i.item(), j.item(), k.item()) for i, j, k in zip(*edges)]
                labels = labels.float().to(device)
                pred_probs = model(edges)
                
                val_preds.extend(pred_probs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
            
            val_auc = roc_auc_score(val_labels, val_preds)
            history['val_auc'].append(val_auc)
            
            valid_loss, _, _ = criterion(torch.tensor(val_preds, device=device), 
                                        torch.tensor(val_labels, device=device), 
                                        model, adjacency_tensor)
            history['val_loss'].append(valid_loss.item())
        
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                theta = model.compute_full_theta()
                model.update_working_covariance(adjacency_tensor, theta)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {history['train_loss'][-1]:.4f}, "
              f"Val Loss: {history['val_loss'][-1]:.4f}, Val AUC: {history['val_auc'][-1]:.4f}")
    
    return model, history

def evaluate_tgmee(model, data_loader, device):
    """
    Robust evaluation function.
    Ensures shape consistency and flat arrays for metric calculation.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Testing", leave=False):
            if isinstance(batch, list):
                if len(batch) == 3: # [u, v, label]
                    u, v, labels = batch
                    u, v, labels = u.to(device), v.to(device), labels.to(device)
                    preds = model(u, v)
                elif len(batch) == 2: # [edges, label]
                    edges, labels = batch
                    labels = labels.to(device)
                    
                    if isinstance(edges, list): # edges is list [u, v]
                        u, v = edges[0], edges[1]
                        u, v = u.to(device), v.to(device)
                        preds = model(u, v)
                    else: # edges is Tensor
                        edges = edges.to(device)
                        if edges.dim() == 2:
                            if edges.shape[0] == 2: u, v = edges[0], edges[1]
                            else: u, v = edges[:, 0], edges[:, 1]
                            preds = model(u, v)
                        else:
                            preds = model(edges)
                else:
                    continue
            else:
                continue

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    if len(all_preds) > 0:
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)
        
        # Flatten for consistency
        y_pred_flat = y_pred.reshape(-1)
        y_true_flat = y_true.reshape(-1)
        
        # Check consistency
        if y_pred_flat.shape != y_true_flat.shape:
            # Attempt broadcast fix
            if y_pred_flat.shape[0] % y_true_flat.shape[0] == 0:
                ratio = y_pred_flat.shape[0] // y_true_flat.shape[0]
                y_true_flat = np.repeat(y_true_flat, ratio)
            elif y_true_flat.shape[0] % y_pred_flat.shape[0] == 0:
                ratio = y_true_flat.shape[0] // y_pred_flat.shape[0]
                y_pred_flat = np.repeat(y_pred_flat, ratio)
            else:
                print(f"Error: Shape mismatch. Pred: {y_pred_flat.shape}, Label: {y_true_flat.shape}")
                return 0.5, [], []

        try:
            # Handle NaN
            if np.isnan(y_pred_flat).any():
                y_pred_flat = np.nan_to_num(y_pred_flat)
                
            auc = roc_auc_score(y_true_flat, y_pred_flat)
        except Exception as e:
            print(f"AUC Error: {e}")
            auc = 0.5
    else:
        return 0.5, [], []
        
    return auc, all_preds, all_labels

def plot_results(history, dataset_name):
    """Plot training results"""
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{dataset_name} - Loss')
    plt.legend()
    
    # Plot BCE and GEE loss
    plt.subplot(1, 3, 2)
    plt.plot(history['train_bce'], label='BCE Loss')
    plt.plot(history['train_gee'], label='GEE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Components')
    plt.title(f'{dataset_name} - Loss Components')
    plt.legend()
    
    # Plot validation AUC
    plt.subplot(1, 3, 3)
    plt.plot(history['val_auc'], label='Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title(f'{dataset_name} - AUC')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_results.png')
    plt.show()

# ==========================================
# Main Execution & Analysis
# ==========================================

def main():
    # Set device configuration
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Experiment Configurations
    datasets = {
        # --- Scalable Mode ---
        'DBLP_5k': {
            'file_path': 'dblp_5000_edges.npy',   
            'n_nodes': 5000,
            'embedding_dim': 32,
            'epochs': 20,
            'batch_size': 4096,
            'lr': 0.01,
            'is_scalable': True
        },
        'DBLP_5k_Large': {
            'file_path': 'dblp_50000_edges.npy',  
            'n_nodes': 50000,
            'embedding_dim': 32,
            'epochs': 20,
            'batch_size': 8192,
            'lr': 0.01,
            'is_scalable': True
        },
        'DBLP_300k': {
            'file_path': 'dblp_300000_edges.npy', 
            'n_nodes': 317080,
            'embedding_dim': 32,
            'epochs': 20,
            'batch_size': 16384,
            'lr': 0.01,
            'is_scalable': True
        },
    }
    
    results = {}
    
    for dataset_name, config in datasets.items():
        print(f"\n{'-'*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'-'*60}")
        
        # Branch 1: Scalable Mode (Sparse)
        if config.get('is_scalable', False):
            print(f"--> [Scalable Mode] Launching sparse training (N={config['n_nodes']})...")
            
            file_path = config['file_path']
            if not os.path.exists(file_path):
                print(f"Error: File {file_path} not found, skipping.")
                continue
                
            train_dataset = SparseGraphDataset(file_path, n_nodes=config['n_nodes'], mode='train')
            val_dataset = SparseGraphDataset(file_path, n_nodes=config['n_nodes'], mode='val')
            
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
            
            model = ScalableTGMEE(
                n_nodes=config['n_nodes'], 
                n_layers=5, # Assuming 5 layers for DBLP
                embedding_dim=config['embedding_dim']
            ).to(device)
            
            model, history = train_large_scale(
                model,
                train_loader,
                val_loader,
                epochs=config['epochs'],
                lr=config['lr'],
                device=device
            )
            
            test_auc, _, _ = evaluate_sparse(model, val_loader, device)
            
        # Branch 2: Standard Mode (Dense)
        else:
            print(f"--> [Standard Mode] Launching full tensor training...")
            
            tensor_path = config['tensor_path']
            
            try:
                adj_numpy = np.load(tensor_path)
            except FileNotFoundError:
                print(f"Error: File {tensor_path} not found.")
                continue

            train_dataset = MultilayerGraphDataset(tensor_path, mode='train')
            val_dataset = MultilayerGraphDataset(tensor_path, mode='val')
            test_dataset = MultilayerGraphDataset(tensor_path, mode='test')
            
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
            test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
            
            adjacency_tensor = torch.FloatTensor(adj_numpy).to(device)
            n_nodes = adj_numpy.shape[0]
            n_layers = adj_numpy.shape[2]
            
            model = TGMEE(n_nodes, n_layers, embedding_dim=config['embedding_dim']).to(device)
            
            model, history = train_tgmee(
                model, 
                train_loader, 
                val_loader, 
                adjacency_tensor,
                epochs=config['epochs'],
                lr=config['lr'],
                reg_weight=config.get('reg_weight', 0.1),
                device=device
            )
            
            test_auc, _, _ = evaluate_tgmee(model, test_loader, device)

        print(f"\nResult: {dataset_name} | Test AUC: {test_auc:.4f}")
        results[dataset_name] = {'test_auc': test_auc}
        
        torch.save(model.state_dict(), f'{dataset_name}_model.pt')
    
    print("\n" + "="*60)
    print("Summary of Results (AUC):")
    print("="*60)
    for name, res in results.items():
        print(f"{name.ljust(15)}: {res['test_auc']:.4f}")
    print("="*60)

def hyperparameter_analysis():
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    tensor_path = 'krackhardt_tensor.npy'
    if not os.path.exists(tensor_path):
        print(f"File {tensor_path} not found.")
        return

    adjacency_tensor = np.load(tensor_path)
    adjacency_tensor_torch = torch.FloatTensor(adjacency_tensor).to(device)
    
    n_nodes = adjacency_tensor.shape[0]
    n_layers = adjacency_tensor.shape[2]
    
    batch_size = 1024
    epochs = 50 
    lr = 0.01
    
    embedding_dims = [4, 8, 16, 32, 64]
    reg_weights = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
    
    results = {}
    total_runs = len(embedding_dims) * len(reg_weights)
    run_count = 0
    
    train_dataset = MultilayerGraphDataset(tensor_path, mode='train', random_seed=42)
    val_dataset = MultilayerGraphDataset(tensor_path, mode='val', random_seed=42)
    test_dataset = MultilayerGraphDataset(tensor_path, mode='test', random_seed=42)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    results_df = pd.DataFrame(columns=['embedding_dim', 'reg_weight', 'train_loss', 'val_loss', 'val_auc', 'test_auc', 'training_time'])
    
    for embedding_dim in embedding_dims:
        for reg_weight in reg_weights:
            run_count += 1
            print(f"\n{'-'*60}")
            print(f"Run {run_count}/{total_runs}: embedding_dim={embedding_dim}, reg_weight={reg_weight}")
            print(f"{'-'*60}")
            
            model = TGMEE(n_nodes, n_layers, embedding_dim=embedding_dim).to(device)
            start_time = time.time()
            
            model, history = train_tgmee(
                model, 
                train_loader, 
                val_loader, 
                adjacency_tensor_torch,
                epochs=epochs,
                lr=lr,
                reg_weight=reg_weight,
                device=device
            )
            
            training_time = time.time() - start_time
            test_auc, _, _ = evaluate_tgmee(model, test_loader, device)
            
            results[(embedding_dim, reg_weight)] = {
                'train_loss': history['train_loss'][-1],
                'val_loss': history['val_loss'][-1],
                'val_auc': history['val_auc'][-1],
                'test_auc': test_auc,
                'training_time': training_time
            }
            
            new_row = pd.DataFrame({
                'embedding_dim': [embedding_dim],
                'reg_weight': [reg_weight],
                'train_loss': [history['train_loss'][-1]],
                'val_loss': [history['val_loss'][-1]],
                'val_auc': [history['val_auc'][-1]],
                'test_auc': [test_auc],
                'training_time': [training_time]
            })
            results_df = pd.concat([results_df, new_row], ignore_index=True)
            results_df.to_csv('hyperparameter_analysis_results.csv', index=False)
    
    best_idx = results_df['test_auc'].idxmax()
    best_params = results_df.iloc[best_idx]
    print("\n" + "="*70)
    print(f"Best hyperparameter combination:")
    print(f"Embedding dimension: {best_params['embedding_dim']}")
    print(f"Regularization weight: {best_params['reg_weight']}")
    print(f"Test AUC: {best_params['test_auc']:.4f}")
    print("="*70)
    
    return results_df, results

def visualize_hyperparameter_results(results_df):
    plt.figure(figsize=(12, 8))
    pivot_table = results_df.pivot_table(
        values='test_auc', 
        index='embedding_dim', 
        columns='reg_weight'
    )
    sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'Test AUC'})
    plt.title('Impact of Hyperparameters on Test AUC', fontsize=16)
    plt.xlabel('Regularization Weight', fontsize=14)
    plt.ylabel('Embedding Dimension', fontsize=14)
    plt.tight_layout()
    plt.savefig('hyperparameter_heatmap.png', dpi=300)
    plt.show()

def complexity_analysis(results_df):
    n_nodes = 21
    n_layers = 21
    
    results_df['model_params'] = results_df['embedding_dim'].apply(
        lambda dim: n_nodes * dim + n_layers * dim
    )
    
    plt.figure(figsize=(12, 6))
    for reg_weight in results_df['reg_weight'].unique():
        subset = results_df[results_df['reg_weight'] == reg_weight]
        plt.plot(subset['model_params'], subset['test_auc'], marker='o', label=f'Reg Weight = {reg_weight}')
    
    plt.title('Relationship Between Model Complexity and Test AUC', fontsize=16)
    plt.xlabel('Number of Model Parameters', fontsize=14)
    plt.ylabel('Test AUC', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('model_complexity.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()