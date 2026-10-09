#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm


# In[2]:


class MultilayerGraphDataset(Dataset):
    def __init__(self, tensor_path, train_ratio=0.8, val_ratio=0.1, mode='train', random_seed=42):
        """
        Multilayer graph dataset loading class
        
        Parameters:
            tensor_path: Path to tensor data
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            mode: 'train', 'val', or 'test'
            random_seed: Random seed for reproducibility
        """
        # Load tensor
        self.adjacency_tensor = np.load(tensor_path)
        
        # Convert to PyTorch tensor and move to GPU
        self.adjacency_tensor = torch.FloatTensor(self.adjacency_tensor)
        
        # Get tensor dimensions
        self.n_nodes = self.adjacency_tensor.shape[0]
        self.n_layers = self.adjacency_tensor.shape[2]
        
        # Split train/validation/test sets for edge prediction task
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # Create all possible edge indices
        indices = []
        for i in range(self.n_nodes):
            for j in range(i+1, self.n_nodes):  # Only consider upper triangular matrix
                for k in range(self.n_layers):
                    indices.append((i, j, k))
        
        # Randomly shuffle indices
        np.random.shuffle(indices)
        
        # Split dataset
        n_samples = len(indices)
        train_size = int(train_ratio * n_samples)
        val_size = int(val_ratio * n_samples)
        
        if mode == 'train':
            self.indices = indices[:train_size]
        elif mode == 'val':
            self.indices = indices[train_size:train_size+val_size]
        else:  # test
            self.indices = indices[train_size+val_size:]
            
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        i, j, k = self.indices[idx]
        edge_exists = self.adjacency_tensor[i, j, k]
        # Return edge index and label
        return (i, j, k), edge_exists


# In[3]:


class TGMEE(nn.Module):
    def __init__(self, n_nodes, n_layers, embedding_dim=32, link_fn='logit'):
        """
        Tensor Generalized Multilayer Graph Estimating Equation model
        
        Parameters:
            n_nodes: Number of nodes
            n_layers: Number of layers
            embedding_dim: Embedding dimension
            link_fn: Link function type, optional 'logit' or 'identity'
        """
        super(TGMEE, self).__init__()
        
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        
        # Initialize node embedding matrix α
        self.alpha = nn.Parameter(torch.randn(n_nodes, embedding_dim) * 0.1)
        
        # Initialize layer embedding matrix β
        self.beta = nn.Parameter(torch.randn(n_layers, embedding_dim) * 0.1)
        
        # Set link function
        self.link_fn = link_fn
        
        # Initialize working covariance matrix W
        self.W = nn.Parameter(torch.eye(n_layers), requires_grad=False)
    
    def get_gamma(self):
        """Return parameter vector γ = [vec(α)^T, vec(β)^T]^T"""
        alpha_vec = self.alpha.view(-1)
        beta_vec = self.beta.view(-1)
        return torch.cat([alpha_vec, beta_vec])
    
    def forward(self, edges=None):
        """
        Forward pass to compute edge probabilities
        
        Parameters:
            edges: List of edge index tuples [(i, j, k), ...]
        
        Returns:
            Predicted edge probabilities
        """
        if edges is None:
            # Compute complete parameter tensor Θ
            theta = self.compute_full_theta()
            
            # Apply link function to calculate probabilities
            if self.link_fn == 'identity':
                prob = theta
            elif self.link_fn == 'logit':
                prob = torch.sigmoid(theta)
            return prob
        else:
            # Only calculate probabilities for requested edges
            i_indices, j_indices, k_indices = zip(*edges)
            
            # Calculate theta[i,j,k] from embeddings
            i_embeddings = self.alpha[list(i_indices)]
            j_embeddings = self.alpha[list(j_indices)]
            k_embeddings = self.beta[list(k_indices)]
            
            # CP decomposition: theta[i,j,k] = sum_r alpha[i,r] * alpha[j,r] * beta[k,r]
            theta_values = torch.sum(i_embeddings * j_embeddings * k_embeddings, dim=1)
            
            # Apply link function
            if self.link_fn == 'identity':
                prob = theta_values
            elif self.link_fn == 'logit':
                prob = torch.sigmoid(theta_values)
            
            return prob
    
    def compute_full_theta(self):
        """Compute complete parameter tensor Θ using CP decomposition"""
        # Initialize parameter tensor
        theta = torch.zeros(self.n_nodes, self.n_nodes, self.n_layers, device=self.alpha.device)
        
        # Calculate for each embedding dimension
        for r in range(self.embedding_dim):
            # Calculate outer product alpha^(r) ○ alpha^(r) ○ beta^(r)
            alpha_r = self.alpha[:, r].unsqueeze(1)  # [n_nodes, 1]
            beta_r = self.beta[:, r]  # [n_layers]
            
            # Outer product of node embeddings
            alpha_outer = torch.matmul(alpha_r, alpha_r.transpose(0, 1))  # [n_nodes, n_nodes]
            
            # Apply layer embedding for each layer
            for m in range(self.n_layers):
                theta[:, :, m] += alpha_outer * beta_r[m]
        
        return theta
    
    def update_working_covariance(self, adjacency_tensor, theta_tensor):
        """
        Update working covariance matrix W
        
        Parameters:
            adjacency_tensor: Actually observed adjacency tensor
            theta_tensor: Model predicted parameter tensor
        """
        # Calculate probability tensor P
        if self.link_fn == 'identity':
            P = theta_tensor
        elif self.link_fn == 'logit':
            P = torch.sigmoid(theta_tensor)
        
        # Construct diagonal matrix Gamma[i,j]
        residuals = []
        gamma_inv_sqrt = []
        
        n = self.n_nodes
        for i in range(n):
            for j in range(i+1, n):
                P_ij = P[i, j, :]  # [n_layers]
                A_ij = adjacency_tensor[i, j, :]  # [n_layers]
                
                # Calculate Gamma[i,j]^(-1/2)
                gamma_diag = P_ij * (1 - P_ij) + 1e-8  # Avoid division by zero
                gamma_inv_sqrt_ij = 1.0 / torch.sqrt(gamma_diag)
                
                # Calculate residuals
                residual = (A_ij - P_ij)
                
                # Apply Gamma[i,j]^(-1/2)
                scaled_residual = residual * gamma_inv_sqrt_ij
                
                residuals.append(scaled_residual)
                gamma_inv_sqrt.append(gamma_inv_sqrt_ij)
        
        # Convert lists to tensors
        residuals = torch.stack(residuals)  # [n_pairs, n_layers]
        
        # Calculate W matrix
        W_new = torch.matmul(residuals.T, residuals) * (2.0 / (n * (n + 1)))
        
        # Update W in the model (can use smooth update)
        alpha = 0.9  # Smoothing coefficient
        self.W.data = alpha * self.W.data + (1 - alpha) * W_new
    
    def compute_gee_loss(self, adjacency_tensor):
        """
        Calculate loss based on generalized estimating equations
        
        Parameters:
            adjacency_tensor: Actually observed adjacency tensor
        
        Returns:
            gee_loss: T-GMEE based loss
        """
        # Get complete parameter tensor Θ
        theta = self.compute_full_theta()
        
        # Calculate probability tensor P
        if self.link_fn == 'identity':
            P = theta
        elif self.link_fn == 'logit':
            P = torch.sigmoid(theta)
        
        # Calculate GEE loss
        loss = 0
        n_pairs = 0
        
        # Calculate weighted residuals for each node pair
        for i in range(self.n_nodes):
            for j in range(i+1, self.n_nodes):
                # Get tensor slice for node pair (i,j)
                P_ij = P[i, j, :]  # [n_layers]
                A_ij = adjacency_tensor[i, j, :]  # [n_layers]
                
                # Calculate Sigma[i,j]^(-1)
                Gamma_ij = torch.diag(P_ij * (1 - P_ij) + 1e-8)  # Avoid division by zero
                Sigma_ij_inv = torch.inverse(torch.matmul(torch.matmul(
                    torch.sqrt(Gamma_ij), self.W), torch.sqrt(Gamma_ij)
                ) + 1e-8 * torch.eye(self.n_layers, device=self.alpha.device))
                
                # Calculate residuals
                residual = (A_ij - P_ij)
                
                # Calculate weighted sum of squared residuals
                weighted_residual = torch.matmul(torch.matmul(residual, Sigma_ij_inv), residual)
                loss += weighted_residual
                n_pairs += 1
        
        # Normalize loss
        gee_loss = loss / n_pairs
        return gee_loss


# In[4]:


class TGMEELoss(nn.Module):
    """
    T-GMEE loss function, combining cross-entropy loss and generalized estimating equation loss
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


# In[5]:


def train_tgmee(model, train_loader, val_loader, adjacency_tensor, epochs=100, lr=0.01, weight_decay=1e-5, reg_weight=0.1, device="cuda"):
    """
    Train T-GMEE model
    
    Parameters:
        model: TGMEE model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        adjacency_tensor: Complete adjacency tensor
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay coefficient
        reg_weight: Regularization weight
        device: Training device
    
    Returns:
        Model and training history
    """
    # Move model and data to GPU
    model = model.to(device)
    adjacency_tensor = adjacency_tensor.to(device)
    
    # Set optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Set loss function
    criterion = TGMEELoss(regularization_weight=reg_weight)
    
    # Initialize training history
    history = {
        'train_loss': [],
        'train_bce': [],
        'train_gee': [],
        'val_loss': [],
        'val_auc': []
    }
    
    # Update initial covariance matrix
    with torch.no_grad():
        theta = model.compute_full_theta()
        model.update_working_covariance(adjacency_tensor, theta)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_bce = 0
        epoch_gee = 0
        
        # Train one epoch
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (edges, labels) in enumerate(pbar):
            # Transfer data to device
            edges = [(i.item(), j.item(), k.item()) for i, j, k in zip(*edges)]
            labels = labels.float().to(device)
            
            # Forward pass
            pred_probs = model(edges)
            
            # Calculate loss
            loss, bce, gee = criterion(pred_probs, labels, model, adjacency_tensor)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update loss statistics
            epoch_loss += loss.item()
            epoch_bce += bce.item()
            epoch_gee += gee.item()
            
            # Update progress bar
            pbar.set_postfix({
                'train_loss': epoch_loss / (batch_idx + 1),
                'bce': epoch_bce / (batch_idx + 1),
                'gee': epoch_gee / (batch_idx + 1)
            })
        
        # Record training loss
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
            
            # Calculate validation AUC
            val_auc = roc_auc_score(val_labels, val_preds)
            history['val_auc'].append(val_auc)
            
            # Calculate validation loss
            valid_loss, _, _ = criterion(torch.tensor(val_preds, device=device), 
                                        torch.tensor(val_labels, device=device), 
                                        model, adjacency_tensor)
            history['val_loss'].append(valid_loss.item())
        
        # Periodically update covariance matrix
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                theta = model.compute_full_theta()
                model.update_working_covariance(adjacency_tensor, theta)
        
        # Print current progress
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {history['train_loss'][-1]:.4f}, "
              f"Val Loss: {history['val_loss'][-1]:.4f}, Val AUC: {history['val_auc'][-1]:.4f}")
    
    return model, history


# In[6]:


def evaluate_tgmee(model, test_loader, device="cuda"):
    """
    Evaluate T-GMEE model
    
    Parameters:
        model: TGMEE model instance
        test_loader: Test data loader
        device: Computation device
    
    Returns:
        auc: Area under ROC curve
        pred_probs: Predicted probabilities
        true_labels: True labels
    """
    model.eval()
    pred_probs = []
    true_labels = []
    
    with torch.no_grad():
        for edges, labels in tqdm(test_loader, desc="Testing"):
            edges = [(i.item(), j.item(), k.item()) for i, j, k in zip(*edges)]
            labels = labels.float().to(device)
            outputs = model(edges)
            
            pred_probs.extend(outputs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # Calculate AUC
    auc = roc_auc_score(true_labels, pred_probs)
    
    return auc, pred_probs, true_labels


# In[7]:


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


# In[ ]:


def main():
    # Detect GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset paths
    datasets = {
        # 'aucs': {
        #     'tensor_path': 'aucs_tensor.npy', # 0.92
        #     'embedding_dim': 16,
        #     'epochs': 0,
        #     'batch_size': 100000,
        #     'lr': 0.01,
        #     'reg_weight': 0.1
        # },
        # 'wat': {
        #     'tensor_path': 'wat_tensor.npy', # 0.87
        #     'labels_path': None,
        #     'embedding_dim': 16,
        #     'epochs': 0,
        #     'batch_size': 400000,
        #     'lr': 0.01,
        #     'reg_weight': 0.1
        # },
        # 'yeast': {
        #     'tensor_path': 'yeast_tensor.npy', # 0.91
        #     'embedding_dim': 16,
        #     'epochs': 0,
        #     'batch_size': 50000,
        #     'lr': 0.01,
        #     'reg_weight': 0.1
        # },
        # 'krackhardt': {
        #     'tensor_path': 'krackhardt_tensor.npy', #0.94
        #     'labels_path': None,
        #     'embedding_dim': 16,
        #     'epochs': 0,
        #     'batch_size': 10000,
        #     'lr': 0.01,
        #     'reg_weight': 0.1
        # }
        'synthetic': {
            'tensor_path': 'synthetic_multilayer_graph.npy',
            'embedding_dim': 32,
            'epochs': 50,
            'batch_size': 10000,
            'lr': 0.01,
            'reg_weight': 0.1
        },
    }
    
    # Train and evaluate each dataset
    results = {}
    
    for dataset_name, config in datasets.items():
        print(f"\n{'-'*50}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'-'*50}")
        
        # Load dataset
        tensor_path = config['tensor_path']
        
        # Create data loaders
        train_dataset = MultilayerGraphDataset(tensor_path, mode='train')
        val_dataset = MultilayerGraphDataset(tensor_path, mode='val')
        test_dataset = MultilayerGraphDataset(tensor_path, mode='test')
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
        
        # Load complete tensor for covariance estimation
        adjacency_tensor = torch.FloatTensor(np.load(tensor_path)).to(device)
        
        # Initialize model
        n_nodes = adjacency_tensor.shape[0]
        n_layers = adjacency_tensor.shape[2]
        model = TGMEE(n_nodes, n_layers, embedding_dim=config['embedding_dim']).to(device)
        
        # Train model
        model, history = train_tgmee(
            model, 
            train_loader, 
            val_loader, 
            adjacency_tensor,
            epochs=config['epochs'],
            lr=config['lr'],
            reg_weight=config['reg_weight'],
            device=device
        )
        
        # Evaluate model
        test_auc, pred_probs, true_labels = evaluate_tgmee(model, test_loader, device)
        print(f"\nTest AUC for {dataset_name}: {test_auc:.4f}")
        
        # Save results
        results[dataset_name] = {
            'test_auc': test_auc,
            'history': history
        }
        
        # Plot results
        plot_results(history, dataset_name)
        
        # Save model
        torch.save(model.state_dict(), f'{dataset_name}_tgmee_model.pt')
    
    # Print results for all datasets
    print("\n\n" + "="*60)
    print("Summary of Results (AUC):")
    print("="*60)
    for dataset_name, result in results.items():
        print(f"{dataset_name.ljust(15)}: {result['test_auc']:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()


# In[ ]:


# Hyperparameter analysis: impact of embedding dimension and regularization weight
def hyperparameter_analysis():
    # Set random seed to ensure reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Detect GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Krackhardt dataset
    tensor_path = 'krackhardt_tensor.npy'
    adjacency_tensor = np.load(tensor_path)
    adjacency_tensor_torch = torch.FloatTensor(adjacency_tensor).to(device)
    
    # Get dataset dimensions
    n_nodes = adjacency_tensor.shape[0]
    n_layers = adjacency_tensor.shape[2]
    
    # Set fixed parameters
    batch_size = 1024
    epochs = 50  # Reduce epochs to speed up analysis
    lr = 0.01
    
    # Hyperparameters to test
    embedding_dims = [4, 8, 16, 32, 64]
    reg_weights = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
    
    # Storage for results
    results = {}
    
    # Progress tracking
    total_runs = len(embedding_dims) * len(reg_weights)
    run_count = 0
    
    # Create data loaders (fixed random seed to ensure same split each time)
    train_dataset = MultilayerGraphDataset(tensor_path, mode='train', random_seed=42)
    val_dataset = MultilayerGraphDataset(tensor_path, mode='val', random_seed=42)
    test_dataset = MultilayerGraphDataset(tensor_path, mode='test', random_seed=42)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Create results storage table
    results_df = pd.DataFrame(columns=['embedding_dim', 'reg_weight', 'train_loss', 'val_loss', 'val_auc', 'test_auc', 'training_time'])
    
    # Hyperparameter grid search
    for embedding_dim in embedding_dims:
        for reg_weight in reg_weights:
            run_count += 1
            print(f"\n{'-'*60}")
            print(f"Run {run_count}/{total_runs}: embedding_dim={embedding_dim}, reg_weight={reg_weight}")
            print(f"{'-'*60}")
            
            # Initialize model
            model = TGMEE(n_nodes, n_layers, embedding_dim=embedding_dim).to(device)
            
            # Record training start time
            start_time = time.time()
            
            # Train model
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
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Test model
            test_auc, _, _ = evaluate_tgmee(model, test_loader, device)
            
            # Save results
            results[(embedding_dim, reg_weight)] = {
                'train_loss': history['train_loss'][-1],
                'val_loss': history['val_loss'][-1],
                'val_auc': history['val_auc'][-1],
                'test_auc': test_auc,
                'training_time': training_time
            }
            
            # Add to DataFrame - use concat instead of append
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
            
            # Save intermediate results to prevent loss due to interruption
            results_df.to_csv('hyperparameter_analysis_results.csv', index=False)
    
    # Print best results
    best_idx = results_df['test_auc'].idxmax()
    best_params = results_df.iloc[best_idx]
    print("\n" + "="*70)
    print(f"Best hyperparameter combination:")
    print(f"Embedding dimension: {best_params['embedding_dim']}")
    print(f"Regularization weight: {best_params['reg_weight']}")
    print(f"Test AUC: {best_params['test_auc']:.4f}")
    print("="*70)
    
    return results_df, results

# Visualize hyperparameter results
def visualize_hyperparameter_results(results_df):
    # Create heatmap showing impact of embedding_dim and reg_weight on test AUC
    plt.figure(figsize=(12, 8))
    
    # Prepare pivot table
    pivot_table = results_df.pivot_table(
        values='test_auc', 
        index='embedding_dim', 
        columns='reg_weight'
    )
    
    # Draw heatmap
    sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'Test AUC'})
    plt.title('Impact of Hyperparameters on Test AUC (Krackhardt Dataset)', fontsize=16)
    plt.xlabel('Regularization Weight', fontsize=14)
    plt.ylabel('Embedding Dimension', fontsize=14)
    plt.tight_layout()
    plt.savefig('hyperparameter_heatmap.png', dpi=300)
    plt.show()
    
    # Plot embedding_dim impact on AUC for different reg_weights
    plt.figure(figsize=(12, 6))
    
    for reg_weight in results_df['reg_weight'].unique():
        subset = results_df[results_df['reg_weight'] == reg_weight]
        plt.plot(subset['embedding_dim'], subset['test_auc'], marker='o', label=f'Reg Weight = {reg_weight}')
    
    plt.title('Impact of Embedding Dimension on Test AUC', fontsize=16)
    plt.xlabel('Embedding Dimension', fontsize=14)
    plt.ylabel('Test AUC', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('embedding_dim_impact.png', dpi=300)
    plt.show()
    
    # Plot reg_weight impact on AUC for different embedding_dims
    plt.figure(figsize=(12, 6))
    
    for dim in results_df['embedding_dim'].unique():
        subset = results_df[results_df['embedding_dim'] == dim]
        plt.plot(subset['reg_weight'], subset['test_auc'], marker='o', label=f'Embedding Dim = {dim}')
    
    plt.title('Impact of Regularization Weight on Test AUC', fontsize=16)
    plt.xlabel('Regularization Weight', fontsize=14)
    plt.ylabel('Test AUC', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('reg_weight_impact.png', dpi=300)
    plt.show()
    
    # Plot training time vs embedding_dim
    plt.figure(figsize=(10, 6))
    
    avg_times = results_df.groupby('embedding_dim')['training_time'].mean()
    plt.bar(avg_times.index, avg_times.values)
    plt.title('Impact of Embedding Dimension on Training Time', fontsize=16)
    plt.xlabel('Embedding Dimension', fontsize=14)
    plt.ylabel('Average Training Time (seconds)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    plt.savefig('embedding_dim_time.png', dpi=300)
    plt.show()
    
    # Plot test AUC vs training time scatter plot
    plt.figure(figsize=(12, 8))
    
    scatter = plt.scatter(
        results_df['training_time'], 
        results_df['test_auc'],
        c=results_df['embedding_dim'],
        s=results_df['reg_weight']*500 + 50,  # Adjust point size to represent reg_weight
        alpha=0.7,
        cmap='viridis'
    )
    
    # Add color bar to represent embedding_dim
    cbar = plt.colorbar(scatter)
    cbar.set_label('Embedding Dimension', fontsize=12)
    
    # Add text labels
    for i, row in results_df.iterrows():
        plt.annotate(
            f"({row['embedding_dim']}, {row['reg_weight']})",
            (row['training_time'], row['test_auc']),
            fontsize=8,
            alpha=0.8
        )
    
    plt.title('Performance vs Computational Cost Trade-off Analysis', fontsize=16)
    plt.xlabel('Training Time (seconds)', fontsize=14)
    plt.ylabel('Test AUC', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('performance_cost_tradeoff.png', dpi=300)
    plt.show()

# Complexity analysis: relationship between model parameter count and performance
def complexity_analysis(results_df):
    # Calculate number of model parameters for each embedding_dim
    n_nodes = 21  # Number of nodes in Krackhardt dataset
    n_layers = 21  # Number of layers in Krackhardt dataset
    
    # Add model parameter count column
    results_df['model_params'] = results_df['embedding_dim'].apply(
        lambda dim: n_nodes * dim + n_layers * dim  # Model parameter count = α parameters + β parameters
    )
    
    # Plot model complexity vs performance
    plt.figure(figsize=(12, 6))
    
    # Group by regularization weight
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
    
    # Plot regularization effect vs model complexity
    plt.figure(figsize=(12, 8))
    
    # Calculate overfitting degree for each configuration (difference between validation and test AUC)
    results_df['overfit_gap'] = results_df['val_auc'] - results_df['test_auc']
    
    pivot_overfit = results_df.pivot_table(
        values='overfit_gap', 
        index='embedding_dim', 
        columns='reg_weight'
    )
    
    sns.heatmap(pivot_overfit, annot=True, fmt=".4f", cmap="coolwarm", center=0,
                cbar_kws={'label': 'Overfitting Degree (Val AUC - Test AUC)'})
    plt.title('Regularization Effect vs Model Complexity', fontsize=16)
    plt.xlabel('Regularization Weight', fontsize=14)
    plt.ylabel('Embedding Dimension (Complexity)', fontsize=14)
    plt.tight_layout()
    plt.savefig('regularization_effect.png', dpi=300)
    plt.show()

# Run hyperparameter analysis
if __name__ == "__main__":
    import time
    import seaborn as sns
    import pandas as pd
    
    # Record analysis start time
    analysis_start_time = time.time()
    
    print(f"Starting hyperparameter analysis for Krackhardt dataset...")
    print(f"Current time: 2025-05-13 12:58:31")
    print(f"User: Chenniubility")
    print(f"{'-'*60}")
    
    # Perform hyperparameter analysis
    results_df, results_dict = hyperparameter_analysis()
    
    # Visualize results
    visualize_hyperparameter_results(results_df)
    
    # Perform complexity analysis
    complexity_analysis(results_df)
    
    # Total analysis time
    total_time = time.time() - analysis_start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nHyperparameter analysis complete!")
    print(f"Total time: {int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds")
    print(f"Results saved to 'hyperparameter_analysis_results.csv'")
    print(f"Visualization charts saved as PNG files")


# In[ ]: