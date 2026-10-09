import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader

# Ensure tgiee.py is in the same directory or Python path
try:
    from tgiee import TGMEE, train_tgmee
except ImportError:
    # Placeholder or handling if module is missing during standalone check
    pass

# Ignore unnecessary warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. Stratified Sampling Dataset Class
# ==========================================
class StratifiedMultilayerGraphDataset(Dataset):
    def __init__(self, tensor_path, train_ratio=0.8, val_ratio=0.1, mode='train', random_seed=42):
        # Load data using numpy
        self.adjacency_tensor = np.load(tensor_path)
        self.n_nodes = self.adjacency_tensor.shape[0]
        self.n_layers = self.adjacency_tensor.shape[2]
        
        np.random.seed(random_seed)
        
        # --- Stratified Sampling Logic ---
        pos_indices = []
        neg_indices = []
        
        for i in range(self.n_nodes):
            for j in range(i+1, self.n_nodes): 
                for k in range(self.n_layers):
                    if self.adjacency_tensor[i, j, k] > 0.5:
                        pos_indices.append((i, j, k))
                    else:
                        neg_indices.append((i, j, k))
                        
        np.random.shuffle(pos_indices)
        np.random.shuffle(neg_indices)
        
        n_pos = len(pos_indices)
        n_neg = len(neg_indices)
        
        pos_train_end = int(n_pos * train_ratio)
        pos_val_end = int(n_pos * (train_ratio + val_ratio))
        
        neg_train_end = int(n_neg * train_ratio)
        neg_val_end = int(n_neg * (train_ratio + val_ratio))
        
        if mode == 'train':
            self.indices = pos_indices[:pos_train_end] + neg_indices[:neg_train_end]
        elif mode == 'val':
            self.indices = pos_indices[pos_train_end:pos_val_end] + neg_indices[neg_train_end:neg_val_end]
        else: # test
            self.indices = pos_indices[pos_val_end:] + neg_indices[neg_val_end:]
            
        np.random.shuffle(self.indices)
            
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        i, j, k = self.indices[idx]
        edge_exists = self.adjacency_tensor[i, j, k]
        return (i, j, k), edge_exists

# ==========================================
# 2. Data Generation (With Minimum Edge Guarantee)
# ==========================================
def generate_robust_synthetic_data(n_nodes, n_layers, rho, theta, min_edges=10, seed=42):
    np.random.seed(seed)
    
    P_base = np.random.rand(n_nodes, n_nodes)
    P_base = (P_base + P_base.T) / 2
    U = np.random.rand(n_nodes, n_nodes, n_layers)
    for k in range(n_layers):
        U[:, :, k] = (U[:, :, k] + U[:, :, k].T) / 2
    P = rho * P_base[:, :, np.newaxis] + (1 - rho) * U
    adjacency_tensor = (P < theta).astype(np.float32)
    for k in range(n_layers):
        np.fill_diagonal(adjacency_tensor[:, :, k], 0)
    
    # Enforce minimum number of edges
    current_edges = np.sum(adjacency_tensor)
    if current_edges < min_edges:
        while np.sum(adjacency_tensor) < min_edges:
            i, j = np.random.randint(0, n_nodes, 2)
            k = np.random.randint(0, n_layers)
            if i != j and adjacency_tensor[i, j, k] == 0:
                adjacency_tensor[i, j, k] = 1.0
                adjacency_tensor[j, i, k] = 1.0 
                
    return adjacency_tensor

# ==========================================
# 3. Evaluation Function
# ==========================================
def rigorous_evaluate(model, test_loader, device):
    model.eval()
    pred_probs = []
    true_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 2:
                edges, labels = batch
            else:
                edges = batch
                labels = torch.zeros(len(edges[0]))
            
            u_list, v_list, k_list = edges
            edge_list = []
            for i in range(len(u_list)):
                edge_list.append((u_list[i].item(), v_list[i].item(), k_list[i].item()))
            
            labels = labels.float().to(device)
            outputs = model(edge_list)
            
            pred_probs.extend(outputs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    try:
        if len(np.unique(true_labels)) < 2:
             # Cannot calculate AUC with only one class, return 0.5 (random guess)
            return 0.5
        auc = roc_auc_score(true_labels, pred_probs)
    except Exception:
        auc = 0.5
        
    return auc

# ==========================================
# 4. Main Experiment (With Repeated Averaging)
# ==========================================
def run_experiment_with_averaging():
    # Force CPU usage for consistency and to avoid potential hardware compatibility issues
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # === Parameters setup for increased difficulty ===
    n_nodes = 100
    n_layers = 3
    rho = 0.05       # Extremely low correlation (adds noise)
    embedding_dim = 4 # Extremely low dimension (limits model capacity)
    
    epochs = 30
    batch_size = 5000 # Feed all data at once to accelerate CPU training
    lr = 0.01
    reg_weight = 0.1
    
    # Sparsity levels
    sparsity_levels = [0.001, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.15]
    
    # Number of repeats per point (for smoothing)
    n_repeats = 5 
    
    # Initialize results array
    all_results = np.zeros((len(sparsity_levels), n_repeats))
    
    print(f"\nStarting Averaged Sensitivity Experiment ({n_repeats} runs per level)...")
    print("="*60)
    
    for i, theta in enumerate(sparsity_levels):
        print(f"\n=== Processing Sparsity Level: {theta} ===")
        
        for run in range(n_repeats):
            # 1. Generate Data (Changing Seed)
            tensor = generate_robust_synthetic_data(n_nodes, n_layers, rho, theta, min_edges=10, seed=42+run+int(theta*1000))
            
            temp_filename = f'temp_tensor_{i}_{run}.npy'
            np.save(temp_filename, tensor)
            
            # 2. Datasets
            train_dataset = StratifiedMultilayerGraphDataset(temp_filename, mode='train', random_seed=42+run)
            test_dataset = StratifiedMultilayerGraphDataset(temp_filename, mode='test', random_seed=42+run)
            
            # Use full batch for acceleration
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            # 3. Initialize Model
            adjacency_tensor_torch = torch.FloatTensor(tensor).to(device)
            model = TGMEE(n_nodes, n_layers, embedding_dim=embedding_dim).to(device)
            
            # 4. Training (Catch exceptions for robustness)
            try:
                # Pass train_loader as placeholder for validation
                model, _ = train_tgmee(model, train_loader, train_loader, adjacency_tensor_torch, 
                                       epochs=epochs, lr=lr, reg_weight=reg_weight, device=device)
                
                # 5. Evaluation
                auc = rigorous_evaluate(model, test_loader, device)
            except Exception as e:
                print(f"Warning: Run failed ({e}), defaulting to 0.5")
                auc = 0.5
            
            all_results[i, run] = auc
            print(f"  Run {run+1}/{n_repeats} -> AUC: {auc:.4f}")
            
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    # --- Calculate Statistics ---
    means = np.mean(all_results, axis=1)
    stds = np.std(all_results, axis=1)
    
    # --- Plotting ---
    plt.figure(figsize=(8, 6))
    
    # Plot main line
    plt.plot(sparsity_levels, means, marker='o', linestyle='-', linewidth=2, label='Mean AUC')
    
    # Plot error band
    plt.fill_between(sparsity_levels, means - stds, means + stds, alpha=0.2, label='Std Dev')
    
    # Annotate values
    for i, txt in enumerate(means):
        plt.annotate(f"{txt:.2f}", (sparsity_levels[i], means[i]), 
                     textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.title(f'Sensitivity to Sparsity (Avg over {n_repeats} runs)', fontsize=14)
    plt.xlabel('Sparsity Level (Proportion of 1s)', fontsize=12)
    plt.ylabel('Test AUC', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.ylim(0.0, 1.05)
    
    plt.tight_layout()
    plt.savefig('sensitivity_avg.png', dpi=300)
    print("\nExperiment Complete! Saved to 'sensitivity_avg.png'")

if __name__ == "__main__":
    run_experiment_with_averaging()