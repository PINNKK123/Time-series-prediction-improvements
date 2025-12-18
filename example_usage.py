"""
Example usage of GNN-STAR model for InSAR time series prediction.
Demonstrates the complete pipeline with all improvements.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from gnn_star import (
    GNNSTAR,
    GNNSTARTrainer,
    InSARDataPreprocessor,
    VMDDecomposer,
    DynamicKCLConstraint,
    bayesian_hyperparameter_optimization
)


class SyntheticInSARDataset(Dataset):
    """
    Synthetic InSAR time series dataset for demonstration.
    """
    
    def __init__(self, n_samples=100, n_nodes=10, seq_len=50, n_features=3):
        """
        Generate synthetic InSAR data.
        
        Args:
            n_samples: Number of samples
            n_nodes: Number of spatial nodes
            seq_len: Sequence length (time steps)
            n_features: Number of features per node
        """
        self.n_samples = n_samples
        self.n_nodes = n_nodes
        self.seq_len = seq_len
        self.n_features = n_features
        
        # Generate synthetic data
        self.data = []
        self.targets = []
        self.edge_indices = []
        self.adjacency_matrices = []
        
        for _ in range(n_samples):
            # Generate time series with seasonal and trend components
            t = np.linspace(0, 4 * np.pi, seq_len)
            
            # Create node data with different patterns
            node_data = np.zeros((n_nodes, seq_len, n_features))
            for node in range(n_nodes):
                for feat in range(n_features):
                    # Seasonal component
                    seasonal = np.sin(t + node * 0.5 + feat * 0.3)
                    # Trend component
                    trend = 0.01 * t * (node + 1)
                    # Noise
                    noise = np.random.randn(seq_len) * 0.1
                    
                    node_data[node, :, feat] = seasonal + trend + noise
            
            # Target: predict next time step for each node
            target = np.zeros((n_nodes, seq_len, 1))
            for node in range(n_nodes):
                # Shift by one time step
                target[node, :-1, 0] = node_data[node, 1:, 0]
                target[node, -1, 0] = node_data[node, -1, 0]  # Use last value for final step
            
            # Create simple adjacency matrix (grid-like structure)
            adj_matrix = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes - 1):
                adj_matrix[i, i + 1] = 1
                adj_matrix[i + 1, i] = 1
            
            # Create edge index from adjacency
            edge_index = []
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if adj_matrix[i, j] > 0:
                        edge_index.append([i, j])
            edge_index = np.array(edge_index).T if edge_index else np.array([[], []])
            
            self.data.append(node_data)
            self.targets.append(target)
            self.edge_indices.append(edge_index)
            self.adjacency_matrices.append(adj_matrix)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.data[idx]),
            torch.FloatTensor(self.targets[idx]),
            torch.LongTensor(self.edge_indices[idx]) if self.edge_indices[idx].size > 0 else None,
            torch.FloatTensor(self.adjacency_matrices[idx])
        )


def collate_fn(batch):
    """Custom collate function for batching."""
    x_batch = torch.stack([item[0] for item in batch])
    y_batch = torch.stack([item[1] for item in batch])
    
    # For simplicity, use the first edge_index (assuming same graph structure)
    edge_index = batch[0][2]
    adjacency = batch[0][3]
    
    return x_batch, y_batch, edge_index, adjacency


def demonstrate_preprocessing():
    """Demonstrate data preprocessing pipeline."""
    print("=" * 60)
    print("1. Data Preprocessing Demonstration")
    print("=" * 60)
    
    # Generate sample data
    n_samples, n_features, n_timesteps = 5, 3, 100
    data = np.random.randn(n_samples, n_features, n_timesteps)
    
    # Add some anomalies
    data[0, 0, 10] = 100  # Outlier
    data[1, 1, 50] = -100  # Outlier
    
    timestamps = np.linspace(0, 99, n_timesteps)
    
    # Create preprocessor
    preprocessor = InSARDataPreprocessor(
        anomaly_threshold=3.0,
        gaussian_sigma=1.5,
        boundary_padding=2
    )
    
    # Preprocess data
    processed_data, anomaly_mask = preprocessor.preprocess(data, timestamps)
    
    print(f"Original data shape: {data.shape}")
    print(f"Processed data shape: {processed_data.shape}")
    print(f"Number of anomalies detected: {np.sum(anomaly_mask)}")
    print("✓ Preprocessing completed successfully\n")


def demonstrate_vmd():
    """Demonstrate VMD decomposition."""
    print("=" * 60)
    print("2. VMD Decomposition Demonstration")
    print("=" * 60)
    
    # Generate sample signal
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t) + 0.1 * np.random.randn(len(t))
    
    # Create VMD decomposer with enhanced parameter space
    vmd = VMDDecomposer(
        k_range=(3, 8),
        alpha_range=(1e2, 1e4)
    )
    
    # Optimize parameters
    print("Optimizing VMD parameters...")
    best_params, best_quality = vmd.optimize_parameters(signal, n_trials=10)
    print(f"Best parameters: k={best_params['k']}, alpha={best_params['alpha']:.2f}")
    print(f"Quality score: {best_quality:.4f}")
    
    # Decompose with best parameters
    u, u_hat, omega = vmd.decompose(signal)
    print(f"Decomposed into {u.shape[0]} modes")
    print(f"Center frequencies: {omega}")
    print("✓ VMD decomposition completed successfully\n")


def demonstrate_kcl_constraint():
    """Demonstrate dynamic KCL constraint."""
    print("=" * 60)
    print("3. Dynamic KCL Constraint Demonstration")
    print("=" * 60)
    
    # Create sample node values
    batch_size, n_nodes, n_features = 2, 5, 10
    node_values = torch.randn(batch_size, n_nodes, n_features)
    
    # Create edge index (simple chain)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                               [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
    
    # Create KCL constraint
    kcl = DynamicKCLConstraint(
        window_size=5,
        base_threshold=0.1,
        std_multiplier=2.0,
        directional_penalty_weight=0.5
    )
    
    # Compute KCL loss
    loss, components = kcl.compute_kcl_loss(node_values, edge_index=edge_index)
    
    print(f"Total KCL loss: {loss.item():.4f}")
    print(f"Soft violations: {components['soft_loss']:.4f}")
    print(f"Hard violations: {components['hard_loss']:.4f}")
    print(f"Directional penalty: {components['directional_loss']:.4f}")
    print(f"Adaptive threshold: {components['adaptive_threshold']:.4f}")
    print(f"Mean violation: {components['mean_violation']:.4f}")
    print("✓ KCL constraint computation completed successfully\n")


def demonstrate_model_training():
    """Demonstrate model training."""
    print("=" * 60)
    print("4. Model Training Demonstration")
    print("=" * 60)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create datasets
    print("Creating synthetic datasets...")
    train_dataset = SyntheticInSARDataset(n_samples=80, n_nodes=5, seq_len=30, n_features=3)
    val_dataset = SyntheticInSARDataset(n_samples=20, n_nodes=5, seq_len=30, n_features=3)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    # Create model with enhanced architecture
    print("Creating GNN-STAR model...")
    model = GNNSTAR(
        input_dim=3,
        hidden_dim=64,  # Enhanced hidden dimension (24-128 range)
        output_dim=1,
        num_gru_layers=2,
        num_edge_conv_layers=2,
        num_attention_heads=4,
        k_neighbors=3,
        dropout=0.2,  # Enhanced dropout (0.1-0.4 range)
        use_edge_conv=True
    )
    
    # Create KCL constraint
    kcl_constraint = DynamicKCLConstraint()
    
    # Create trainer
    trainer = GNNSTARTrainer(
        model=model,
        kcl_constraint=kcl_constraint,
        device=device,
        lr=0.001,  # Enhanced learning rate range (0.0001-0.01)
        kcl_weight=1.0,
        noise_level=0.01,
        noise_decay=0.99
    )
    
    # Train
    print("Training model...")
    history = trainer.train(train_loader, val_loader, epochs=30, verbose=True)
    
    # Print final metrics
    final_metrics = history['val_metrics'][-1]
    print("\nFinal Validation Metrics:")
    print(f"  MSE: {final_metrics['mse']:.4f}")
    print(f"  MAE: {final_metrics['mae']:.4f}")
    print(f"  R²: {final_metrics['r2']:.4f}")
    print(f"  Adjusted R²: {final_metrics['adjusted_r2']:.4f}")
    print("✓ Model training completed successfully\n")
    
    return history


def demonstrate_bayesian_optimization():
    """Demonstrate Bayesian hyperparameter optimization."""
    print("=" * 60)
    print("5. Bayesian Hyperparameter Optimization Demonstration")
    print("=" * 60)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create small datasets for fast optimization
    print("Creating datasets for hyperparameter optimization...")
    train_dataset = SyntheticInSARDataset(n_samples=40, n_nodes=5, seq_len=20, n_features=3)
    val_dataset = SyntheticInSARDataset(n_samples=10, n_nodes=5, seq_len=20, n_features=3)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    # Run Bayesian optimization (reduced trials for demo)
    print("Running Bayesian optimization (this may take a while)...")
    print("Note: Using reduced trials for demonstration purposes")
    
    best_params, study = bayesian_hyperparameter_optimization(
        train_loader=train_loader,
        val_loader=val_loader,
        n_trials=5,  # Reduced for demo (normally 50+)
        device=device,
        verbose=True
    )
    
    print("✓ Bayesian optimization completed successfully\n")


def plot_training_history(history):
    """Plot training history."""
    print("=" * 60)
    print("6. Visualizing Training History")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training and validation loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # R² score over epochs
    r2_scores = [m['r2'] for m in history['val_metrics']]
    adjusted_r2_scores = [m['adjusted_r2'] for m in history['val_metrics']]
    axes[0, 1].plot(r2_scores, label='R²', linewidth=2)
    axes[0, 1].plot(adjusted_r2_scores, label='Adjusted R²', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('R² Score')
    axes[0, 1].set_title('R² Metrics Over Training')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # MSE over epochs
    mse_scores = [m['mse'] for m in history['val_metrics']]
    axes[1, 0].plot(mse_scores, linewidth=2, color='orange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].set_title('Validation MSE Over Training')
    axes[1, 0].grid(True, alpha=0.3)
    
    # MAE over epochs
    mae_scores = [m['mae'] for m in history['val_metrics']]
    axes[1, 1].plot(mae_scores, linewidth=2, color='green')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].set_title('Validation MAE Over Training')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("Training history plot saved as 'training_history.png'")
    print("✓ Visualization completed successfully\n")


def main():
    """Main demonstration function."""
    print("\n" + "=" * 60)
    print("GNN-STAR Model Demonstration")
    print("Time Series Prediction with InSAR Data")
    print("=" * 60 + "\n")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run demonstrations
    demonstrate_preprocessing()
    demonstrate_vmd()
    demonstrate_kcl_constraint()
    history = demonstrate_model_training()
    plot_training_history(history)
    
    # Optional: Uncomment to run Bayesian optimization (takes longer)
    # demonstrate_bayesian_optimization()
    
    print("=" * 60)
    print("All Demonstrations Completed Successfully!")
    print("=" * 60)
    print("\nKey Improvements Implemented:")
    print("✓ Enhanced data preprocessing with anomaly removal and Gaussian filtering")
    print("✓ Extended VMD parameter space (k ∈ [3, 8], alpha ∈ [10², 10⁴])")
    print("✓ Dynamic KCL constraints with adaptive thresholds")
    print("✓ EdgeConv layers for dynamic graph construction")
    print("✓ GRU with self-attention for temporal modeling")
    print("✓ Enhanced hidden dimensions (24-128 range)")
    print("✓ Improved regularization with soft/hard violation penalties")
    print("✓ Enhanced hyperparameter ranges for Bayesian optimization")
    print("✓ Corrected R² metric to avoid overestimation")
    

if __name__ == "__main__":
    main()
