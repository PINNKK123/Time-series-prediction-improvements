"""
Training module for GNN-STAR with Bayesian hyperparameter optimization.
Enhanced training with improved noise injection, dropout, and KCL regularization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, Tuple
import optuna


class GNNSTARTrainer:
    """
    Trainer for GNN-STAR model with enhanced regularization and optimization.
    """
    
    def __init__(self,
                 model,
                 kcl_constraint,
                 device='cpu',
                 lr=0.001,
                 weight_decay=1e-5,
                 kcl_weight=1.0,
                 noise_level=0.01,
                 noise_decay=0.99):
        """
        Initialize trainer.
        
        Args:
            model: GNN-STAR model
            kcl_constraint: KCL constraint module
            device: Device to use
            lr: Learning rate (0.0001-0.01 recommended)
            weight_decay: Weight decay for regularization
            kcl_weight: Weight for KCL loss
            noise_level: Initial noise level for noise injection
            noise_decay: Decay rate for noise level
        """
        self.model = model.to(device)
        self.kcl_constraint = kcl_constraint
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.kcl_weight = kcl_weight
        self.noise_level = noise_level
        self.noise_decay = noise_decay
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.kcl_losses = []
        
    def inject_noise(self, x, current_epoch, total_epochs):
        """
        Inject adaptive noise for robustness.
        
        Args:
            x: Input tensor
            current_epoch: Current training epoch
            total_epochs: Total number of epochs
            
        Returns:
            x_noisy: Input with injected noise
        """
        # Decay noise level over training
        epoch_factor = (1.0 - current_epoch / total_epochs)
        current_noise_level = self.noise_level * (self.noise_decay ** current_epoch) * epoch_factor
        
        # Add Gaussian noise
        noise = torch.randn_like(x) * current_noise_level
        x_noisy = x + noise
        
        return x_noisy
    
    def compute_loss(self, predictions, targets, node_values, edge_index=None, adjacency_matrix=None):
        """
        Compute total loss including prediction loss and KCL regularization.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            node_values: Node values for KCL computation
            edge_index: Optional edge indices
            adjacency_matrix: Optional adjacency matrix
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with loss components
        """
        # Prediction loss (MSE)
        pred_loss = self.criterion(predictions, targets)
        
        # KCL regularization loss
        kcl_loss, kcl_components = self.kcl_constraint.compute_kcl_loss(
            node_values,
            adjacency_matrix=adjacency_matrix,
            edge_index=edge_index
        )
        
        # Total loss
        total_loss = pred_loss + self.kcl_weight * kcl_loss
        
        # Loss dictionary
        loss_dict = {
            'total_loss': total_loss.item(),
            'pred_loss': pred_loss.item(),
            'kcl_loss': kcl_loss.item(),
            **kcl_components
        }
        
        return total_loss, loss_dict
    
    def train_epoch(self, train_loader, epoch, total_epochs):
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            total_epochs: Total number of epochs
            
        Returns:
            avg_loss: Average loss for the epoch
            loss_components: Average loss components
        """
        self.model.train()
        total_loss = 0.0
        loss_components_sum = {}
        n_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Unpack batch
            x, y, edge_index, adjacency_matrix = batch
            x = x.to(self.device)
            y = y.to(self.device)
            
            if edge_index is not None:
                edge_index = edge_index.to(self.device)
            if adjacency_matrix is not None:
                adjacency_matrix = adjacency_matrix.to(self.device)
            
            # Inject noise for robustness
            x_noisy = self.inject_noise(x, epoch, total_epochs)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(x_noisy, edge_index)
            
            # Compute loss
            # Use predictions as node values for KCL (assuming predictions represent node states)
            node_values = predictions.squeeze(-1) if predictions.dim() > 3 else predictions
            loss, loss_dict = self.compute_loss(
                predictions, y, node_values, edge_index, adjacency_matrix
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            for key, value in loss_dict.items():
                if key not in loss_components_sum:
                    loss_components_sum[key] = 0.0
                loss_components_sum[key] += value
            
            n_batches += 1
        
        # Average losses
        avg_loss = total_loss / n_batches
        avg_components = {k: v / n_batches for k, v in loss_components_sum.items()}
        
        self.train_losses.append(avg_loss)
        self.kcl_losses.append(avg_components.get('kcl_loss', 0.0))
        
        return avg_loss, avg_components
    
    def validate(self, val_loader):
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            avg_loss: Average validation loss
            metrics: Validation metrics (MSE, MAE, R2)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x, y, edge_index, adjacency_matrix = batch
                x = x.to(self.device)
                y = y.to(self.device)
                
                if edge_index is not None:
                    edge_index = edge_index.to(self.device)
                if adjacency_matrix is not None:
                    adjacency_matrix = adjacency_matrix.to(self.device)
                
                # Forward pass
                predictions = self.model(x, edge_index)
                
                # Compute loss
                node_values = predictions.squeeze(-1) if predictions.dim() > 3 else predictions
                loss, _ = self.compute_loss(
                    predictions, y, node_values, edge_index, adjacency_matrix
                )
                
                total_loss += loss.item()
                all_predictions.append(predictions.cpu())
                all_targets.append(y.cpu())
                n_batches += 1
        
        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute metrics
        mse = torch.mean((all_predictions - all_targets) ** 2).item()
        mae = torch.mean(torch.abs(all_predictions - all_targets)).item()
        
        # R² score (corrected to avoid overestimation)
        ss_res = torch.sum((all_targets - all_predictions) ** 2)
        ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        r2 = r2.item()
        
        # Adjusted R² to penalize complexity
        n_samples = all_predictions.numel()
        n_features = all_predictions.shape[-1] if all_predictions.dim() > 1 else 1
        adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
        
        avg_loss = total_loss / n_batches
        self.val_losses.append(avg_loss)
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'adjusted_r2': adjusted_r2
        }
        
        return avg_loss, metrics
    
    def train(self, train_loader, val_loader, epochs, verbose=True):
        """
        Complete training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            verbose: Whether to print progress
            
        Returns:
            history: Training history dictionary
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }
        
        for epoch in range(epochs):
            # Train
            train_loss, train_components = self.train_epoch(train_loader, epoch, epochs)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            
            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_metrics'].append(val_metrics)
            
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  KCL Loss: {train_components.get('kcl_loss', 0.0):.4f}")
                print(f"  Val Metrics: MSE={val_metrics['mse']:.4f}, "
                      f"MAE={val_metrics['mae']:.4f}, "
                      f"R²={val_metrics['r2']:.4f}, "
                      f"Adj-R²={val_metrics['adjusted_r2']:.4f}")
        
        return history


def bayesian_hyperparameter_optimization(
    train_loader,
    val_loader,
    n_trials=50,
    device='cpu',
    verbose=True
):
    """
    Perform Bayesian hyperparameter optimization using Optuna.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        n_trials: Number of optimization trials
        device: Device to use
        verbose: Whether to print progress
        
    Returns:
        best_params: Best hyperparameters found
        study: Optuna study object
    """
    
    def objective(trial):
        # Hyperparameter search space (enhanced ranges)
        hidden_dim = trial.suggest_int('hidden_dim', 24, 128, step=8)
        num_gru_layers = trial.suggest_int('num_gru_layers', 1, 3)
        num_edge_conv_layers = trial.suggest_int('num_edge_conv_layers', 1, 3)
        num_attention_heads = trial.suggest_categorical('num_attention_heads', [2, 4, 8])
        k_neighbors = trial.suggest_int('k_neighbors', 3, 10)
        dropout = trial.suggest_float('dropout', 0.1, 0.4)
        lr = trial.suggest_float('lr', 0.0001, 0.01, log=True)
        kcl_weight = trial.suggest_float('kcl_weight', 0.1, 5.0)
        
        # Get input/output dimensions from first batch
        sample_batch = next(iter(train_loader))
        input_dim = sample_batch[0].shape[-1]
        output_dim = sample_batch[1].shape[-1]
        
        # Import here to avoid circular dependency
        from .gnn_star import GNNSTAR
        from ..utils.kcl_constraint import DynamicKCLConstraint
        
        # Create model
        model = GNNSTAR(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_gru_layers=num_gru_layers,
            num_edge_conv_layers=num_edge_conv_layers,
            num_attention_heads=num_attention_heads,
            k_neighbors=k_neighbors,
            dropout=dropout
        )
        
        # Create KCL constraint
        kcl_constraint = DynamicKCLConstraint()
        
        # Create trainer
        trainer = GNNSTARTrainer(
            model=model,
            kcl_constraint=kcl_constraint,
            device=device,
            lr=lr,
            kcl_weight=kcl_weight
        )
        
        # Train for a few epochs
        history = trainer.train(train_loader, val_loader, epochs=20, verbose=False)
        
        # Return validation loss (metric to minimize)
        final_val_loss = history['val_loss'][-1]
        
        return final_val_loss
    
    # Create Optuna study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)
    
    best_params = study.best_params
    
    if verbose:
        print(f"\nBest hyperparameters found:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print(f"\nBest validation loss: {study.best_value:.4f}")
    
    return best_params, study
