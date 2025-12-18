"""
Dynamic Kirchhoff's Current Law (KCL) constraint module.
Implements adaptive thresholds based on statistical metrics.
"""

import numpy as np
import torch
import torch.nn as nn


class DynamicKCLConstraint:
    """
    Dynamic KCL constraint with adaptive thresholds.
    """
    
    def __init__(self, 
                 window_size=10, 
                 base_threshold=0.1,
                 std_multiplier=2.0,
                 soft_margin=0.05,
                 directional_penalty_weight=0.5):
        """
        Initialize dynamic KCL constraint.
        
        Args:
            window_size: Size of moving window for statistics
            base_threshold: Base threshold for KCL violations
            std_multiplier: Multiplier for standard deviation in threshold calculation
            soft_margin: Margin for soft violations
            directional_penalty_weight: Weight for directional penalties
        """
        self.window_size = window_size
        self.base_threshold = base_threshold
        self.std_multiplier = std_multiplier
        self.soft_margin = soft_margin
        self.directional_penalty_weight = directional_penalty_weight
        
        # Statistics buffers
        self.moving_average = None
        self.rolling_std = None
        
    def compute_adaptive_threshold(self, node_flows):
        """
        Compute adaptive threshold based on current data statistics.
        
        Args:
            node_flows: Tensor of shape (batch_size, n_nodes, n_timesteps)
            
        Returns:
            adaptive_threshold: Computed adaptive threshold
        """
        if isinstance(node_flows, torch.Tensor):
            node_flows_np = node_flows.detach().cpu().numpy()
        else:
            node_flows_np = node_flows
        
        # Compute moving average
        if node_flows_np.shape[-1] >= self.window_size:
            # Use convolution for efficient moving average
            kernel = np.ones(self.window_size) / self.window_size
            moving_avg = np.apply_along_axis(
                lambda x: np.convolve(x, kernel, mode='valid'), 
                axis=-1, 
                arr=node_flows_np
            )
            self.moving_average = np.mean(np.abs(moving_avg))
        else:
            self.moving_average = np.mean(np.abs(node_flows_np))
        
        # Compute rolling standard deviation
        if node_flows_np.shape[-1] >= self.window_size:
            rolling_std_values = []
            for i in range(node_flows_np.shape[-1] - self.window_size + 1):
                window = node_flows_np[..., i:i + self.window_size]
                rolling_std_values.append(np.std(window))
            self.rolling_std = np.mean(rolling_std_values)
        else:
            self.rolling_std = np.std(node_flows_np)
        
        # Adaptive threshold combines base threshold with data statistics
        adaptive_threshold = max(
            self.base_threshold,
            self.moving_average + self.std_multiplier * self.rolling_std
        )
        
        return adaptive_threshold
    
    def compute_kcl_violations(self, node_values, adjacency_matrix, edge_index=None):
        """
        Compute KCL violations at each node.
        KCL states that sum of flows into a node equals sum of flows out.
        
        Args:
            node_values: Node values (batch_size, n_nodes, n_features)
            adjacency_matrix: Adjacency matrix (n_nodes, n_nodes) or edge_index for sparse format
            edge_index: Optional edge index for graph (2, n_edges)
            
        Returns:
            kcl_violations: KCL violation values at each node
        """
        if isinstance(node_values, np.ndarray):
            node_values = torch.from_numpy(node_values).float()
        
        if edge_index is not None:
            # Sparse graph format
            batch_size, n_nodes, n_features = node_values.shape
            
            # Compute flow differences for each edge
            source_nodes = edge_index[0]
            target_nodes = edge_index[1]
            
            # Flow from source to target
            flow_diffs = node_values[:, source_nodes, :] - node_values[:, target_nodes, :]
            
            # Aggregate flows at each node
            node_inflow = torch.zeros(batch_size, n_nodes, n_features, device=node_values.device)
            node_outflow = torch.zeros(batch_size, n_nodes, n_features, device=node_values.device)
            
            for i, (src, tgt) in enumerate(zip(source_nodes, target_nodes)):
                node_outflow[:, src, :] += flow_diffs[:, i, :]
                node_inflow[:, tgt, :] += flow_diffs[:, i, :]
            
            # KCL: inflow should equal outflow
            kcl_violations = torch.abs(node_inflow - node_outflow)
            
        else:
            # Dense adjacency matrix format
            if isinstance(adjacency_matrix, np.ndarray):
                adjacency_matrix = torch.from_numpy(adjacency_matrix).float()
            
            batch_size, n_nodes, n_features = node_values.shape
            
            # Compute flows based on adjacency
            # Flow from i to j: (node_values[i] - node_values[j]) * adjacency[i, j]
            node_values_expanded = node_values.unsqueeze(2)  # (batch, n_nodes, 1, n_features)
            node_values_tiled = node_values.unsqueeze(1)     # (batch, 1, n_nodes, n_features)
            
            flow_matrix = (node_values_expanded - node_values_tiled)  # (batch, n_nodes, n_nodes, n_features)
            
            # Apply adjacency mask
            adjacency_expanded = adjacency_matrix.unsqueeze(0).unsqueeze(-1)  # (1, n_nodes, n_nodes, 1)
            flow_matrix = flow_matrix * adjacency_expanded
            
            # Sum inflows and outflows
            node_outflow = torch.sum(flow_matrix, dim=2)  # Sum over target nodes
            node_inflow = -torch.sum(flow_matrix, dim=1)  # Sum over source nodes (negative because flow direction)
            
            # KCL violations
            kcl_violations = torch.abs(node_inflow - node_outflow)
        
        return kcl_violations
    
    def compute_directional_penalties(self, node_values, edge_index, expected_directions=None):
        """
        Compute directional penalties for flow directions.
        
        Args:
            node_values: Node values (batch_size, n_nodes, n_features)
            edge_index: Edge indices (2, n_edges)
            expected_directions: Optional expected flow directions (+1 or -1)
            
        Returns:
            directional_penalty: Penalty for violating expected flow directions
        """
        if isinstance(node_values, np.ndarray):
            node_values = torch.from_numpy(node_values).float()
        
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
        
        # Compute actual flow directions
        flow_values = node_values[:, source_nodes, :] - node_values[:, target_nodes, :]
        actual_directions = torch.sign(flow_values)
        
        if expected_directions is not None:
            # Penalize flows in wrong direction
            direction_mismatch = torch.abs(actual_directions - expected_directions)
            directional_penalty = torch.mean(direction_mismatch)
        else:
            # Without expected directions, penalize inconsistent flows
            # (flows should be relatively consistent across time)
            flow_std = torch.std(actual_directions, dim=0)
            directional_penalty = torch.mean(flow_std)
        
        return directional_penalty
    
    def compute_kcl_loss(self, node_values, adjacency_matrix=None, edge_index=None, expected_directions=None):
        """
        Compute complete KCL loss with soft/hard violations and directional penalties.
        
        Args:
            node_values: Node values (batch_size, n_nodes, n_features)
            adjacency_matrix: Dense adjacency matrix or None
            edge_index: Sparse edge index or None
            expected_directions: Optional expected flow directions
            
        Returns:
            total_loss: Combined KCL loss
            loss_components: Dictionary with individual loss components
        """
        # Compute KCL violations
        kcl_violations = self.compute_kcl_violations(node_values, adjacency_matrix, edge_index)
        
        # Compute adaptive threshold
        adaptive_threshold = self.compute_adaptive_threshold(kcl_violations)
        
        # Soft violations (within margin)
        soft_violations = torch.relu(kcl_violations - adaptive_threshold + self.soft_margin)
        soft_loss = torch.mean(soft_violations)
        
        # Hard violations (beyond margin)
        hard_violations = torch.relu(kcl_violations - adaptive_threshold)
        hard_loss = torch.mean(hard_violations ** 2)  # Quadratic penalty for hard violations
        
        # Directional penalties
        if edge_index is not None:
            directional_loss = self.compute_directional_penalties(
                node_values, edge_index, expected_directions
            )
        else:
            directional_loss = torch.tensor(0.0, device=node_values.device)
        
        # Combined loss
        total_loss = (
            soft_loss + 
            2.0 * hard_loss +  # Weight hard violations more
            self.directional_penalty_weight * directional_loss
        )
        
        loss_components = {
            'soft_loss': soft_loss.item() if isinstance(soft_loss, torch.Tensor) else soft_loss,
            'hard_loss': hard_loss.item() if isinstance(hard_loss, torch.Tensor) else hard_loss,
            'directional_loss': directional_loss.item() if isinstance(directional_loss, torch.Tensor) else directional_loss,
            'adaptive_threshold': adaptive_threshold,
            'mean_violation': torch.mean(kcl_violations).item()
        }
        
        return total_loss, loss_components
