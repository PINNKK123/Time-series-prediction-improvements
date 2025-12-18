"""
GNN-STAR model architecture with EdgeConv and GRU with self-attention.
Enhanced model for time series prediction using graph neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class EdgeConvLayer(MessagePassing):
    """
    Edge Convolution layer for dynamic graph construction.
    Dynamically constructs edges based on feature similarity.
    """
    
    def __init__(self, in_channels, out_channels, k=5, aggr='max'):
        """
        Initialize EdgeConv layer.
        
        Args:
            in_channels: Number of input features
            out_channels: Number of output features
            k: Number of nearest neighbors for dynamic edge construction
            aggr: Aggregation method ('max', 'mean', 'add')
        """
        super(EdgeConvLayer, self).__init__(aggr=aggr)
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
    def forward(self, x, edge_index=None, batch=None):
        """
        Forward pass with dynamic edge construction.
        
        Args:
            x: Node features (n_nodes, in_channels)
            edge_index: Optional existing edge index
            batch: Optional batch assignment
            
        Returns:
            out: Updated node features
        """
        if edge_index is None:
            # Construct k-NN graph dynamically
            edge_index = self.knn_graph(x, self.k, batch)
        
        return self.propagate(edge_index, x=x)
    
    def message(self, x_i, x_j):
        """
        Construct messages from neighboring nodes.
        
        Args:
            x_i: Features of target nodes
            x_j: Features of source nodes
            
        Returns:
            messages: Constructed messages
        """
        # Concatenate node features with neighbor features
        edge_features = torch.cat([x_i, x_j - x_i], dim=-1)
        return self.mlp(edge_features)
    
    def knn_graph(self, x, k, batch=None):
        """
        Construct k-NN graph based on feature distance.
        
        Args:
            x: Node features (n_nodes, in_channels)
            k: Number of nearest neighbors
            batch: Batch assignment
            
        Returns:
            edge_index: Constructed edge indices
        """
        n_nodes = x.size(0)
        
        # Compute pairwise distances
        dist_matrix = torch.cdist(x, x, p=2)
        
        # Find k nearest neighbors for each node
        _, indices = torch.topk(dist_matrix, k=k + 1, dim=-1, largest=False)
        indices = indices[:, 1:]  # Exclude self
        
        # Construct edge index
        source = torch.arange(n_nodes, device=x.device).view(-1, 1).repeat(1, k).view(-1)
        target = indices.view(-1)
        
        edge_index = torch.stack([source, target], dim=0)
        
        return edge_index


class SelfAttention(nn.Module):
    """
    Self-attention mechanism for temporal modeling.
    """
    
    def __init__(self, hidden_dim, num_heads=4):
        """
        Initialize self-attention layer.
        
        Args:
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
        """
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, mask=None):
        """
        Forward pass with multi-head attention.
        
        Args:
            x: Input tensor (batch_size, seq_len, hidden_dim)
            mask: Optional attention mask
            
        Returns:
            out: Attention output
            attention_weights: Attention weights for visualization
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )
        
        # Output projection
        out = self.output(attention_output)
        
        return out, attention_weights


class GRUWithAttention(nn.Module):
    """
    GRU with self-attention mechanism for enhanced temporal modeling.
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers=2, num_heads=4, dropout=0.2):
        """
        Initialize GRU with attention.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            num_layers: Number of GRU layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(GRUWithAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = SelfAttention(hidden_dim, num_heads)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden=None):
        """
        Forward pass.
        
        Args:
            x: Input sequence (batch_size, seq_len, input_dim)
            hidden: Initial hidden state
            
        Returns:
            output: Output sequence
            hidden: Final hidden state
            attention_weights: Attention weights
        """
        # GRU forward pass
        gru_out, hidden = self.gru(x, hidden)
        
        # Self-attention
        attn_out, attention_weights = self.attention(gru_out)
        
        # Residual connection and layer normalization
        output = self.layer_norm(gru_out + self.dropout(attn_out))
        
        return output, hidden, attention_weights


class GNNSTAR(nn.Module):
    """
    GNN-STAR model for time series prediction with KCL constraints.
    """
    
    def __init__(self, 
                 input_dim,
                 hidden_dim=64,
                 output_dim=1,
                 num_gru_layers=2,
                 num_edge_conv_layers=2,
                 num_attention_heads=4,
                 k_neighbors=5,
                 dropout=0.2,
                 use_edge_conv=True):
        """
        Initialize GNN-STAR model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension (24-128 range recommended)
            output_dim: Output dimension
            num_gru_layers: Number of GRU layers
            num_edge_conv_layers: Number of EdgeConv layers
            num_attention_heads: Number of attention heads
            k_neighbors: Number of neighbors for EdgeConv
            dropout: Dropout rate (0.1-0.4 recommended)
            use_edge_conv: Whether to use EdgeConv layers
        """
        super(GNNSTAR, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_edge_conv = use_edge_conv
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # EdgeConv layers for spatial modeling
        if use_edge_conv:
            self.edge_conv_layers = nn.ModuleList([
                EdgeConvLayer(hidden_dim, hidden_dim, k=k_neighbors)
                for _ in range(num_edge_conv_layers)
            ])
            self.edge_conv_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim)
                for _ in range(num_edge_conv_layers)
            ])
        
        # GRU with attention for temporal modeling
        self.gru_attention = GRUWithAttention(
            hidden_dim, 
            hidden_dim, 
            num_layers=num_gru_layers,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index=None, return_attention=False):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, n_nodes, seq_len, input_dim)
            edge_index: Optional edge indices for graph
            return_attention: Whether to return attention weights
            
        Returns:
            output: Predictions (batch_size, n_nodes, seq_len, output_dim)
            attention_weights: Optional attention weights
        """
        batch_size, n_nodes, seq_len, _ = x.size()
        
        # Input projection
        x_proj = self.input_proj(x)  # (batch, n_nodes, seq_len, hidden_dim)
        
        # Process each time step with spatial GNN
        if self.use_edge_conv:
            spatial_features = []
            for t in range(seq_len):
                x_t = x_proj[:, :, t, :]  # (batch, n_nodes, hidden_dim)
                
                # Flatten batch and nodes for EdgeConv
                x_t_flat = x_t.reshape(batch_size * n_nodes, self.hidden_dim)
                
                # Apply EdgeConv layers
                for edge_conv, norm in zip(self.edge_conv_layers, self.edge_conv_norms):
                    x_t_flat_new = edge_conv(x_t_flat, edge_index)
                    x_t_flat = norm(x_t_flat + self.dropout(x_t_flat_new))
                
                # Reshape back
                x_t = x_t_flat.reshape(batch_size, n_nodes, self.hidden_dim)
                spatial_features.append(x_t)
            
            # Stack temporal dimension
            x_spatial = torch.stack(spatial_features, dim=2)  # (batch, n_nodes, seq_len, hidden_dim)
        else:
            x_spatial = x_proj
        
        # Process each node's temporal sequence with GRU+Attention
        temporal_features = []
        all_attention_weights = []
        
        for n in range(n_nodes):
            x_n = x_spatial[:, n, :, :]  # (batch, seq_len, hidden_dim)
            out_n, _, attn_weights = self.gru_attention(x_n)
            temporal_features.append(out_n)
            if return_attention:
                all_attention_weights.append(attn_weights)
        
        # Stack nodes
        x_temporal = torch.stack(temporal_features, dim=1)  # (batch, n_nodes, seq_len, hidden_dim)
        
        # Output projection
        output = self.output_proj(x_temporal)  # (batch, n_nodes, seq_len, output_dim)
        
        if return_attention:
            return output, all_attention_weights
        else:
            return output
