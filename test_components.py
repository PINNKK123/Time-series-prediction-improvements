"""
Basic tests for GNN-STAR model components.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from gnn_star import (
    GNNSTAR,
    GRUWithAttention,
    EdgeConvLayer,
    InSARDataPreprocessor,
    VMDDecomposer,
    DynamicKCLConstraint
)


def test_data_preprocessing():
    """Test data preprocessing module."""
    print("Testing data preprocessing...")
    
    # Create sample data
    data = np.random.randn(5, 3, 100)
    timestamps = np.linspace(0, 99, 100)
    
    # Create preprocessor
    preprocessor = InSARDataPreprocessor()
    
    # Preprocess
    processed_data, anomaly_mask = preprocessor.preprocess(data, timestamps)
    
    assert processed_data.shape == data.shape, "Shape mismatch in preprocessing"
    assert anomaly_mask.shape == data.shape, "Anomaly mask shape mismatch"
    
    print("✓ Data preprocessing test passed")


def test_vmd_decomposition():
    """Test VMD decomposition."""
    print("Testing VMD decomposition...")
    
    # Create sample signal
    t = np.linspace(0, 1, 500)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    
    # Create VMD decomposer
    vmd = VMDDecomposer(k_range=(3, 5), alpha_range=(1e2, 1e3))
    
    # Decompose
    u, u_hat, omega = vmd.decompose(signal, k=3, alpha=500)
    
    assert u.shape[0] == 3, "Wrong number of modes"
    assert u.shape[1] == len(signal), "Wrong signal length"
    
    print("✓ VMD decomposition test passed")


def test_kcl_constraint():
    """Test KCL constraint."""
    print("Testing KCL constraint...")
    
    # Create sample data
    node_values = torch.randn(2, 5, 10)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                               [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
    
    # Create KCL constraint
    kcl = DynamicKCLConstraint()
    
    # Compute loss
    loss, components = kcl.compute_kcl_loss(node_values, edge_index=edge_index)
    
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert 'soft_loss' in components, "Missing soft_loss component"
    assert 'hard_loss' in components, "Missing hard_loss component"
    
    print("✓ KCL constraint test passed")


def test_edge_conv_layer():
    """Test EdgeConv layer."""
    print("Testing EdgeConv layer...")
    
    # Create sample data
    x = torch.randn(10, 16)  # 10 nodes, 16 features
    
    # Create EdgeConv layer
    edge_conv = EdgeConvLayer(in_channels=16, out_channels=32, k=3)
    
    # Forward pass
    out = edge_conv(x)
    
    assert out.shape == (10, 32), "Wrong output shape"
    
    print("✓ EdgeConv layer test passed")


def test_gru_with_attention():
    """Test GRU with attention."""
    print("Testing GRU with attention...")
    
    # Create sample data
    x = torch.randn(4, 20, 16)  # batch=4, seq_len=20, input_dim=16
    
    # Create GRU with attention
    gru_attn = GRUWithAttention(input_dim=16, hidden_dim=32, num_layers=2, num_heads=4)
    
    # Forward pass
    output, hidden, attention_weights = gru_attn(x)
    
    assert output.shape == (4, 20, 32), "Wrong output shape"
    assert hidden.shape == (2, 4, 32), "Wrong hidden shape"
    
    print("✓ GRU with attention test passed")


def test_gnn_star_model():
    """Test GNN-STAR model."""
    print("Testing GNN-STAR model...")
    
    # Create sample data
    x = torch.randn(2, 5, 10, 3)  # batch=2, nodes=5, seq_len=10, features=3
    
    # Create model
    model = GNNSTAR(
        input_dim=3,
        hidden_dim=32,
        output_dim=1,
        num_gru_layers=2,
        num_edge_conv_layers=2,
        num_attention_heads=4,
        k_neighbors=3,
        dropout=0.2
    )
    
    # Forward pass
    output = model(x)
    
    assert output.shape == (2, 5, 10, 1), "Wrong output shape"
    
    print("✓ GNN-STAR model test passed")


def test_integration():
    """Test integration of all components."""
    print("Testing integration...")
    
    # Create synthetic data
    n_samples = 10
    n_nodes = 5
    seq_len = 20
    n_features = 3
    
    data = np.random.randn(n_samples, n_nodes, seq_len, n_features)
    
    # Preprocessing (on a single sample)
    timestamps = np.linspace(0, seq_len - 1, seq_len)
    preprocessor = InSARDataPreprocessor()
    
    sample_2d = data[0].transpose(0, 2, 1)  # (nodes, seq_len, features) -> (nodes, features, seq_len)
    processed_sample, _ = preprocessor.preprocess(sample_2d, timestamps)
    
    # Create model
    model = GNNSTAR(
        input_dim=n_features,
        hidden_dim=32,
        output_dim=1,
        num_gru_layers=1,
        num_edge_conv_layers=1
    )
    
    # Forward pass
    x_tensor = torch.FloatTensor(data)
    output = model(x_tensor)
    
    assert output.shape == (n_samples, n_nodes, seq_len, 1), "Wrong output shape"
    
    # KCL constraint
    kcl = DynamicKCLConstraint()
    node_values = output.squeeze(-1)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                               [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
    
    loss, components = kcl.compute_kcl_loss(node_values, edge_index=edge_index)
    
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    
    print("✓ Integration test passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running GNN-STAR Component Tests")
    print("=" * 60 + "\n")
    
    try:
        test_data_preprocessing()
        test_vmd_decomposition()
        test_kcl_constraint()
        test_edge_conv_layer()
        test_gru_with_attention()
        test_gnn_star_model()
        test_integration()
        
        print("\n" + "=" * 60)
        print("All Tests Passed Successfully! ✓")
        print("=" * 60)
        return True
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
