"""
Configuration file for GNN-STAR model hyperparameters.
"""

# Model Architecture
MODEL_CONFIG = {
    'input_dim': 3,
    'hidden_dim': 64,  # Range: [24, 128]
    'output_dim': 1,
    'num_gru_layers': 2,  # Range: [1, 3]
    'num_edge_conv_layers': 2,  # Range: [1, 3]
    'num_attention_heads': 4,  # Options: [2, 4, 8]
    'k_neighbors': 5,  # Range: [3, 10]
    'dropout': 0.2,  # Range: [0.1, 0.4]
    'use_edge_conv': True
}

# Training Configuration
TRAINING_CONFIG = {
    'lr': 0.001,  # Range: [0.0001, 0.01]
    'weight_decay': 1e-5,
    'kcl_weight': 1.0,  # Range: [0.1, 5.0]
    'noise_level': 0.01,
    'noise_decay': 0.99,
    'batch_size': 16,
    'epochs': 50,
    'gradient_clip_norm': 1.0
}

# Preprocessing Configuration
PREPROCESSING_CONFIG = {
    'anomaly_threshold': 3.0,  # Z-score threshold
    'gaussian_sigma': 1.5,
    'boundary_padding': 2
}

# VMD Configuration
VMD_CONFIG = {
    'k_range': (3, 8),  # Number of modes
    'alpha_range': (1e2, 1e4),  # Balancing parameter
    'tau': 0.0,
    'DC': 0,
    'init': 1,
    'tol': 1e-7,
    'n_optimization_trials': 20
}

# KCL Constraint Configuration
KCL_CONFIG = {
    'window_size': 10,
    'base_threshold': 0.1,
    'std_multiplier': 2.0,
    'soft_margin': 0.05,
    'directional_penalty_weight': 0.5
}

# Bayesian Optimization Configuration
BAYESIAN_OPT_CONFIG = {
    'n_trials': 50,
    'search_space': {
        'hidden_dim': {'type': 'int', 'low': 24, 'high': 128, 'step': 8},
        'num_gru_layers': {'type': 'int', 'low': 1, 'high': 3},
        'num_edge_conv_layers': {'type': 'int', 'low': 1, 'high': 3},
        'num_attention_heads': {'type': 'categorical', 'choices': [2, 4, 8]},
        'k_neighbors': {'type': 'int', 'low': 3, 'high': 10},
        'dropout': {'type': 'float', 'low': 0.1, 'high': 0.4},
        'lr': {'type': 'float', 'low': 0.0001, 'high': 0.01, 'log': True},
        'kcl_weight': {'type': 'float', 'low': 0.1, 'high': 5.0}
    }
}

# Device Configuration
DEVICE_CONFIG = {
    'use_cuda': True,  # Use CUDA if available
    'device': 'cuda' if True else 'cpu'
}
