# GNN-STAR Model for Time Series Prediction

## Overview

This repository implements an enhanced GNN-STAR (Graph Neural Network with Spatio-Temporal Attention and Rolling prediction) model for InSAR time series prediction with Kirchhoff's Current Law (KCL) constraints. The implementation addresses previous performance issues through systematic improvements.

## Key Improvements

### 1. Data Quality and Preprocessing
- **Anomaly Detection & Removal**: Z-score based anomaly detection with configurable thresholds
- **Gaussian Filtering**: Smooth time series data to reduce noise
- **Enhanced Causal Interpolation**: Boundary corrections using extrapolation for improved edge handling
- **Implementation**: `gnn_star/preprocessing/data_preprocessing.py`

### 2. VMD Parameter Optimization
- **Extended Parameter Space**: 
  - k (number of modes) ∈ [3, 8]
  - alpha (balancing parameter) ∈ [10², 10⁴]
- **Bayesian Optimization**: Grid search over expanded parameter space
- **Quality Metrics**: Decomposition quality assessment with visualization
- **Implementation**: `gnn_star/preprocessing/vmd_decomposition.py`

### 3. Dynamic KCL Threshold
- **Adaptive Thresholds**: Based on moving average and rolling standard deviation
- **Statistical Metrics**: Real-time threshold adjustment
- **Violation Types**: Soft and hard violations with different penalties
- **Directional Penalties**: Enforce expected flow directions
- **Implementation**: `gnn_star/utils/kcl_constraint.py`

### 4. Model Architecture Enhancements
- **EdgeConv Layer**: Dynamic edge construction based on k-nearest neighbors
- **GRU with Self-Attention**: Multi-head attention mechanism for temporal dependencies
- **Expanded Hidden Dimensions**: Hidden dim ∈ [24, 128] for better capacity
- **Dropout Range**: Improved dropout ∈ [0.1, 0.4] for regularization
- **Implementation**: `gnn_star/models/gnn_star.py`

### 5. Regularization and Training
- **Enhanced KCL Regularization**: 
  - Soft violations with margin-based penalties
  - Hard violations with quadratic penalties
  - Directional penalties for flow consistency
- **Improved Noise Injection**: Adaptive noise with decay over training
- **Gradient Clipping**: Stability during training
- **Implementation**: `gnn_star/models/trainer.py`

### 6. Hyperparameter Optimization
- **Enhanced Search Spaces**:
  - hidden_dim ∈ [24, 128]
  - lr ∈ [0.0001, 0.01]
  - dropout ∈ [0.1, 0.4]
  - kcl_weight ∈ [0.1, 5.0]
- **Bayesian Optimization**: Using Optuna for efficient hyperparameter search
- **Implementation**: `gnn_star/models/trainer.py`

### 7. Corrected R² Metric
- **Issue Fixed**: Previous R² calculation could be overestimated
- **Solution**: Implemented adjusted R² to account for model complexity
- **Formula**: Adjusted R² = 1 - (1 - R²) × (n - 1) / (n - p - 1)

## Installation

### Requirements
```bash
pip install -r requirements.txt
```

### Dependencies
- PyTorch >= 2.0.0
- PyTorch Geometric >= 2.3.0
- NumPy >= 1.24.0
- SciPy >= 1.10.0
- scikit-learn >= 1.2.0
- Optuna >= 3.1.0
- Matplotlib >= 3.7.0
- Pandas >= 2.0.0
- vmdpy >= 0.2

## Usage

### Quick Start

```python
from gnn_star import GNNSTAR, GNNSTARTrainer, DynamicKCLConstraint

# Create model
model = GNNSTAR(
    input_dim=3,
    hidden_dim=64,
    output_dim=1,
    num_gru_layers=2,
    num_edge_conv_layers=2,
    num_attention_heads=4,
    k_neighbors=5,
    dropout=0.2
)

# Create KCL constraint
kcl_constraint = DynamicKCLConstraint()

# Create trainer
trainer = GNNSTARTrainer(
    model=model,
    kcl_constraint=kcl_constraint,
    device='cuda',
    lr=0.001,
    kcl_weight=1.0
)

# Train
history = trainer.train(train_loader, val_loader, epochs=50)
```

### Data Preprocessing

```python
from gnn_star import InSARDataPreprocessor

# Create preprocessor
preprocessor = InSARDataPreprocessor(
    anomaly_threshold=3.0,
    gaussian_sigma=1.5,
    boundary_padding=2
)

# Preprocess data
processed_data, anomaly_mask = preprocessor.preprocess(
    data, 
    timestamps
)
```

### VMD Decomposition

```python
from gnn_star import VMDDecomposer

# Create VMD decomposer
vmd = VMDDecomposer(
    k_range=(3, 8),
    alpha_range=(1e2, 1e4)
)

# Optimize parameters
best_params, quality = vmd.optimize_parameters(signal, n_trials=20)

# Decompose signal
u, u_hat, omega = vmd.decompose(signal)

# Visualize
fig = vmd.visualize_decomposition(signal, u, omega, save_path='vmd_result.png')
```

### Bayesian Hyperparameter Optimization

```python
from gnn_star import bayesian_hyperparameter_optimization

# Run optimization
best_params, study = bayesian_hyperparameter_optimization(
    train_loader=train_loader,
    val_loader=val_loader,
    n_trials=50,
    device='cuda',
    verbose=True
)

print(f"Best parameters: {best_params}")
```

### Complete Example

Run the complete demonstration:

```bash
python example_usage.py
```

This will:
1. Demonstrate data preprocessing
2. Show VMD decomposition
3. Display KCL constraint computation
4. Train the model
5. Generate training history plots
6. (Optional) Run Bayesian optimization

## Project Structure

```
Time-series-prediction-improvements/
├── gnn_star/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gnn_star.py          # Model architecture
│   │   └── trainer.py           # Training loop & optimization
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── data_preprocessing.py # Data cleaning & interpolation
│   │   └── vmd_decomposition.py  # VMD implementation
│   ├── utils/
│   │   ├── __init__.py
│   │   └── kcl_constraint.py    # KCL constraint module
│   └── optimization/
│       └── __init__.py
├── example_usage.py             # Complete demonstration
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Model Architecture

### EdgeConv Layer
- Dynamically constructs k-NN graph based on feature similarity
- Message passing with edge features
- Aggregation methods: max, mean, add

### GRU with Self-Attention
- Multi-head attention mechanism
- Residual connections
- Layer normalization
- Enhanced temporal modeling

### GNN-STAR Pipeline
1. Input projection
2. Spatial modeling with EdgeConv layers
3. Temporal modeling with GRU + Attention
4. Output projection

## KCL Constraint

### Dynamic Threshold Computation
```python
adaptive_threshold = max(
    base_threshold,
    moving_average + std_multiplier * rolling_std
)
```

### Loss Components
- **Soft Violations**: Violations within margin, linear penalty
- **Hard Violations**: Violations beyond margin, quadratic penalty
- **Directional Penalties**: Enforce flow direction consistency

### Total KCL Loss
```python
total_loss = soft_loss + 2.0 * hard_loss + w_dir * directional_loss
```

## Training Features

### Adaptive Noise Injection
- Noise level decays over training
- Epoch-based decay factor
- Improves model robustness

### Gradient Clipping
- Max norm of 1.0
- Prevents gradient explosion
- Improves training stability

### Validation Metrics
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score
- Adjusted R² Score (corrected metric)

## Hyperparameter Ranges

| Parameter | Range | Default |
|-----------|-------|---------|
| hidden_dim | [24, 128] | 64 |
| num_gru_layers | [1, 3] | 2 |
| num_edge_conv_layers | [1, 3] | 2 |
| num_attention_heads | {2, 4, 8} | 4 |
| k_neighbors | [3, 10] | 5 |
| dropout | [0.1, 0.4] | 0.2 |
| lr | [0.0001, 0.01] | 0.001 |
| kcl_weight | [0.1, 5.0] | 1.0 |

## Performance Improvements

### Before
- Static KCL threshold not adaptive to data variability
- Narrow VMD parameter space
- Overestimated R² metric
- Limited model capacity
- Basic regularization

### After
- ✓ Dynamic KCL thresholds with statistical adaptation
- ✓ Extended VMD parameter space with optimization
- ✓ Corrected R² metric (adjusted R²)
- ✓ Enhanced model architecture with EdgeConv and attention
- ✓ Improved regularization with soft/hard violations
- ✓ Comprehensive hyperparameter optimization

## Citation

If you use this code in your research, please cite:

```bibtex
@software{gnn_star_2025,
  title={GNN-STAR: Enhanced Time Series Prediction with KCL Constraints},
  author={PINNKK123},
  year={2025},
  url={https://github.com/PINNKK123/Time-series-prediction-improvements}
}
```

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Issues

If you encounter any problems or have suggestions, please open an issue on GitHub.
