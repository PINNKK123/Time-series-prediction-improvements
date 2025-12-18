# Time-series-prediction-improvements

利用KCL约束和优化增强GNN-STAR模型以进行InSAR数据时间序列预测

Enhanced GNN-STAR Model for InSAR Time Series Prediction with KCL Constraints

## 🚀 Key Features

- **Enhanced Data Preprocessing**: Anomaly removal, Gaussian filtering, causal interpolation
- **VMD with Extended Parameters**: k ∈ [3, 8], alpha ∈ [10², 10⁴]
- **Dynamic KCL Constraints**: Adaptive thresholds with soft/hard violations
- **Advanced Architecture**: EdgeConv + GRU with self-attention
- **Bayesian Optimization**: Comprehensive hyperparameter tuning
- **Corrected Metrics**: Adjusted R² to avoid overestimation

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🎯 Quick Start

```python
from gnn_star import GNNSTAR, GNNSTARTrainer, DynamicKCLConstraint

# Create and train model
model = GNNSTAR(input_dim=3, hidden_dim=64, output_dim=1)
trainer = GNNSTARTrainer(model, DynamicKCLConstraint())
history = trainer.train(train_loader, val_loader, epochs=50)
```

## 📖 Documentation

See [DOCUMENTATION.md](DOCUMENTATION.md) for detailed information.

## 🔬 Example

```bash
python example_usage.py
```

## 📊 Improvements

✓ Dynamic KCL thresholds  
✓ Extended VMD parameter space  
✓ EdgeConv for dynamic graphs  
✓ GRU with multi-head attention  
✓ Enhanced regularization  
✓ Bayesian hyperparameter optimization  
✓ Corrected R² metric  

## 📝 License

MIT License
