# SPLCA: Self-Predictive Local Credit Assignment

A PyTorch library implementing biologically-plausible learning with local prediction error, eligibility traces, and global modulation.

## Installation

```bash
pip install splca
```

Or install from source:

```bash
git clone https://github.com/yourusername/splca.git
cd splca
pip install -e .
```

## Quick Start

### Vision (MNIST)

```python
from splca import VisionClassifier, SPLCAOptimizer, ValidationModulator
from splca.utils import train_splca_model

model = VisionClassifier(in_channels=1, num_classes=10)
optimizer = SPLCAOptimizer(model.parameters(), lr=1e-3, gamma=0.95)
modulator = ValidationModulator(alpha=1.0)

history = train_splca_model(
    model, train_loader, val_loader, 
    optimizer, modulator, epochs=20
)
```

### Text Classification

```python
from splca import TextClassifier, SPLCAOptimizer, ValidationModulator

model = TextClassifier(
    vocab_size=5000, 
    embed_dim=128, 
    hidden_dim=256, 
    num_classes=2
)
optimizer = SPLCAOptimizer(model.parameters())
modulator = ValidationModulator()

history = train_splca_model(
    model, train_loader, val_loader,
    optimizer, modulator, epochs=15
)
```

### Audio Classification

```python
from splca import AudioClassifier, SPLCAOptimizer, ValidationModulator

model = AudioClassifier(input_dim=128, num_classes=10)
optimizer = SPLCAOptimizer(model.parameters())
modulator = ValidationModulator()

history = train_splca_model(
    model, train_loader, val_loader,
    optimizer, modulator, epochs=15
)
```

## How It Works

SPLCA replaces backpropagation with local learning rules:

1. **Local Prediction**: Each layer predicts its next activation
2. **Prediction Error**: e = y(t+1) - ŷ(t+1)
3. **Eligibility Traces**: E(t+1) = γE(t) + x
4. **Global Modulation**: Scalar m(t) from validation improvement
5. **Update Rule**: Δw = -η·m·e·E - η_heb·Hebbian - η_s·decay

## Key Features

- ✅ **Biologically plausible** - local updates, no backprop
- ✅ **Temporal credit assignment** - eligibility traces handle delays
- ✅ **Multi-modal** - supports text, vision, audio
- ✅ **Easy integration** - drop-in replacement for standard layers
- ✅ **Modular design** - swap predictors, modulators, layers

## Examples

Run the demos:

```bash
python examples/mnist_demo.py
python examples/text_classification_demo.py
python examples/audio_classification_demo.py
```

## Architecture

```
splca/
├── core.py          # SPLCAOptimizer, EligibilityTrace
├── layers.py        # SPLCALinear, SPLCAConv2d
├── predictors.py    # LinearPredictor, MLPPredictor
├── modulation.py    # ValidationModulator, RewardModulator
├── models/
│   ├── text.py      # TextClassifier
│   ├── vision.py    # VisionClassifier
│   └── audio.py     # AudioClassifier
└── utils.py         # Training utilities
```

## Hyperparameters

- `lr` (η): Main learning rate (1e-3 to 1e-4)
- `gamma` (γ): Eligibility trace decay (0.9-0.99)
- `eta_pred`: Predictor learning rate (1e-3)
- `eta_heb`: Hebbian term coefficient (1e-4)
- `eta_stab`: Weight decay coefficient (1e-5)

## Citation

If you use this library, please cite:

```bibtex
@software{splca2024,
  title={SPLCA: Self-Predictive Local Credit Assignment},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/splca}
}
```

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or PR.

<!-- # ============================================================================
# INSTALLATION INSTRUCTIONS
# ============================================================================
"""
To create the pip-installable package:

1. Create directory structure:
   mkdir -p splca/splca/models examples tests
   
2. Copy all code sections above into respective files

3. Install in development mode:
   cd splca
   pip install -e .

4. Run demos:
   python examples/mnist_demo.py
   python examples/text_classification_demo.py
   python examples/audio_classification_demo.py

5. To publish to PyPI:
   pip install build twine
   python -m build
   twine upload dist/*
""" -->
