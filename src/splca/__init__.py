# ============================================================================
# splca/__init__.py
# ============================================================================

from .core import SPLCAOptimizer
from .layers import SPLCALinear, SPLCAConv2d
from .predictors import LinearPredictor, MLPPredictor
from .modulation import ValidationModulator, RewardModulator
from .models import TextClassifier, VisionClassifier, AudioClassifier

__version__ = "0.1.0"
__all__ = [
    "SPLCAOptimizer",
    "SPLCALinear", 
    "SPLCAConv2d",
    "LinearPredictor",
    "MLPPredictor",
    "ValidationModulator",
    "RewardModulator",
    "TextClassifier",
    "VisionClassifier", 
    "AudioClassifier",
]