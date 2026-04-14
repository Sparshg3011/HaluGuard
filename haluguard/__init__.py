"""
HaluGuard: Execution-Grounded Contrastive Context Selection
for Hallucination-Free Repository-Level Code Generation.

Benchmark: RepoBench v1.1 (``tianyang/repobench_python_v1.1``, split ``cross_file_first``)
"""

__version__ = "0.1.0"

from haluguard.hccs import HallucinationType, HCCSScorer
from haluguard.models import (
    DualEncoder,
    DualEncoderDeep,
    ListwiseMLP,
    PairwiseMLP,
    InteractionMLP,
    BilinearScorer,
    EnsembleScorer,
    MODEL_REGISTRY,
    build_model,
    count_parameters,
)

__all__ = [
    # Shared enum
    "HallucinationType",
    # Legacy scorer (kept for checkpoint compatibility)
    "HCCSScorer",
    # All scorer architectures
    "DualEncoder",
    "DualEncoderDeep",
    "ListwiseMLP",
    "PairwiseMLP",
    "InteractionMLP",
    "BilinearScorer",
    "EnsembleScorer",
    # Registry helpers
    "MODEL_REGISTRY",
    "build_model",
    "count_parameters",
]
