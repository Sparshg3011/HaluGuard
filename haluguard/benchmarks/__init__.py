"""
benchmarks — Unified adapters for code-generation benchmarks.

Each submodule exposes a :class:`BenchmarkLoader` that yields
:class:`Example` instances, independent of the underlying dataset format.

Available loaders:
    - :class:`~haluguard.benchmarks.repobench.RepoBenchLoader`
    - :class:`~haluguard.benchmarks.crosscodeeval.CrossCodeEvalLoader`
"""

from haluguard.benchmarks.base import BenchmarkLoader, Example

__all__ = ["BenchmarkLoader", "Example"]
