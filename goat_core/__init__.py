"""
G.O.A.T Core AI Engine

This module provides the foundational AI processing capabilities for the G.O.A.T system.
It includes neural processing, decision-making, semantic analysis, and knowledge graph
management.
"""

__version__ = "3.1.0"
__author__ = "G.O.A.T Development Team"

from .neural_processor import NeuralProcessor
from .decision_engine import DecisionEngine
from .semantic_analyzer import SemanticAnalyzer
from .context_graph import ContextGraph
from .metrics_collector import MetricsCollector

__all__ = [
    'NeuralProcessor',
    'DecisionEngine',
    'SemanticAnalyzer',
    'ContextGraph',
    'MetricsCollector'
]

# Core configuration
CORE_CONFIG = {
    'max_context_length': 8192,
    'embedding_dimension': 768,
    'decision_threshold': 0.75,
    'graph_max_depth': 5,
    'metrics_buffer_size': 10000
}
