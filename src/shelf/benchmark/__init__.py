"""
Benchmark generation and evaluation tools.

Two approaches available:
1. Government-focused (archetypes) - based on CGP MARC record analysis
2. Universal LC-based (lc_generator) - pure LC taxonomy sampling
"""

# Government publication archetypes (from CGP analysis)
from .archetypes import ARCHETYPES, Archetype
from .generator import BenchmarkGenerator

# Universal LC-based generator (pure LC taxonomies)
from .lc_taxonomy import (
    LCCClass,
    LCC_NAMES,
    LCGFTCategory,
    LCGFT_CHILDREN,
    LCSH_TOPICS_BY_LCC,
    LCDGT_GROUPS,
    BenchmarkProfile,
    get_taxonomy_stats,
)
from .lc_generator import (
    LCDocument,
    LCGeneratorConfig,
    LCBenchmarkGenerator,
)

__all__ = [
    # Government archetypes
    "ARCHETYPES",
    "Archetype",
    "BenchmarkGenerator",
    # LC taxonomy
    "LCCClass",
    "LCC_NAMES",
    "LCGFTCategory",
    "LCGFT_CHILDREN",
    "LCSH_TOPICS_BY_LCC",
    "LCDGT_GROUPS",
    "BenchmarkProfile",
    "get_taxonomy_stats",
    # LC generator
    "LCDocument",
    "LCGeneratorConfig",
    "LCBenchmarkGenerator",
]
