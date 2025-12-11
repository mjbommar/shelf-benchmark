"""
Composable document sampler for LC taxonomy benchmarks.

Example usage:

    from shelf.sampler import DocumentSampler, DocumentGenerator

    # Quick start - sample with defaults
    sampler = DocumentSampler(seed=42)
    doc = sampler.sample()

    # Generate text with OpenAI gpt-5.1
    generator = DocumentGenerator(seed=42)
    generated = generator.generate(doc)
    print(generated.title)
    print(generated.body)

    # Fluent configuration
    sampler = (
        DocumentSampler()
        .with_seed(42)
        .with_lcc_classes(["H", "K", "P"])  # Social sciences, Law, Literature
        .with_lcgft_categories(["Informational works", "Literature"])
        .with_topics_per_doc(2, 4)
        .with_audience(True)
        .with_geographic(True)
    )

    # Save artifacts
    from shelf.sampler import ArtifactStore, export_dataset

    store = ArtifactStore("./output", model="gpt-5.1")
    store.save(generated)
"""

from .dimensions import (
    Sampler,
    LCCSampler,
    LCGFTSampler,
    TopicSampler,
    AudienceSampler,
    GeographicSampler,
    # Real LC data samplers with URIs
    LCTerm,
    RealLCGFTSampler,
    RealLCSHSampler,
    RealLCDGTSampler,
)
from .lc_data import load_lc_data, LCDataLoader
from .document import DocumentSampler, Document
from .generator import (
    DocumentGenerator,
    GeneratedDocument,
    GeneratedContent,
    TemplateGenerator,
    LengthSampler,
    RegisterSampler,
    SamplingParams,
    SamplingParamsSampler,
    DocumentLength,
    Register,
    LENGTH_WORD_RANGES,
    REGISTER_DESCRIPTIONS,
    GENERATION_INSTRUCTIONS,
    DEFAULT_MODEL,
    DEFAULT_SERVICE_TIER,
    TEMPERATURE_RANGE,
    TOP_P_RANGE,
    build_generation_prompt,
)
from .artifacts import (
    BenchmarkArtifact,
    ArtifactStore,
    DatasetMetadata,
    LCCLabel,
    LCGFTLabel,
    GenerationParams,
    GenerationMetadata,
    export_dataset,
)

__all__ = [
    # Base
    "Sampler",
    # Dimension samplers (hardcoded data)
    "LCCSampler",
    "LCGFTSampler",
    "TopicSampler",
    "AudienceSampler",
    "GeographicSampler",
    # Real LC data samplers (with URIs from id.loc.gov)
    "LCTerm",
    "RealLCGFTSampler",
    "RealLCSHSampler",
    "RealLCDGTSampler",
    "load_lc_data",
    "LCDataLoader",
    # Document sampler
    "DocumentSampler",
    "Document",
    # Text generation
    "DocumentGenerator",
    "GeneratedDocument",
    "GeneratedContent",
    "TemplateGenerator",
    "LengthSampler",
    "RegisterSampler",
    "DocumentLength",
    "Register",
    "LENGTH_WORD_RANGES",
    "REGISTER_DESCRIPTIONS",
    "GENERATION_INSTRUCTIONS",
    "DEFAULT_MODEL",
    "DEFAULT_SERVICE_TIER",
    "build_generation_prompt",
    # Artifacts
    "BenchmarkArtifact",
    "ArtifactStore",
    "DatasetMetadata",
    "LCCLabel",
    "LCGFTLabel",
    "GenerationParams",
    "GenerationMetadata",
    "export_dataset",
]
