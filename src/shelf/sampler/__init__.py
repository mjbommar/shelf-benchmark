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

from .artifacts import (
    ArtifactStore,
    BenchmarkArtifact,
    DatasetMetadata,
    GenerationMetadata,
    GenerationParams,
    LCCLabel,
    LCGFTLabel,
    export_dataset,
)
from .dimensions import (
    AudienceSampler,
    GeographicSampler,
    LCCSampler,
    LCGFTSampler,
    # Real LC data samplers with URIs
    LCTerm,
    RealLCDGTSampler,
    RealLCGFTSampler,
    RealLCSHSampler,
    Sampler,
    TopicSampler,
)
from .document import Document, DocumentSampler
from .generator import (
    CATEGORY_OUTPUT_FORMATS,
    DEFAULT_MODEL,
    DEFAULT_PROMPT_VARIANT_WEIGHTS,
    DEFAULT_SERVICE_TIER,
    FORM_OUTPUT_FORMATS,
    GENERATION_INSTRUCTIONS,
    LENGTH_WORD_RANGES,
    REGISTER_DESCRIPTIONS,
    SHOW_DONT_TELL_BLOCK,
    TEMPERATURE_RANGE,
    TOP_P_RANGE,
    DocumentGenerator,
    DocumentLength,
    GeneratedContent,
    GeneratedDocument,
    LengthSampler,
    OutputFormat,
    # Prompt variants (v0.4)
    PromptVariant,
    PromptVariantSampler,
    Register,
    RegisterSampler,
    SamplingParams,
    SamplingParamsSampler,
    TemplateGenerator,
    build_generation_prompt,
    build_system_prompt,
    resolve_output_format,
)
from .lc_data import LCDataLoader, load_lc_data
from .specs import (
    DocumentSpec,
    SpecBlock,
    assign_blocks_to_splits,
    draw_spec_blocks,
    load_spec_block,
    save_spec_block,
)

__all__ = [
    "DocumentSpec",
    "SpecBlock",
    "assign_blocks_to_splits",
    "draw_spec_blocks",
    "load_spec_block",
    "save_spec_block",
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
    "TEMPERATURE_RANGE",
    "TOP_P_RANGE",
    "SamplingParams",
    "SamplingParamsSampler",
    "build_generation_prompt",
    # Prompt variants (v0.4)
    "PromptVariant",
    "OutputFormat",
    "PromptVariantSampler",
    "DEFAULT_PROMPT_VARIANT_WEIGHTS",
    "SHOW_DONT_TELL_BLOCK",
    "CATEGORY_OUTPUT_FORMATS",
    "FORM_OUTPUT_FORMATS",
    "resolve_output_format",
    "build_system_prompt",
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
