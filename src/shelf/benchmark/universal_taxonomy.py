"""
Universal Document Taxonomy for Benchmarking

This extends the LC-based taxonomy to cover ALL common document types,
not just government publications. It combines:
- LCGFT genre/form terms (where they exist)
- Custom document types (for gaps in LCGFT)
- LCC subject domains (for topical coverage)
- Practical document archetypes for generation
"""

from enum import Enum
from pydantic import BaseModel, Field


class DocumentDomain(str, Enum):
    """High-level document domains (universal, not LC-specific)."""

    BUSINESS = "business"  # Corporate, professional
    LEGAL = "legal"  # Law, contracts, regulations
    ACADEMIC = "academic"  # Scholarly, research
    JOURNALISM = "journalism"  # News, media
    TECHNICAL = "technical"  # Engineering, software, science
    CREATIVE = "creative"  # Literary, artistic
    EDUCATIONAL = "educational"  # Teaching, learning
    GOVERNMENT = "government"  # Policy, legislation
    MEDICAL = "medical"  # Healthcare, clinical
    FINANCIAL = "financial"  # Investments, accounting


class DocumentType(BaseModel):
    """A document type with LCGFT mapping if available."""

    id: str = Field(description="Unique identifier")
    name: str = Field(description="Human-readable name")
    domain: DocumentDomain = Field(description="High-level domain")
    lcgft_term: str | None = Field(default=None, description="LCGFT term if exists")
    lcc_classes: list[str] = Field(
        default_factory=list, description="Relevant LCC classes"
    )
    description: str = Field(default="", description="What this document type is")
    examples: list[str] = Field(default_factory=list, description="Example documents")


# Define comprehensive document types
DOCUMENT_TYPES: list[DocumentType] = [
    # =========================================================================
    # BUSINESS & PROFESSIONAL
    # =========================================================================
    DocumentType(
        id="business_report",
        name="Business Report",
        domain=DocumentDomain.BUSINESS,
        lcgft_term="Annual reports",  # partial match
        lcc_classes=["H"],
        description="Corporate reports, analyses, and business summaries",
        examples=["Annual Report 2023", "Market Analysis Q4", "Strategic Plan"],
    ),
    DocumentType(
        id="business_correspondence",
        name="Business Correspondence",
        domain=DocumentDomain.BUSINESS,
        lcgft_term="Business correspondence",
        lcc_classes=["H"],
        description="Professional letters, memos, and communications",
        examples=["Cover letter", "Business proposal letter", "Meeting minutes"],
    ),
    DocumentType(
        id="contract",
        name="Contract",
        domain=DocumentDomain.BUSINESS,
        lcgft_term="Contracts",
        lcc_classes=["K"],
        description="Legal agreements between parties",
        examples=["Employment contract", "Service agreement", "NDA"],
    ),
    DocumentType(
        id="proposal",
        name="Proposal",
        domain=DocumentDomain.BUSINESS,
        lcgft_term=None,
        lcc_classes=["H"],
        description="Business proposals, RFP responses, project proposals",
        examples=["Grant proposal", "Project proposal", "Business plan"],
    ),
    DocumentType(
        id="presentation",
        name="Presentation",
        domain=DocumentDomain.BUSINESS,
        lcgft_term=None,
        lcc_classes=["H"],
        description="Slide decks, pitch presentations, keynotes",
        examples=["Sales pitch", "Quarterly review", "Product demo"],
    ),
    # =========================================================================
    # LEGAL & REGULATORY
    # =========================================================================
    DocumentType(
        id="court_filing",
        name="Court Filing",
        domain=DocumentDomain.LEGAL,
        lcgft_term="Court decisions and opinions",
        lcc_classes=["K"],
        description="Legal briefs, motions, and court documents",
        examples=["Motion to dismiss", "Appellate brief", "Complaint"],
    ),
    DocumentType(
        id="legislation",
        name="Legislation",
        domain=DocumentDomain.LEGAL,
        lcgft_term="Statutes and codes",
        lcc_classes=["K"],
        description="Laws, statutes, and legislative acts",
        examples=["Federal statute", "State law", "Municipal code"],
    ),
    DocumentType(
        id="regulation",
        name="Regulation",
        domain=DocumentDomain.LEGAL,
        lcgft_term="Administrative regulations",
        lcc_classes=["K"],
        description="Administrative rules and regulatory guidance",
        examples=["FDA regulation", "EPA rule", "SEC filing requirement"],
    ),
    DocumentType(
        id="patent",
        name="Patent",
        domain=DocumentDomain.LEGAL,
        lcgft_term="Patents",
        lcc_classes=["T"],
        description="Patent applications and grants",
        examples=["Utility patent", "Design patent", "Patent application"],
    ),
    # =========================================================================
    # ACADEMIC & SCHOLARLY
    # =========================================================================
    DocumentType(
        id="research_paper",
        name="Research Paper",
        domain=DocumentDomain.ACADEMIC,
        lcgft_term="Academic theses",  # partial match
        lcc_classes=["Q", "H", "P"],
        description="Peer-reviewed research articles and papers",
        examples=["Journal article", "Conference paper", "Working paper"],
    ),
    DocumentType(
        id="thesis",
        name="Thesis/Dissertation",
        domain=DocumentDomain.ACADEMIC,
        lcgft_term="Academic theses",
        lcc_classes=["Q", "H", "L"],
        description="Graduate theses and doctoral dissertations",
        examples=["PhD dissertation", "Master's thesis", "Honors thesis"],
    ),
    DocumentType(
        id="textbook",
        name="Textbook",
        domain=DocumentDomain.ACADEMIC,
        lcgft_term="Textbooks",
        lcc_classes=["L"],
        description="Educational textbooks and course materials",
        examples=["Introduction to Psychology", "Calculus I", "Organic Chemistry"],
    ),
    DocumentType(
        id="literature_review",
        name="Literature Review",
        domain=DocumentDomain.ACADEMIC,
        lcgft_term="Bibliographies",  # partial match
        lcc_classes=["Z"],
        description="Systematic reviews and meta-analyses",
        examples=["Systematic review", "Meta-analysis", "Survey paper"],
    ),
    # =========================================================================
    # JOURNALISM & MEDIA
    # =========================================================================
    DocumentType(
        id="news_article",
        name="News Article",
        domain=DocumentDomain.JOURNALISM,
        lcgft_term=None,
        lcc_classes=["P"],
        description="News reports and journalism",
        examples=["Breaking news", "Feature story", "Investigative report"],
    ),
    DocumentType(
        id="editorial",
        name="Editorial/Opinion",
        domain=DocumentDomain.JOURNALISM,
        lcgft_term="Editorials",
        lcc_classes=["P"],
        description="Opinion pieces, editorials, and op-eds",
        examples=["Editorial", "Op-ed", "Column"],
    ),
    DocumentType(
        id="press_release",
        name="Press Release",
        domain=DocumentDomain.JOURNALISM,
        lcgft_term="Press releases",
        lcc_classes=["H"],
        description="Official announcements and PR communications",
        examples=["Product launch", "Earnings announcement", "Executive change"],
    ),
    DocumentType(
        id="interview",
        name="Interview",
        domain=DocumentDomain.JOURNALISM,
        lcgft_term="Interviews",
        lcc_classes=["P"],
        description="Interview transcripts and Q&A sessions",
        examples=["Celebrity interview", "Expert Q&A", "Podcast transcript"],
    ),
    # =========================================================================
    # TECHNICAL & SCIENTIFIC
    # =========================================================================
    DocumentType(
        id="technical_manual",
        name="Technical Manual",
        domain=DocumentDomain.TECHNICAL,
        lcgft_term="Handbooks and manuals",
        lcc_classes=["T"],
        description="Technical documentation and user guides",
        examples=["User manual", "Installation guide", "API documentation"],
    ),
    DocumentType(
        id="technical_report",
        name="Technical Report",
        domain=DocumentDomain.TECHNICAL,
        lcgft_term="Technical reports",
        lcc_classes=["T", "Q"],
        description="Technical analyses and engineering reports",
        examples=["Feasibility study", "Technical assessment", "Lab report"],
    ),
    DocumentType(
        id="specification",
        name="Specification",
        domain=DocumentDomain.TECHNICAL,
        lcgft_term=None,
        lcc_classes=["T"],
        description="Technical specifications and standards",
        examples=["Product spec", "API spec", "Design document"],
    ),
    DocumentType(
        id="scientific_data",
        name="Scientific Data",
        domain=DocumentDomain.TECHNICAL,
        lcgft_term="Statistics",
        lcc_classes=["Q"],
        description="Datasets, data reports, and statistical analyses",
        examples=["Dataset documentation", "Data dictionary", "Statistical report"],
    ),
    # =========================================================================
    # CREATIVE & LITERARY
    # =========================================================================
    DocumentType(
        id="novel",
        name="Novel",
        domain=DocumentDomain.CREATIVE,
        lcgft_term="Novels",
        lcc_classes=["P"],
        description="Long-form fiction",
        examples=["Literary novel", "Genre fiction", "Young adult novel"],
    ),
    DocumentType(
        id="short_story",
        name="Short Story",
        domain=DocumentDomain.CREATIVE,
        lcgft_term="Short stories",
        lcc_classes=["P"],
        description="Short fiction",
        examples=["Flash fiction", "Short story", "Novella"],
    ),
    DocumentType(
        id="poetry",
        name="Poetry",
        domain=DocumentDomain.CREATIVE,
        lcgft_term="Poetry",
        lcc_classes=["P"],
        description="Poems and verse",
        examples=["Sonnet", "Free verse", "Epic poem"],
    ),
    DocumentType(
        id="essay",
        name="Essay",
        domain=DocumentDomain.CREATIVE,
        lcgft_term="Essays",
        lcc_classes=["P"],
        description="Personal and literary essays",
        examples=["Personal essay", "Literary essay", "Reflective essay"],
    ),
    DocumentType(
        id="screenplay",
        name="Screenplay/Script",
        domain=DocumentDomain.CREATIVE,
        lcgft_term="Screenplays",
        lcc_classes=["P"],
        description="Scripts for film, TV, and theater",
        examples=["Movie screenplay", "TV script", "Stage play"],
    ),
    # =========================================================================
    # EDUCATIONAL & INSTRUCTIONAL
    # =========================================================================
    DocumentType(
        id="lesson_plan",
        name="Lesson Plan",
        domain=DocumentDomain.EDUCATIONAL,
        lcgft_term="Lesson plans",
        lcc_classes=["L"],
        description="Teaching plans and curricula",
        examples=["Daily lesson plan", "Unit plan", "Curriculum guide"],
    ),
    DocumentType(
        id="tutorial",
        name="Tutorial",
        domain=DocumentDomain.EDUCATIONAL,
        lcgft_term=None,
        lcc_classes=["L"],
        description="How-to guides and instructional content",
        examples=["How-to guide", "Step-by-step tutorial", "Video tutorial transcript"],
    ),
    DocumentType(
        id="study_guide",
        name="Study Guide",
        domain=DocumentDomain.EDUCATIONAL,
        lcgft_term="Study guides",
        lcc_classes=["L"],
        description="Study materials and exam prep",
        examples=["Exam study guide", "Course notes", "Review sheet"],
    ),
    # =========================================================================
    # GOVERNMENT & POLICY
    # =========================================================================
    DocumentType(
        id="policy_brief",
        name="Policy Brief",
        domain=DocumentDomain.GOVERNMENT,
        lcgft_term="Policy briefs",
        lcc_classes=["J", "H"],
        description="Policy analyses and recommendations",
        examples=["Policy memo", "White paper", "Issue brief"],
    ),
    DocumentType(
        id="government_report",
        name="Government Report",
        domain=DocumentDomain.GOVERNMENT,
        lcgft_term="Statistics",  # partial
        lcc_classes=["J", "H"],
        description="Official government reports and analyses",
        examples=["GAO report", "Agency report", "Inspector General report"],
    ),
    DocumentType(
        id="hearing_testimony",
        name="Hearing/Testimony",
        domain=DocumentDomain.GOVERNMENT,
        lcgft_term="Legislative hearings",
        lcc_classes=["K", "J"],
        description="Congressional hearings and testimony",
        examples=["Senate hearing", "House testimony", "Regulatory hearing"],
    ),
    # =========================================================================
    # MEDICAL & HEALTHCARE
    # =========================================================================
    DocumentType(
        id="clinical_note",
        name="Clinical Note",
        domain=DocumentDomain.MEDICAL,
        lcgft_term=None,
        lcc_classes=["R"],
        description="Medical records and clinical documentation",
        examples=["Progress note", "Discharge summary", "Consultation note"],
    ),
    DocumentType(
        id="medical_research",
        name="Medical Research",
        domain=DocumentDomain.MEDICAL,
        lcgft_term="Academic theses",  # partial
        lcc_classes=["R"],
        description="Medical research papers and clinical studies",
        examples=["Clinical trial report", "Case study", "Medical review"],
    ),
    DocumentType(
        id="patient_education",
        name="Patient Education",
        domain=DocumentDomain.MEDICAL,
        lcgft_term=None,
        lcc_classes=["R"],
        description="Patient-facing health information",
        examples=["Patient handout", "Medication guide", "Health FAQ"],
    ),
    # =========================================================================
    # FINANCIAL & ECONOMIC
    # =========================================================================
    DocumentType(
        id="financial_report",
        name="Financial Report",
        domain=DocumentDomain.FINANCIAL,
        lcgft_term="Annual reports",  # partial
        lcc_classes=["H"],
        description="Financial statements and reports",
        examples=["10-K filing", "Quarterly earnings", "Audit report"],
    ),
    DocumentType(
        id="investment_analysis",
        name="Investment Analysis",
        domain=DocumentDomain.FINANCIAL,
        lcgft_term=None,
        lcc_classes=["H"],
        description="Investment research and recommendations",
        examples=["Equity research", "Credit analysis", "Due diligence report"],
    ),
    DocumentType(
        id="economic_report",
        name="Economic Report",
        domain=DocumentDomain.FINANCIAL,
        lcgft_term="Statistics",
        lcc_classes=["H"],
        description="Economic analyses and forecasts",
        examples=["GDP report", "Employment statistics", "Inflation analysis"],
    ),
]


def get_document_types_by_domain(domain: DocumentDomain) -> list[DocumentType]:
    """Get all document types for a given domain."""
    return [dt for dt in DOCUMENT_TYPES if dt.domain == domain]


def get_document_type_by_id(doc_type_id: str) -> DocumentType | None:
    """Get a document type by its ID."""
    for dt in DOCUMENT_TYPES:
        if dt.id == doc_type_id:
            return dt
    return None


def get_types_with_lcgft() -> list[DocumentType]:
    """Get document types that have LCGFT mappings."""
    return [dt for dt in DOCUMENT_TYPES if dt.lcgft_term is not None]


def get_types_without_lcgft() -> list[DocumentType]:
    """Get document types without LCGFT mappings (custom types)."""
    return [dt for dt in DOCUMENT_TYPES if dt.lcgft_term is None]


# Summary statistics
DOMAIN_COUNTS = {
    domain: len(get_document_types_by_domain(domain)) for domain in DocumentDomain
}

TOTAL_DOCUMENT_TYPES = len(DOCUMENT_TYPES)
TYPES_WITH_LCGFT = len(get_types_with_lcgft())
TYPES_WITHOUT_LCGFT = len(get_types_without_lcgft())
