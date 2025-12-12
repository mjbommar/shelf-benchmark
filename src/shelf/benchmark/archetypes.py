"""
Document archetypes based on co-occurrence analysis of CGP MARC records.

Each archetype represents a common "document profile" - a realistic combination
of agency, genre, LCC class, and topics that frequently occur together.
"""

from pydantic import BaseModel, Field


class Archetype(BaseModel):
    """A document archetype representing a common profile."""

    id: str = Field(description="Unique identifier")
    name: str = Field(description="Human-readable name")
    description: str = Field(default="", description="What this archetype represents")

    # Primary dimensions (always present)
    agency: str = Field(description="SuDoc agency code (e.g., 'I', 'C', 'Y')")
    agency_name: str = Field(description="Full agency name")
    genres: list[str] = Field(description="LCGFT genre/form terms")
    lcc_class: str = Field(description="LCC main class letter")
    lcc_name: str = Field(description="LCC class name")

    # Secondary dimensions (sample from these)
    topics: list[str] = Field(description="Likely LCSH topical subjects")
    geographic: list[str] = Field(
        default_factory=list, description="Likely geographic areas"
    )

    # Sampling weight (proportion of dataset)
    weight: float = Field(default=0.05, description="Relative weight in dataset (0-1)")

    # Generation hints
    title_patterns: list[str] = Field(
        default_factory=list, description="Example title patterns for generation"
    )


# Define archetypes based on co-occurrence analysis
ARCHETYPES: list[Archetype] = [
    Archetype(
        id="usgs_topo_map",
        name="USGS Topographic/Geologic Map",
        description="Maps produced by US Geological Survey covering geology, soils, and land features",
        agency="I",
        agency_name="Department of the Interior",
        genres=["Topographic maps", "Maps"],
        lcc_class="G",
        lcc_name="Geography, Anthropology, Recreation",
        topics=[
            "Geology",
            "Soils",
            "Land use",
            "Groundwater",
            "Mineral resources",
            "Coal",
            "Water-supply",
            "Geology, Stratigraphic",
            "Hydrology",
        ],
        geographic=[
            "United States",
            "California",
            "Colorado",
            "Montana",
            "Wyoming",
            "Alaska",
        ],
        weight=0.12,
        title_patterns=[
            "Geologic map of {location}",
            "Soil survey of {county}, {state}",
            "Groundwater resources of {region}",
        ],
    ),
    Archetype(
        id="census_statistics",
        name="Census Bureau Statistics",
        description="Statistical reports and census data from the Census Bureau",
        agency="C",
        agency_name="Department of Commerce (Census Bureau)",
        genres=["Statistics", "Census data"],
        lcc_class="H",
        lcc_name="Social Sciences",
        topics=[
            "Population",
            "Housing",
            "Manufactures",
            "Retail trade",
            "Agriculture",
            "Wholesale trade",
            "Service industries",
            "Construction industry",
            "Households",
            "Income",
            "Employment",
        ],
        geographic=["United States"],
        weight=0.10,
        title_patterns=[
            "Census of {topic}: {year}",
            "{topic} statistics: {year}",
            "Current population reports: {topic}",
        ],
    ),
    Archetype(
        id="congressional_hearing",
        name="Congressional Hearing",
        description="Testimony and proceedings from Congressional hearings",
        agency="Y",
        agency_name="Congress",
        genres=["Legislative hearings", "Legislative materials"],
        lcc_class="K",
        lcc_name="Law",
        topics=[
            "National security",
            "Terrorism",
            "Veterans",
            "Medicare",
            "Social security",
            "Immigration",
            "Budget",
            "Taxation",
            "Armed Forces",
            "Foreign relations",
        ],
        geographic=["United States"],
        weight=0.15,
        title_patterns=[
            "Hearing on {topic}",
            "{topic}: hearing before the {committee}",
            "Oversight hearing on {agency}",
        ],
    ),
    Archetype(
        id="defense_manual",
        name="Defense Technical Manual",
        description="Technical manuals and handbooks from the Department of Defense",
        agency="D",
        agency_name="Department of Defense",
        genres=["Handbooks and manuals", "Technical reports"],
        lcc_class="U",
        lcc_name="Military Science",
        topics=[
            "Guided missiles",
            "Military equipment",
            "Airplanes, Military",
            "Weapons systems",
            "Military operations",
            "Logistics",
        ],
        geographic=["United States"],
        weight=0.08,
        title_patterns=[
            "Technical manual: {equipment}",
            "Operator's manual for {system}",
            "Maintenance procedures for {equipment}",
        ],
    ),
    Archetype(
        id="agriculture_report",
        name="USDA Agriculture Report",
        description="Agricultural statistics and research from USDA",
        agency="A",
        agency_name="Department of Agriculture",
        genres=["Statistics", "Maps"],
        lcc_class="S",
        lcc_name="Agriculture",
        topics=[
            "Agriculture",
            "Crops",
            "Livestock",
            "Farm income",
            "Forests and forestry",
            "Soil conservation",
            "Food supply",
            "Agricultural prices",
        ],
        geographic=["United States", "California", "Texas", "Iowa", "Kansas"],
        weight=0.08,
        title_patterns=[
            "Agricultural statistics: {year}",
            "{crop} production: annual summary",
            "Soil survey of {county}",
        ],
    ),
    Archetype(
        id="health_statistics",
        name="Health Statistics Report",
        description="Public health statistics from HHS agencies",
        agency="HE",
        agency_name="Department of Health and Human Services",
        genres=["Statistics", "Periodicals"],
        lcc_class="R",
        lcc_name="Medicine",
        topics=[
            "Public health",
            "Vital statistics",
            "Mortality",
            "Diseases",
            "Health services",
            "Medicare",
            "Medicaid",
            "Hospitals",
        ],
        geographic=["United States"],
        weight=0.08,
        title_patterns=[
            "Vital statistics of the United States: {year}",
            "Health, United States: {year}",
            "Morbidity and mortality weekly report",
        ],
    ),
    Archetype(
        id="epa_environmental",
        name="EPA Environmental Report",
        description="Environmental reports and conference proceedings from EPA",
        agency="EP",
        agency_name="Environmental Protection Agency",
        genres=["Conference papers and proceedings", "Technical reports"],
        lcc_class="T",
        lcc_name="Technology",
        topics=[
            "Pollution",
            "Water quality",
            "Air quality",
            "Hazardous wastes",
            "Environmental protection",
            "Water--Pollution",
            "Drinking water",
        ],
        geographic=["United States"],
        weight=0.06,
        title_patterns=[
            "Environmental impact statement: {project}",
            "Water quality assessment: {watershed}",
            "Air quality trends report",
        ],
    ),
    Archetype(
        id="nasa_technical",
        name="NASA Technical Report",
        description="Technical reports and conference papers from NASA",
        agency="NAS",
        agency_name="National Aeronautics and Space Administration",
        genres=["Conference papers and proceedings", "Technical reports"],
        lcc_class="T",
        lcc_name="Technology",
        topics=[
            "Aeronautics",
            "Space flight",
            "Astronautics",
            "Rockets",
            "Satellites",
            "Space vehicles",
            "Aerospace engineering",
        ],
        geographic=["United States"],
        weight=0.06,
        title_patterns=[
            "NASA technical report: {topic}",
            "Proceedings of the {conference}",
            "{mission} technical summary",
        ],
    ),
    Archetype(
        id="education_report",
        name="Education Statistics Report",
        description="Education statistics and reports from the Department of Education",
        agency="ED",
        agency_name="Department of Education",
        genres=["Statistics"],
        lcc_class="L",
        lcc_name="Education",
        topics=[
            "Education",
            "Schools",
            "Students",
            "Teachers",
            "Higher education",
            "Educational tests",
            "Literacy",
            "School enrollment",
        ],
        geographic=["United States"],
        weight=0.05,
        title_patterns=[
            "Digest of education statistics: {year}",
            "Condition of education: {year}",
            "National assessment of educational progress",
        ],
    ),
    Archetype(
        id="loc_bibliography",
        name="Library of Congress Bibliography",
        description="Bibliographies and catalogs from the Library of Congress",
        agency="LC",
        agency_name="Library of Congress",
        genres=["Bibliographies", "Catalogs"],
        lcc_class="Z",
        lcc_name="Bibliography, Library Science",
        topics=[
            "Government publications",
            "Bibliography",
            "Library resources",
            "Cataloging",
            "Subject headings",
        ],
        geographic=["United States"],
        weight=0.04,
        title_patterns=[
            "Monthly catalog of United States government publications",
            "Bibliography of {topic}",
            "Subject guide to {collection}",
        ],
    ),
    Archetype(
        id="labor_statistics",
        name="Bureau of Labor Statistics Report",
        description="Employment and wage statistics from BLS",
        agency="L",
        agency_name="Department of Labor",
        genres=["Statistics", "Periodicals"],
        lcc_class="H",
        lcc_name="Social Sciences",
        topics=[
            "Wages",
            "Employment",
            "Labor supply",
            "Occupations",
            "Cost of living",
            "Prices",
            "Unemployment",
        ],
        geographic=["United States"],
        weight=0.05,
        title_patterns=[
            "Employment and earnings: {month} {year}",
            "Occupational outlook handbook",
            "Consumer price index: {month} {year}",
        ],
    ),
    Archetype(
        id="gao_audit",
        name="GAO Audit Report",
        description="Audit and oversight reports from the Government Accountability Office",
        agency="GA",
        agency_name="Government Accountability Office",
        genres=["Technical reports"],
        lcc_class="H",
        lcc_name="Social Sciences",
        topics=[
            "Auditing",
            "Government spending",
            "Program evaluation",
            "Administrative agencies",
            "Government accountability",
        ],
        geographic=["United States"],
        weight=0.05,
        title_patterns=[
            "{agency}: {finding}",
            "Audit of {program}",
            "{topic}: improvements needed",
        ],
    ),
    Archetype(
        id="transportation_report",
        name="Transportation Report",
        description="Transportation statistics and studies from DOT",
        agency="TD",
        agency_name="Department of Transportation",
        genres=["Statistics", "Handbooks and manuals"],
        lcc_class="H",
        lcc_name="Social Sciences",
        topics=[
            "Transportation",
            "Highways",
            "Railroads",
            "Aviation",
            "Traffic safety",
            "Automobiles",
            "Shipping",
        ],
        geographic=["United States"],
        weight=0.04,
        title_patterns=[
            "National transportation statistics",
            "Highway statistics: {year}",
            "Traffic safety facts",
        ],
    ),
    Archetype(
        id="energy_report",
        name="Energy Information Report",
        description="Energy statistics and analysis from DOE/EIA",
        agency="E",
        agency_name="Department of Energy",
        genres=["Statistics", "Periodicals"],
        lcc_class="T",
        lcc_name="Technology",
        topics=[
            "Energy",
            "Petroleum",
            "Natural gas",
            "Coal",
            "Nuclear energy",
            "Renewable energy",
            "Electric power",
            "Energy consumption",
        ],
        geographic=["United States"],
        weight=0.04,
        title_patterns=[
            "Annual energy review",
            "Monthly energy review",
            "{fuel} production report",
        ],
    ),
]


def get_archetype_by_id(archetype_id: str) -> Archetype | None:
    """Get an archetype by its ID."""
    for arch in ARCHETYPES:
        if arch.id == archetype_id:
            return arch
    return None


def get_total_weight() -> float:
    """Get the sum of all archetype weights."""
    return sum(arch.weight for arch in ARCHETYPES)


def normalize_weights() -> list[float]:
    """Get normalized weights that sum to 1.0."""
    total = get_total_weight()
    return [arch.weight / total for arch in ARCHETYPES]
