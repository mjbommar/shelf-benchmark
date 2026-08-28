"""Task registry for SHELF evaluation.

This module defines all available evaluation tasks with their
specifications.
"""

from __future__ import annotations

from shelf.evaluate.tasks import TaskSpec, TaskType

# Library of Congress Classification codes (21 classes)
LCC_CODES = (
    "A",  # General Works
    "B",  # Philosophy, Psychology, Religion
    "C",  # Auxiliary Sciences of History
    "D",  # World History
    "E",  # History of the Americas (General)
    "F",  # History of the Americas (Local)
    "G",  # Geography, Anthropology, Recreation
    "H",  # Social Sciences
    "J",  # Political Science
    "K",  # Law
    "L",  # Education
    "M",  # Music
    "N",  # Fine Arts
    "P",  # Language and Literature
    "Q",  # Science
    "R",  # Medicine
    "S",  # Agriculture
    "T",  # Technology
    "U",  # Military Science
    "V",  # Naval Science
    "Z",  # Bibliography, Library Science
)

# LCC subclasses (80 codes) -- the v0.4 Phase 2 difficulty tier.
#
# `lcc_classification` over the 21 main classes is lexically saturated: TF-IDF+LR
# reaches 0.892 macro-F1 and still scores 0.754 on 22-word documents, because a
# main class is decodable from domain vocabulary alone. Subclasses are not --
# QA/QC/QH all speak the vocabulary of science, KF/KJ/KZ all speak the vocabulary
# of law -- so this task measures within-domain discrimination.
#
# Sourced from data/taxonomies/lcc_subclass_top100.json via
# `shelf.sampler.dimensions.load_lcc_subclass_pool()`, which is the authority;
# the pool is repeated here so the macro-F1 denominator and confusion-matrix
# ordering stay fixed even on a subset that does not observe every subclass.
# The 100-entry table yields 80 usable codes:
#   - 3 dropped as extraction artifacts, not LCC codes (IN, PAR, NOT);
#   - 16 dropped as bare main-class letters ("Q" versus "QA" is genuinely
#     ambiguous for a generated document, so it is label noise here);
#   - 1 dropped for having no LC description to prompt from (JX, a discontinued
#     subclass the classification API no longer resolves).
# Sampling over these is UNIFORM, never by MARC frequency: KF alone is 122,484
# of the reference corpus, more than the other 99 codes combined.
LCC_SUBCLASS_CODES = (
    "CD",  # C -- Diplomatics. Archives. Seals
    "DK",  # D -- History of Russia. Soviet Union. Former Soviet Republics
    "DR",  # D -- History of Balkan Peninsula
    "DS",  # D -- History of Asia
    "DT",  # D -- History of Africa
    "GB",  # G -- Physical geography
    "GC",  # G -- Oceanography
    "GE",  # G -- Environmental sciences
    "GN",  # G -- Anthropology
    "GV",  # G -- Recreation. Leisure
    "HA",  # H -- Statistics
    "HB",  # H -- Economic theory. Demography
    "HC",  # H -- Economic history and conditions
    "HD",  # H -- Industries. Land use. Labor
    "HE",  # H -- Transportation and communications
    "HF",  # H -- Commerce
    "HG",  # H -- Finance
    "HJ",  # H -- Public finance
    "HN",  # H -- Social history and conditions. Social problems. Social reform
    "HQ",  # H -- The Family. Marriage. Women
    "HT",  # H -- Communities. Classes. Races
    "HV",  # H -- Social pathology. Social and public welfare. Criminology
    "JC",  # J -- Political theory. The state. Theories of the state
    "JK",  # J -- Political institutions and public administration (United States)
    "JN",  # J -- Political institutions and public administration (Europe)
    "JV",  # J -- Colonies and colonization
    "JZ",  # J -- International relations
    "KF",  # K -- United States (General)
    "KZ",  # K -- Law of nations
    "LA",  # L -- History of education
    "LB",  # L -- Theory and practice of education
    "LC",  # L -- Special aspects of education
    "ML",  # M -- Literature on music
    "NA",  # N -- Architecture
    "NX",  # N -- Arts in general
    "PN",  # P -- Literature (General)
    "QA",  # Q -- Mathematics
    "QB",  # Q -- Astronomy
    "QC",  # Q -- Physics
    "QD",  # Q -- Chemistry
    "QE",  # Q -- Geology
    "QH",  # Q -- Biology (General)
    "QK",  # Q -- Botany
    "QL",  # Q -- Zoology
    "QP",  # Q -- Physiology
    "RA",  # R -- Public aspects of medicine
    "RC",  # R -- Internal medicine
    "RD",  # R -- Surgery
    "RG",  # R -- Gynecology and obstetrics
    "RJ",  # R -- Pediatrics
    "RM",  # R -- Therapeutics. Pharmacology
    "SB",  # S -- Plant culture
    "SD",  # S -- Forestry
    "SF",  # S -- Animal culture
    "SH",  # S -- Aquaculture. Fisheries. Angling
    "SK",  # S -- Hunting sports
    "TA",  # T -- Engineering (General). Civil engineering (General)
    "TC",  # T -- Hydraulic engineering
    "TD",  # T -- Environmental technology. Sanitary engineering
    "TE",  # T -- Highway engineering. Roads and pavements
    "TH",  # T -- Building construction
    "TJ",  # T -- Mechanical engineering and machinery
    "TK",  # T -- Electrical engineering. Electronics. Nuclear engineering
    "TL",  # T -- Motor vehicles. Aeronautics. Astronautics
    "TN",  # T -- Mining engineering. Metallurgy
    "TP",  # T -- Chemical technology
    "TS",  # T -- Manufactures
    "TX",  # T -- Home economics
    "UA",  # U -- Armies: Organization, distribution, military situation
    "UB",  # U -- Military administration
    "UC",  # U -- Maintenance and transportation
    "UF",  # U -- Artillery
    "UG",  # U -- Air forces. Air warfare
    "UH",  # U -- Other services
    "VA",  # V -- Navies: Organization, distribution, naval situation
    "VC",  # V -- Naval maintenance
    "VE",  # V -- Marines
    "VG",  # V -- Minor services of navies
    "VK",  # V -- Navigation. Merchant marine
    "VM",  # V -- Naval architecture. Shipbuilding. Marine engineering
)

# LCGFT Categories (14 categories)
LCGFT_CATEGORIES = (
    "Cartographic materials",
    "Commemorative works",
    "Creative nonfiction",
    "Discursive works",
    "Ephemera",
    "Informational works",
    "Instructional and educational works",
    "Law materials",
    "Literature",
    "Music",
    "Recreational works",
    "Religious materials",
    "Sound recordings",
    "Visual works",
)

# LCGFT Forms (133 specific genre/form terms, present in every split of mjbommar/SHELF)
LCGFT_FORMS = (
    "Abstracts",
    "Academic theses",
    "Activity books",
    "Administrative decisions",
    "Administrative regulations",
    "Aerial photographs",
    "Analysis",
    "Anniversary publications",
    "Architectural drawings",
    "Art",
    "Art music",
    "Atlases",
    "Audiobooks",
    "Biographies",
    "Blogs",
    "Broadsides",
    "Brochures",
    "Calendars",
    "Case studies",
    "Casebooks (Law)",
    "Chamber music",
    "Charts",
    "Comics (Graphic works)",
    "Commentaries",
    "Conference papers and proceedings",
    "Constitutions",
    "Contracts",
    "Cookbooks",
    "Course materials",
    "Court decisions and opinions",
    "Criticism",
    "Data sets",
    "Databases",
    "Debates",
    "Devotional literature",
    "Diagrams",
    "Diaries",
    "Drama",
    "Drawings",
    "Editorials",
    "Educational films",
    "Essays",
    "Eulogies",
    "Examinations",
    "FAQs",
    "Fantasy fiction",
    "Festschriften",
    "Fiction",
    "Field recordings",
    "Floor plans",
    "Flyers",
    "Folk literature",
    "Folk music",
    "Games",
    "Globes",
    "Greeting cards",
    "Handbooks and manuals",
    "How-to guides",
    "Humor",
    "Illustrations",
    "Infographics",
    "Instrumental music",
    "Interviews",
    "Jokes",
    "Journalism",
    "Lectures",
    "Legal briefs",
    "Legal forms",
    "Legislative materials",
    "Lesson plans",
    "Letters to the editor",
    "Liturgical texts",
    "Maps",
    "Memoirs",
    "Memorial books",
    "Menus",
    "Motion pictures",
    "Music recordings",
    "Mystery fiction",
    "Nautical charts",
    "News articles",
    "Novels",
    "Obituaries",
    "Opera",
    "Opinion pieces",
    "Oral histories",
    "Orchestral music",
    "Pamphlets",
    "Panel discussions",
    "Personal narratives",
    "Photographs",
    "Plays",
    "Podcasts",
    "Poetry",
    "Policy briefs",
    "Popular music",
    "Postcards",
    "Posters",
    "Prayers",
    "Press releases",
    "Profiles",
    "Puzzles",
    "Radio programs",
    "Reference works",
    "Reviews",
    "Riddles",
    "Sacred music",
    "Sacred works",
    "Sagas",
    "Satellite imagery",
    "Satire",
    "Science fiction",
    "Screenplays",
    "Sermons",
    "Short stories",
    "Songs",
    "Speeches",
    "Statistics",
    "Statutes and codes",
    "Study guides",
    "Surveys",
    "Technical reports",
    "Television programs",
    "Textbooks",
    "Theological works",
    "Travel writing",
    "Treaties",
    "Tributes",
    "True crime stories",
    "Tutorials",
    "Video recordings",
    "Workbooks",
    "Yearbooks",
)

# LCSH topical terms used as multi-label targets (112 terms)
TOPICS = (
    "Accounting",
    "Administrative law",
    "Aesthetics",
    "Analysis",
    "Anthropology",
    "Art",
    "Artificial intelligence",
    "Astronomy",
    "Authoritarianism",
    "Biodiversity",
    "Biology",
    "Biotechnology",
    "Carbon emissions",
    "Cardiology",
    "Chemistry",
    "Civil law",
    "Climate",
    "Climate change",
    "Cloud computing",
    "Commerce",
    "Computer science",
    "Conservation",
    "Constitutional law",
    "Contracts",
    "Criminal law",
    "Culture",
    "Cybersecurity",
    "Data science",
    "Defense",
    "Deforestation",
    "Democracy",
    "Demographics",
    "Diplomacy",
    "Diseases",
    "E-commerce",
    "Ecology",
    "Economics",
    "Ecosystems",
    "Elections",
    "Engineering",
    "Entrepreneurship",
    "Environmental law",
    "Epidemiology",
    "Ethics",
    "Evolution",
    "Finance",
    "Genetics",
    "Geology",
    "Globalization",
    "Government",
    "History",
    "Human resources",
    "Human rights",
    "Immigration",
    "Immunology",
    "Information",
    "Innovation",
    "Intellectual property",
    "International law",
    "International relations",
    "Knowledge",
    "Labor",
    "Languages",
    "Leadership",
    "Literature",
    "Machine learning",
    "Management",
    "Marketing",
    "Mathematics",
    "Mental health",
    "Methodology",
    "Music",
    "Nanotechnology",
    "Nationalism",
    "Networks",
    "Neuroscience",
    "Nutrition",
    "Ocean conservation",
    "Oncology",
    "Operations",
    "Pediatrics",
    "Pharmacology",
    "Philosophy",
    "Physics",
    "Political parties",
    "Pollution",
    "Population",
    "Poverty",
    "Property",
    "Psychology",
    "Public administration",
    "Public health",
    "Public policy",
    "Quantum mechanics",
    "Religion",
    "Renewable energy",
    "Research",
    "Robotics",
    "Security",
    "Social welfare",
    "Sociology",
    "Software",
    "Statistics",
    "Strategy",
    "Supply chain",
    "Surgery",
    "Surveys",
    "Sustainability",
    "Therapeutics",
    "Thermodynamics",
    "Torts",
    "Wildlife",
)

# Register of writing styles
REGISTERS = (
    "academic",
    "casual",
    "conversational",
    "creative",
    "formal",
    "journalistic",
    "professional",
    "technical",
)

# Geographic regions (8 clusters for geographic clustering)
GEOGRAPHIC_REGIONS = (
    "North America",
    "South America",
    "Europe",
    "East Asia",
    "South/Southeast Asia",
    "Middle East & North Africa",
    "Sub-Saharan Africa",
    "Central America & Caribbean",
)


# ============================================================================
# TASK REGISTRY
# ============================================================================

TASK_REGISTRY: dict[str, TaskSpec] = {
    # -------------------------------------------------------------------------
    # RETRIEVAL TASKS (highest priority for RAG)
    # -------------------------------------------------------------------------
    "lcc_retrieval": TaskSpec(
        name="lcc_retrieval",
        task_type=TaskType.RETRIEVAL,
        description="Retrieve documents with the same Library of Congress Classification (subject area)",
        text_field="text",
        label_field="lcc_code",
        id_field="id",
        label_space=LCC_CODES,
        primary_metric="ndcg@10",
        secondary_metrics=(
            "graded_ndcg@10",
            "graded_ndcg@100",
            "mrr",
            "recall@10",
            "recall@100",
            "map@10",
        ),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    "form_retrieval": TaskSpec(
        name="form_retrieval",
        task_type=TaskType.RETRIEVAL,
        description="Retrieve documents with the same LCGFT genre/form",
        text_field="text",
        label_field="lcgft_form",
        id_field="id",
        label_space=None,  # 133 forms, open vocabulary
        primary_metric="ndcg@10",
        secondary_metrics=(
            "graded_ndcg@10",
            "graded_ndcg@100",
            "mrr",
            "recall@10",
            "recall@100",
            "map@10",
        ),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    "category_retrieval": TaskSpec(
        name="category_retrieval",
        task_type=TaskType.RETRIEVAL,
        description="Retrieve documents with the same LCGFT category",
        text_field="text",
        label_field="lcgft_category",
        id_field="id",
        label_space=LCGFT_CATEGORIES,
        primary_metric="ndcg@10",
        secondary_metrics=(
            "graded_ndcg@10",
            "graded_ndcg@100",
            "mrr",
            "recall@10",
            "recall@100",
            "map@10",
        ),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    # -------------------------------------------------------------------------
    # INSTRUCTION-FOLLOWING RETRIEVAL TASKS
    #
    # Relevance is defined by the instruction, not by a label field, so the same
    # query document has a different correct answer set under each instruction --
    # and under the first two below, disjoint answer sets. The instruction text
    # and the constraint definitions live in `shelf.evaluate.instructions`;
    # `label_field` here names the anchor facet only, for tooling that assumes
    # every task has one.
    # -------------------------------------------------------------------------
    "instruction_same_subject_diff_form": TaskSpec(
        name="instruction_same_subject_diff_form",
        task_type=TaskType.RETRIEVAL,
        description=(
            "Instruction-following retrieval: same subject area, different genre/form"
        ),
        text_field="text",
        label_field="lcc_code",
        id_field="id",
        label_space=LCC_CODES,
        primary_metric="ndcg@10",
        secondary_metrics=(
            "mrr",
            "recall@10",
            "recall@100",
            "map@10",
            "anchor_match@10",
            "contrast_violation@10",
            "contrast_violation_lift@10",
        ),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    "instruction_same_form_diff_subject": TaskSpec(
        name="instruction_same_form_diff_subject",
        task_type=TaskType.RETRIEVAL,
        description=(
            "Instruction-following retrieval: same genre/form, different subject area"
        ),
        text_field="text",
        label_field="lcgft_form",
        id_field="id",
        label_space=None,  # 133 forms, open vocabulary
        primary_metric="ndcg@10",
        secondary_metrics=(
            "mrr",
            "recall@10",
            "recall@100",
            "map@10",
            "anchor_match@10",
            "contrast_violation@10",
            "contrast_violation_lift@10",
        ),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    "instruction_same_topic_diff_subject": TaskSpec(
        name="instruction_same_topic_diff_subject",
        task_type=TaskType.RETRIEVAL,
        description=(
            "Instruction-following retrieval: shares a topic, different subject area"
        ),
        text_field="text",
        label_field="topics",
        id_field="id",
        label_space=None,  # 112 topics, multi-valued
        primary_metric="ndcg@10",
        secondary_metrics=(
            "mrr",
            "recall@10",
            "recall@100",
            "map@10",
            "anchor_match@10",
            "contrast_violation@10",
            "contrast_violation_lift@10",
        ),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    "instruction_same_audience_diff_register": TaskSpec(
        name="instruction_same_audience_diff_register",
        task_type=TaskType.RETRIEVAL,
        description=(
            "Instruction-following retrieval: same audience, different writing register"
        ),
        text_field="text",
        label_field="audience",
        id_field="id",
        label_space=None,  # 25 audience values, ~70% coverage
        primary_metric="ndcg@10",
        secondary_metrics=(
            "mrr",
            "recall@10",
            "recall@100",
            "map@10",
            "anchor_match@10",
            "contrast_violation@10",
            "contrast_violation_lift@10",
        ),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    # -------------------------------------------------------------------------
    # CLASSIFICATION TASKS
    # -------------------------------------------------------------------------
    "lcc_classification": TaskSpec(
        name="lcc_classification",
        task_type=TaskType.CLASSIFICATION,
        description="Classify documents into 21 Library of Congress subject classes",
        text_field="text",
        label_field="lcc_code",
        id_field="id",
        label_space=LCC_CODES,
        primary_metric="macro_f1",
        secondary_metrics=("micro_f1", "accuracy", "weighted_f1"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    "lcc_subclass_classification": TaskSpec(
        name="lcc_subclass_classification",
        task_type=TaskType.CLASSIFICATION,
        description=(
            "Classify documents into 80 Library of Congress subclasses -- the "
            "within-domain discrimination tier (QA vs QC vs QH, KF vs KZ) that "
            "domain vocabulary alone cannot solve"
        ),
        text_field="text",
        label_field="lcc_subclass",
        id_field="id",
        label_space=LCC_SUBCLASS_CODES,
        primary_metric="macro_f1",
        secondary_metrics=("micro_f1", "accuracy", "weighted_f1"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        # v0.4 only. `lcc_subclass` is empty for every v0.3.1 document, so this
        # task has nothing to score until the Phase 2 slice is published.
        default_split="test",
    ),
    "form_classification": TaskSpec(
        name="form_classification",
        task_type=TaskType.CLASSIFICATION,
        description="Classify documents into 133 LCGFT genre/form terms",
        text_field="text",
        label_field="lcgft_form",
        id_field="id",
        # Explicit 133-term space: keeps the macro-F1 denominator, the
        # confusion-matrix ordering and prediction validation stable even when
        # evaluating a filtered subset that does not observe every form.
        label_space=LCGFT_FORMS,
        primary_metric="macro_f1",
        secondary_metrics=("micro_f1", "accuracy", "weighted_f1"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    "lcgft_category_classification": TaskSpec(
        name="lcgft_category_classification",
        task_type=TaskType.CLASSIFICATION,
        description="Classify documents into 14 LCGFT genre categories",
        text_field="text",
        label_field="lcgft_category",
        id_field="id",
        label_space=LCGFT_CATEGORIES,
        primary_metric="macro_f1",
        secondary_metrics=("micro_f1", "accuracy", "weighted_f1"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    "register_classification": TaskSpec(
        name="register_classification",
        task_type=TaskType.CLASSIFICATION,
        description="Classify documents by writing register/style",
        text_field="text",
        label_field="register",
        id_field="id",
        label_space=REGISTERS,
        primary_metric="macro_f1",
        secondary_metrics=("micro_f1", "accuracy", "weighted_f1"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    # -------------------------------------------------------------------------
    # MULTI-LABEL CLASSIFICATION TASKS
    # -------------------------------------------------------------------------
    "topic_classification": TaskSpec(
        name="topic_classification",
        task_type=TaskType.MULTILABEL,
        description=(
            "Assign each document its set of LCSH topical terms "
            "(multi-label, 112 terms, 1-4 labels per document)"
        ),
        text_field="text",
        label_field="topics",
        id_field="id",
        label_space=TOPICS,
        # macro_f1 keeps this comparable with the single-label classification
        # tasks and is the averaging that exposes rare-topic failure. Note
        # subset_accuracy is the metric with the most headroom - see
        # shelf.evaluate.metrics.multilabel for why all four averagings plus
        # the ranking metrics are reported together.
        primary_metric="macro_f1",
        secondary_metrics=(
            "micro_f1",
            "samples_f1",
            "weighted_f1",
            "subset_accuracy",
            "hamming_loss",
            "lrap",
            "map_micro",
        ),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    # -------------------------------------------------------------------------
    # CLUSTERING TASKS
    # -------------------------------------------------------------------------
    "lcc_clustering": TaskSpec(
        name="lcc_clustering",
        task_type=TaskType.CLUSTERING,
        description="Cluster documents into 21 subject groups",
        text_field="text",
        label_field="lcc_code",
        id_field="id",
        label_space=LCC_CODES,
        primary_metric="v_measure",
        secondary_metrics=("nmi", "ari"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    "lcgft_clustering": TaskSpec(
        name="lcgft_clustering",
        task_type=TaskType.CLUSTERING,
        description="Cluster documents into 14 genre categories",
        text_field="text",
        label_field="lcgft_category",
        id_field="id",
        label_space=LCGFT_CATEGORIES,
        primary_metric="v_measure",
        secondary_metrics=("nmi", "ari"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    "register_clustering": TaskSpec(
        name="register_clustering",
        task_type=TaskType.CLUSTERING,
        description="Cluster documents into 8 writing register/style groups",
        text_field="text",
        label_field="register",
        id_field="id",
        label_space=REGISTERS,
        primary_metric="v_measure",
        secondary_metrics=("nmi", "ari"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    "geographic_clustering": TaskSpec(
        name="geographic_clustering",
        task_type=TaskType.CLUSTERING,
        description="Cluster documents into 8 geographic regions based on content",
        text_field="text",
        label_field="geographic_region",  # Requires preprocessing with geographic.py
        id_field="id",
        label_space=GEOGRAPHIC_REGIONS,
        primary_metric="v_measure",
        secondary_metrics=("nmi", "ari"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    # -------------------------------------------------------------------------
    # CLUSTERING TASKS - HDBSCAN DISCOVERY (no fixed k)
    # -------------------------------------------------------------------------
    "lcc_clustering_hdbscan": TaskSpec(
        name="lcc_clustering_hdbscan",
        task_type=TaskType.CLUSTERING,
        description="Discover document clusters using HDBSCAN (no fixed k)",
        text_field="text",
        label_field="lcc_code",
        id_field="id",
        label_space=LCC_CODES,  # Used for evaluation, not clustering
        primary_metric="v_measure",
        secondary_metrics=("nmi", "ari", "noise_ratio", "cluster_k_error"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    "lcgft_clustering_hdbscan": TaskSpec(
        name="lcgft_clustering_hdbscan",
        task_type=TaskType.CLUSTERING,
        description="Discover genre clusters using HDBSCAN (no fixed k)",
        text_field="text",
        label_field="lcgft_category",
        id_field="id",
        label_space=LCGFT_CATEGORIES,
        primary_metric="v_measure",
        secondary_metrics=("nmi", "ari", "noise_ratio", "cluster_k_error"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    "register_clustering_hdbscan": TaskSpec(
        name="register_clustering_hdbscan",
        task_type=TaskType.CLUSTERING,
        description="Discover register clusters using HDBSCAN (no fixed k)",
        text_field="text",
        label_field="register",
        id_field="id",
        label_space=REGISTERS,
        primary_metric="v_measure",
        secondary_metrics=("nmi", "ari", "noise_ratio", "cluster_k_error"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    "geographic_clustering_hdbscan": TaskSpec(
        name="geographic_clustering_hdbscan",
        task_type=TaskType.CLUSTERING,
        description="Discover geographic clusters using HDBSCAN (no fixed k)",
        text_field="text",
        label_field="geographic_region",
        id_field="id",
        label_space=GEOGRAPHIC_REGIONS,
        primary_metric="v_measure",
        secondary_metrics=("nmi", "ari", "noise_ratio", "cluster_k_error"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    # -------------------------------------------------------------------------
    # CLUSTERING TASKS - AGGLOMERATIVE WITH COSINE DISTANCE
    # -------------------------------------------------------------------------
    "lcc_clustering_agglomerative": TaskSpec(
        name="lcc_clustering_agglomerative",
        task_type=TaskType.CLUSTERING,
        description="Cluster documents using agglomerative clustering with cosine distance",
        text_field="text",
        label_field="lcc_code",
        id_field="id",
        label_space=LCC_CODES,
        primary_metric="v_measure",
        secondary_metrics=("nmi", "ari"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    "lcgft_clustering_agglomerative": TaskSpec(
        name="lcgft_clustering_agglomerative",
        task_type=TaskType.CLUSTERING,
        description="Cluster genres using agglomerative clustering with cosine distance",
        text_field="text",
        label_field="lcgft_category",
        id_field="id",
        label_space=LCGFT_CATEGORIES,
        primary_metric="v_measure",
        secondary_metrics=("nmi", "ari"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    "register_clustering_agglomerative": TaskSpec(
        name="register_clustering_agglomerative",
        task_type=TaskType.CLUSTERING,
        description="Cluster registers using agglomerative clustering with cosine distance",
        text_field="text",
        label_field="register",
        id_field="id",
        label_space=REGISTERS,
        primary_metric="v_measure",
        secondary_metrics=("nmi", "ari"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    "geographic_clustering_agglomerative": TaskSpec(
        name="geographic_clustering_agglomerative",
        task_type=TaskType.CLUSTERING,
        description="Cluster geography using agglomerative clustering with cosine distance",
        text_field="text",
        label_field="geographic_region",
        id_field="id",
        label_space=GEOGRAPHIC_REGIONS,
        primary_metric="v_measure",
        secondary_metrics=("nmi", "ari"),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    # -------------------------------------------------------------------------
    # PAIR CLASSIFICATION TASKS
    # -------------------------------------------------------------------------
    "same_lcc_pairs": TaskSpec(
        name="same_lcc_pairs",
        task_type=TaskType.PAIR_CLASSIFICATION,
        description="Predict whether two documents share the same LCC class",
        text_field="text",  # Will need special handling for pairs
        label_field="label",
        id_field="pair_id",
        label_space=("0", "1"),
        primary_metric="auc_roc",
        secondary_metrics=("average_precision", "f1", "accuracy"),
        dataset_name="mjbommar/SHELF",
        dataset_config="same_lcc_pairs",
        default_split="test",
    ),
    "same_form_pairs": TaskSpec(
        name="same_form_pairs",
        task_type=TaskType.PAIR_CLASSIFICATION,
        description="Predict whether two documents share the same LCGFT form",
        text_field="text",  # Will need special handling for pairs
        label_field="label",
        id_field="pair_id",
        label_space=("0", "1"),
        primary_metric="auc_roc",
        secondary_metrics=("average_precision", "f1", "accuracy"),
        dataset_name="mjbommar/SHELF",
        dataset_config="same_form_pairs",
        default_split="test",
    ),
    "same_register_pairs": TaskSpec(
        name="same_register_pairs",
        task_type=TaskType.PAIR_CLASSIFICATION,
        description="Predict whether two documents share the same writing register/style",
        text_field="text",  # Will need special handling for pairs
        label_field="label",
        id_field="pair_id",
        label_space=("0", "1"),
        primary_metric="auc_roc",
        secondary_metrics=("average_precision", "f1", "accuracy"),
        dataset_name="mjbommar/SHELF",
        dataset_config="same_register_pairs",
        default_split="test",
    ),
    "same_audience_pairs": TaskSpec(
        name="same_audience_pairs",
        task_type=TaskType.PAIR_CLASSIFICATION,
        description="Predict whether two documents share the same target audience",
        text_field="text",  # Will need special handling for pairs
        label_field="label",
        id_field="pair_id",
        label_space=("0", "1"),
        primary_metric="auc_roc",
        secondary_metrics=("average_precision", "f1", "accuracy"),
        dataset_name="mjbommar/SHELF",
        dataset_config="same_audience_pairs",
        default_split="test",
    ),
    "same_topic_pairs": TaskSpec(
        name="same_topic_pairs",
        task_type=TaskType.PAIR_CLASSIFICATION,
        description="Predict whether two documents share ANY topic (binary multi-label overlap)",
        text_field="text",  # Will need special handling for pairs
        label_field="label",
        id_field="pair_id",
        label_space=("0", "1"),
        primary_metric="auc_roc",
        secondary_metrics=("average_precision", "f1", "accuracy"),
        dataset_name="mjbommar/SHELF",
        dataset_config="same_topic_pairs",
        default_split="test",
    ),
    # -------------------------------------------------------------------------
    # SUBJECT-CONDITIONAL CLUSTERING
    #
    # Flat k-means cannot isolate a low-variance attribute when subject
    # dominates embedding variance, so register and geography both score
    # V-measure ~0.004-0.009 for every model. Clustering *within* each LCC class
    # holds the dominant factor constant.
    #
    # The primary metric is `ari_pooled`, not V-measure, and that choice is
    # load-bearing. Conditional clustering makes each per-class problem smaller
    # and easier, so raw scores are not comparable to a flat task's. Against a
    # shuffled-label control (MiniLM, geography, unambiguous labels):
    #
    #   V-measure macro   real 0.2584   shuffled 0.0869   <- 34% structural
    #   ARI       pooled  real 0.0067   shuffled -0.0001
    #
    # V-measure is not chance-corrected and inflates badly here; ARI is.
    # -------------------------------------------------------------------------
    "geographic_clustering_conditional": TaskSpec(
        name="geographic_clustering_conditional",
        task_type=TaskType.CLUSTERING,
        description=(
            "Cluster geographic region within each LCC class, holding subject "
            "constant. Use GeographicLabelPolicy.UNAMBIGUOUS_ONLY: 38.5% of "
            "geographically-labelled documents otherwise carry an arbitrary "
            "region label taken from whichever tag happened to come first."
        ),
        text_field="text",
        label_field="geographic_region",
        id_field="id",
        label_space=GEOGRAPHIC_REGIONS,
        primary_metric="ari_pooled",
        secondary_metrics=(
            "ari_macro",
            "v_measure_pooled",
            "v_measure_macro",
            "n_classes_skipped",
        ),
        dataset_name="mjbommar/SHELF",
        dataset_config="default",
        default_split="test",
    ),
    "topic_overlap_pairs": TaskSpec(
        name="topic_overlap_pairs",
        task_type=TaskType.PAIR_CLASSIFICATION,
        # Note: Raw dataset labels are graded overlap levels 0-3 (see
        # `overlap_count`/`label` in the mjbommar/SHELF `topic_overlap_pairs`
        # config: {0: no shared topics, 1/2/3+: number of shared topics}).
        # `compute_pair_metrics` (shelf/evaluate/metrics/pair.py) binarizes
        # these internally (0 vs. non-0) at scoring time, but the label
        # space here reflects the actual ground-truth label cardinality.
        description="Predict whether two documents share any topics (binarized from 4-class overlap)",
        text_field="text",  # Will need special handling for pairs
        label_field="label",
        id_field="pair_id",
        label_space=("0", "1", "2", "3"),  # Raw overlap-count labels (0-3)
        primary_metric="auc_roc",
        secondary_metrics=("average_precision", "f1", "accuracy"),
        dataset_name="mjbommar/SHELF",
        dataset_config="topic_overlap_pairs",
        default_split="test",
    ),
}


def get_task(name: str) -> TaskSpec:
    """Get task specification by name.

    Args:
        name: Task name (e.g., "lcc_retrieval")

    Returns:
        TaskSpec for the requested task

    Raises:
        ValueError: If task name is not found
    """
    if name not in TASK_REGISTRY:
        available = sorted(TASK_REGISTRY.keys())
        raise ValueError(f"Unknown task: {name!r}. Available tasks: {available}")
    return TASK_REGISTRY[name]


def list_tasks(task_type: TaskType | None = None) -> list[str]:
    """List available task names.

    Args:
        task_type: Optional filter by task type

    Returns:
        List of task names
    """
    if task_type is None:
        return sorted(TASK_REGISTRY.keys())

    return sorted(
        name for name, spec in TASK_REGISTRY.items() if spec.task_type == task_type
    )


def list_retrieval_tasks() -> list[str]:
    """List all retrieval task names."""
    return list_tasks(TaskType.RETRIEVAL)


def list_instruction_retrieval_tasks() -> list[str]:
    """List the instruction-conditioned retrieval task names.

    These are retrieval tasks whose relevance is defined by an instruction
    rather than by a label field; see `shelf.evaluate.instructions`.
    """
    from shelf.evaluate.instructions import is_instruction_task

    return [
        name for name in list_tasks(TaskType.RETRIEVAL) if is_instruction_task(name)
    ]


def list_label_retrieval_tasks() -> list[str]:
    """List retrieval task names whose relevance comes from a label field."""
    from shelf.evaluate.instructions import is_instruction_task

    return [
        name for name in list_tasks(TaskType.RETRIEVAL) if not is_instruction_task(name)
    ]


def list_classification_tasks() -> list[str]:
    """List all classification task names."""
    return list_tasks(TaskType.CLASSIFICATION)


def list_clustering_tasks() -> list[str]:
    """List all clustering task names."""
    return list_tasks(TaskType.CLUSTERING)


def list_pair_tasks() -> list[str]:
    """List all pair classification task names."""
    return list_tasks(TaskType.PAIR_CLASSIFICATION)


def list_multilabel_tasks() -> list[str]:
    """List all multi-label classification task names."""
    return list_tasks(TaskType.MULTILABEL)


def list_clustering_kmeans_tasks() -> list[str]:
    """List k-means clustering task names (default algorithm)."""
    return [
        name
        for name in list_tasks(TaskType.CLUSTERING)
        if "_hdbscan" not in name and "_agglomerative" not in name
    ]


def list_clustering_hdbscan_tasks() -> list[str]:
    """List HDBSCAN clustering task names."""
    return [name for name in list_tasks(TaskType.CLUSTERING) if "_hdbscan" in name]


def list_clustering_agglomerative_tasks() -> list[str]:
    """List agglomerative clustering task names."""
    return [
        name for name in list_tasks(TaskType.CLUSTERING) if "_agglomerative" in name
    ]
