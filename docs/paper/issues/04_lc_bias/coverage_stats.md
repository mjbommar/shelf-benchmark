# SHELF Geographic and Topic Coverage Statistics

## Dataset Summary

- **Total documents**: 42,616
- **Documents with geographic metadata**: 28,387 (66.6%)
- **Unique geographic regions**: 44
- **Continents/major regions**: 6
- **Unique LCC classes**: 21
- **Unique topics**: 112
- **Unique audiences**: 24
- **Unique registers**: 8

## Geographic Distribution

### By Continent/Major Region

| Continent/Region | Documents | Percentage |
|------------------|-----------|------------|
| North America    | 16,506    | 38.7%      |
| Asia             | 9,698     | 22.8%      |
| Europe           | 9,642     | 22.6%      |
| South America    | 4,809     | 11.3%      |
| Oceania          | 1,019     | 2.4%       |
| Africa           | 950       | 2.2%       |

**Analysis**: While North America dominates (reflecting U.S. academic library prevalence of LCC), the dataset includes substantial coverage of Asia (22.8%) and Europe (22.6%), with meaningful representation of South America, Oceania, and Africa.

### Top 20 Specific Regions

| Region            | Continent     | Documents |
|-------------------|---------------|-----------|
| United States     | North America | 1,029     |
| Florida           | North America | 1,027     |
| Australia         | Oceania       | 1,019     |
| Ohio              | North America | 1,014     |
| Germany           | Europe        | 1,013     |
| Central America   | South America | 1,012     |
| Japan             | Asia          | 1,009     |
| Italy             | Europe        | 1,002     |
| North Carolina    | North America | 999       |
| New York City     | North America | 998       |
| Southeast Asia    | Asia          | 996       |
| Tokyo             | Asia          | 993       |
| Canada            | North America | 990       |
| Russia            | Europe        | 984       |
| Caribbean         | South America | 983       |
| Pennsylvania      | North America | 981       |
| Beijing           | Asia          | 981       |
| Asia (general)    | Asia          | 981       |
| Los Angeles       | North America | 972       |
| Middle East       | Asia          | 969       |

**Analysis**: Near-uniform distribution across all 44 regions (886-1,029 docs per region, balance ratio 0.86), demonstrating intentional stratification strategy.

### Geographic Balance Metrics

- **Average docs per region**: 969
- **Min docs per region**: 886 (China)
- **Max docs per region**: 1,029 (United States)
- **Balance ratio (min/max)**: 0.86

**Interpretation**: 0.86 balance ratio indicates excellent uniformity—significantly better than natural corpus distribution, where U.S.-centric collections often dominate 80-90%.

## LCC Class Distribution by Geography

All 21 LCC classes appear across all 6 continents/major regions. Sample from the matrix:

| LCC | Africa | Asia | Europe | North America | Oceania | South America |
|-----|--------|------|--------|---------------|---------|---------------|
| A (General Works) | 41 | 502 | 419 | 796 | 50 | 227 |
| B (Philosophy, Psychology, Religion) | 53 | 490 | 453 | 790 | 57 | 226 |
| D (World History) | 52 | 472 | 511 | 839 | 56 | 230 |
| H (Social Sciences) | 43 | 489 | 449 | 769 | 48 | 230 |
| K (Law) | 46 | 432 | 496 | 774 | 51 | 222 |
| P (Language and Literature) | 39 | 484 | 453 | 782 | 60 | 225 |
| Q (Science) | 46 | 464 | 455 | 780 | 49 | 236 |
| R (Medicine) | 47 | 414 | 436 | 795 | 46 | 215 |
| T (Technology) | 49 | 509 | 450 | 830 | 55 | 247 |

**Key finding**: Complete cross-product coverage—every LCC class appears in every geographic region, demonstrating SHELF transcends typical genre-subject correlations found in real-world libraries.

## Topic Coverage

### Topic Distribution Statistics

- **Total unique topics**: 112
- **Most common topics**: Art (3,479), Music (3,462), History (3,451), Culture (3,448), Religion (3,445)
- **Specialized topics**: Range from Quantum mechanics (1,017) to Strategy (162)

### Topics with Broadest Geographic Distribution

All top topics appear across 6/7 continents:

| Topic      | Continents | Total Docs |
|------------|------------|------------|
| Music      | 6/7        | 3,506      |
| Art        | 6/7        | 3,494      |
| History    | 6/7        | 3,494      |
| Religion   | 6/7        | 3,486      |
| Culture    | 6/7        | 3,473      |
| Aesthetics | 6/7        | 3,456      |
| Literature | 6/7        | 3,445      |
| Languages  | 6/7        | 3,414      |
| Ethics     | 6/7        | 3,409      |
| Philosophy | 6/7        | 3,403      |

**Analysis**: Core humanities topics achieve near-universal geographic distribution, demonstrating SHELF's coverage of fundamental human intellectual activities across cultures.

### Non-Western Topic Coverage

#### Topics in Asian Contexts (top 15)

| Topic                 | Documents |
|-----------------------|-----------|
| Music                 | 810       |
| Art                   | 802       |
| History               | 802       |
| Religion              | 801       |
| Literature            | 788       |
| Languages             | 786       |
| Ethics                | 777       |
| Philosophy            | 775       |
| Culture               | 770       |
| Aesthetics            | 747       |
| Public administration | 370       |
| Democracy             | 367       |
| Authoritarianism      | 359       |
| Government            | 356       |
| Political parties     | 350       |

**Significance**: Asia receives both traditional humanities coverage (Art, Music, Literature) AND contemporary political/social science topics (Democracy, Authoritarianism, Government), countering claims that non-Western regions receive only "exotic" cultural treatment.

#### Topics in African Contexts (top 15)

| Topic        | Documents |
|--------------|-----------|
| Music        | 94        |
| Languages    | 87        |
| Philosophy   | 87        |
| Art          | 86        |
| Aesthetics   | 86        |
| Culture      | 86        |
| Religion     | 78        |
| History      | 78        |
| Ethics       | 74        |
| Literature   | 67        |
| Security     | 47        |
| Elections    | 37        |
| Diplomacy    | 36        |
| Defense      | 35        |
| Nationalism  | 35        |

**Significance**: Africa receives comprehensive coverage including Philosophy, Aesthetics, Ethics (intellectual traditions) alongside contemporary political topics (Elections, Security, Diplomacy).

## Audience Distribution

| Audience              | Documents |
|-----------------------|-----------|
| Lawyers               | 1,321     |
| Physicians            | 1,320     |
| Experts               | 1,311     |
| Business professionals| 1,310     |
| Scholars              | 1,287     |
| Policy makers         | 1,258     |
| Adults                | 1,253     |
| Lay readers           | 1,252     |
| Educators             | 1,245     |
| Children              | 1,243     |
| General public        | 1,241     |
| Professionals         | 1,241     |
| Young adults          | 1,240     |
| Specialists           | 1,239     |
| Graduate students     | 1,238     |
| Beginners             | 1,237     |
| Non-specialists       | 1,234     |
| Researchers           | 1,229     |
| Scientists            | 1,220     |
| Older adults          | 1,217     |
| Practitioners         | 1,212     |
| Adolescents           | 1,203     |
| Engineers             | 1,194     |
| Students              | 1,190     |

**Analysis**: Near-perfect balance across 24 audience types (1,190-1,321 docs, range 131). Covers full spectrum from Children to Experts, General public to Specialists.

## Register Distribution

| Register        | Documents |
|-----------------|-----------|
| Professional    | 10,467    |
| Formal          | 6,432     |
| Conversational  | 6,482     |
| Academic        | 6,382     |
| Casual          | 4,323     |
| Technical       | 4,282     |
| Journalistic    | 2,163     |
| Creative        | 2,085     |

**Analysis**: Spans full range of writing styles from Creative and Casual to Academic and Technical, reflecting diverse document types in real-world bibliographic collections.

## Cross-Product Diversity Examples

### Unexpected LCC + Form + Geography Combinations

These examples demonstrate SHELF's "more diverse than reality" claim:

- **Political Science + Jokes + Pennsylvania** (5 docs)
- **Fine Arts + Infographics + New York** (5 docs)
- **Medicine + Lectures + Italy** (5 docs)
- **Political Science + Games + Georgia** (5 docs)
- **Bibliography, Library Science + Architectural drawings + New York City** (5 docs)
- **Political Science + Lectures + Middle East** (5 docs)
- **Social Sciences + Jokes + Japan** (5 docs)
- **Medicine + Tributes + Russia** (5 docs)
- **Auxiliary Sciences of History + Aerial photographs + South Korea** (5 docs)
- **Fine Arts + Puzzles + Europe** (5 docs)
- **Geography, Anthropology, Recreation + Nautical charts + Southeast Asia** (4 docs)
- **Law + Lectures + Illinois** (4 docs)

**Key insight**: Real-world libraries exhibit strong genre-subject correlations (e.g., Medical texts are rarely Jokes, Fine Arts rarely appears as Games). SHELF's synthetic approach creates cross-product combinations that test document understanding across unconventional pairings.

## Evidence of Universality

### 1. Geographic Universality

- **6/7 continents represented** (all except Antarctica)
- **44 specific regions** spanning global scope
- **Balance ratio 0.86**: Near-uniform distribution across regions
- **22.8% Asia, 22.6% Europe, 11.3% South America**: Substantial non-North American coverage

### 2. Topical Universality

- **112 unique topics** spanning all LCC domains
- **Core humanities topics** appear in all continents (Art, Music, Philosophy, Religion, Literature)
- **Contemporary topics** appear globally (Democracy, Security, Climate change, Biotechnology)
- **Specialized topics** achieve global distribution (Quantum mechanics, Oncology, Nanotechnology)

### 3. Functional Universality

- **133 LCGFT forms**: From Lectures to Satellite imagery to Jokes to Legal briefs
- **24 audiences**: Children to Experts, General public to Specialists
- **8 registers**: Creative to Academic, Casual to Technical

### 4. Cross-Product Independence

- **Every LCC class appears with every geography** (21 × 6 = full matrix coverage)
- **Unconventional combinations exist** (Jokes about Law, Maps about Philosophy)
- **More diverse than real corpora** where genre-subject correlations constrain combinations

## Comparison to Real-World Bias Patterns

### Known LC System Biases (from literature)

1. **Religious bias**: Christianity receives multiple subclasses; other religions marginalized
2. **Geographic bias**: European/American history receives detailed classification
3. **Racial terminology**: Outdated terms for African American, Indigenous peoples
4. **LGBTQ+ bias**: Harmful categorization under "Sexual Deviations"
5. **Gender bias**: Male-default assumptions (4,605 "women" terms vs. 444 "men")

### How SHELF Mitigates These Biases

1. **Balanced LCC distribution**: All 21 classes receive near-equal representation (4.6-4.9%)
2. **Global topic generation**: Religion, Philosophy, Ethics appear in Asian, African, European contexts
3. **Contemporary terminology**: Synthetic generation uses modern language conventions
4. **Cross-cultural application**: LCGFT forms (Prayers, Lectures, Games) applied across all geographies
5. **Independent dimensions**: Topics, geographies, forms, audiences generated independently

### What SHELF Cannot Fix

1. **Inherited classification structure**: Still uses LCC's hierarchical Western epistemology
2. **Subclass-level granularity**: Christianity still has more detailed subclasses than Buddhism
3. **Terminology in original taxonomies**: Some LCSH/LCDGT terms remain outdated
4. **Structural assumptions**: Mutually exclusive categories, linear sequences

## Conclusion

SHELF achieves "universal" coverage in the sense of:

1. **Comprehensive scope**: All 21 LCC classes, 133 forms, 112 topics, 44 geographies
2. **Balanced distribution**: Intentional stratification creates near-uniform coverage
3. **Global reach**: Substantial non-Western geographic and topical representation
4. **Cross-product diversity**: More diverse than real-world corpora due to independent dimension sampling

SHELF does NOT claim:

1. **Cultural neutrality**: LCC framework is Western-centric
2. **Bias-free taxonomy**: Inherited limitations from LC system
3. **Equal global weighting**: North America still dominates (38.7% vs. 2.2% Africa)
4. **Epistemological universality**: Hierarchical structure reflects Western thinking

The benchmark is "universal" within the Western bibliographic tradition, with active mitigation of structural biases through balanced synthetic generation—making it more globally representative than typical U.S. academic library collections while acknowledging irreducible limitations of the underlying LC taxonomy.
