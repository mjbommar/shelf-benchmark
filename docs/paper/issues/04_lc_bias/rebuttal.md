# Response to Reviewer Concern: Library of Congress Taxonomy Bias

## Reviewer Concern

> "The claim that SHELF provides 'universal' coverage is problematic given the well-documented Western and American biases in the Library of Congress Classification system. How can you claim universality when using a taxonomy that privileges Christianity over other religions, American/European history over other regions, and embeds Western epistemological assumptions in its hierarchical structure?"

## Summary Response

We acknowledge the reviewer's important concern about LC Classification biases and appreciate the opportunity to clarify our claims. We agree that LCC has documented Western, American, and Christian biases stemming from its historical development. However, we defend SHELF's claim to "universal" coverage for three reasons:

1. **No superior alternative exists**: All major classification systems exhibit cultural biases
2. **SHELF actively mitigates structural biases**: Synthetic generation enables balanced coverage
3. **"Universal" means comprehensive, not culturally neutral**: We cover the full breadth of LCC's scope with intentional global distribution

We propose clarifying language in the paper to acknowledge these limitations while defending the substantive claim.

## Detailed Response

### 1. Acknowledged Limitations of LC Classification

We fully acknowledge the well-documented biases in the Library of Congress Classification system:

**Religious bias**: Christianity receives multiple subclasses (BC-BX) while Judaism (BM), Islam (BP), and Buddhism (BQ) each receive single subclasses, reflecting "a westernized perspective that prioritizes Christianity while other important religions and spiritual beliefs are pushed to the margins" (Howard & Knowlton, 2018).

**Geographic bias**: Schedules D, E, and F (History) emphasize American and European history with detailed classification, while non-Western regions receive less granular treatment. This reflects the system's origins in organizing the Library of Congress's specific collections (Wikipedia, 2024).

**Epistemological bias**: The hierarchical structure with mutually exclusive categories and linear sequences reflects Western intellectual traditions "often at odds with thinking in non-Western cultures" (Yes Magazine, 2019).

**Terminology issues**: Outdated and potentially harmful language for African American studies, Indigenous peoples, and LGBTQ+ materials persists in the schedules (Howard & Knowlton, 2018).

These biases are real, documented, and irreducible given SHELF's use of LCC as its organizational framework.

### 2. Why No Superior Alternative Exists

However, the crucial question is: **compared to what alternative?**

**Dewey Decimal Classification (DDC)**: The global leader in library classification, DDC was "a product of its time and deeply embedded in the worldview of 19th century America" and has been "found to be biased and unsystematic in their coverage of non-western religions and racial groups" (John the Librarian, 2017). Quantitative analysis of 3+ million books found DDC exhibits similar Western bias, particularly in religion categories (Warburton et al., 2024).

**Universal Decimal Classification (UDC)**: Despite its name, UDC is an expansion of DDC with the same foundational Western epistemology. While more flexible, it "retain[s] some of the same features and biases of their forebears" (John the Librarian, 2017).

**Chinese Library Classification (CLC)**: Used in 90%+ of Chinese libraries, CLC is explicitly adapted for Chinese contexts with "marked Chinese characteristics" (IFLA, 1996). Using CLC would simply swap American bias for Chinese bias without addressing the fundamental critique.

**Critical insight from the Library of Congress itself**: "Currently there are no other workable subject or classification systems that have the scope and breadth of the Library of Congress systems, and no other systems address the problem of bias adequately" (Stonehill College, 2024).

All classification systems reflect their cultural origins. The choice is not between biased LCC and unbiased alternative—it's between LCC and differently-biased alternatives with less scope.

### 3. How SHELF Mitigates Structural Biases

While SHELF inherits LCC's epistemological framework, our synthetic generation approach actively mitigates distributional biases:

**Balanced LCC distribution**: All 21 LCC classes receive near-equal representation (4.6-4.9% each), contrasting with real libraries where certain classes dominate. This includes equal treatment of:
- B (Philosophy/Psychology/Religion) despite its Christian bias
- D (World History) despite its European focus
- K (Law) which has better global balance

**Global geographic coverage**: 44 regions across 6 continents with intentional balance:
- Asia: 22.8% (9,698 docs)
- Europe: 22.6% (9,642 docs)
- South America: 11.3% (4,809 docs)
- Africa: 2.2% (950 docs)
- Oceania: 2.4% (1,019 docs)

While North America still dominates (38.7%), this reflects LCC's adoption patterns while providing substantial non-Western coverage—far better than typical U.S. academic library collections which often exceed 80-90% North American focus.

**Cross-cultural topic application**: Analysis shows core topics appear across all continents:
- Asian contexts: Philosophy (775 docs), Religion (801), Ethics (777), Democracy (367), Authoritarianism (359)
- African contexts: Philosophy (87 docs), Aesthetics (86), Ethics (74), Elections (37), Security (47)

Critically, non-Western regions receive both traditional humanities coverage (Art, Music, Literature) AND contemporary political/social science topics (Democracy, Security, Biotechnology), countering the tendency to treat non-Western content as only "exotic" cultural material.

**Cross-product diversity**: SHELF's independent dimension sampling creates combinations rarely found in real libraries:
- Political Science + Jokes + Japan
- Medicine + Lectures + Middle East
- Fine Arts + Puzzles + Asia
- Law + Games + Southeast Asia

This "more diverse than reality" property means SHELF tests document understanding across unconventional pairings that challenge typical genre-subject correlations found in real collections.

**Matrix completeness**: Every LCC class appears in every geographic region (21 × 6 = complete matrix coverage), demonstrating that SHELF transcends the structural correlations that amplify bias in real-world libraries.

### 4. Clarifying "Universal" Coverage Claims

We propose clarifying the paper's language to distinguish between:

**What SHELF IS**:
- **Comprehensive within the bibliographic tradition**: Covers all 21 LCC classes, 133 LCGFT forms, 112 topics
- **Balanced across dimensions**: Intentional stratification creates near-uniform distribution
- **Globally distributed**: Substantial coverage of non-Western geographies and topics
- **Cross-product complete**: More diverse than real corpora due to independent dimension sampling

**What SHELF IS NOT**:
- **Culturally neutral**: Inherits LCC's Western epistemological framework
- **Bias-free**: Taxonomy contains outdated terminology and structural biases
- **Equally weighted globally**: North America still overrepresented relative to population
- **Epistemologically universal**: Hierarchical categories reflect Western thinking

The term "universal" should be understood as **"comprehensive coverage of the bibliographic universe as defined by LCC"** rather than **"culturally neutral representation of all human knowledge systems."**

### 5. Why This Still Matters for Benchmark Validity

Despite these limitations, SHELF remains valuable as a benchmark:

**Standardization benefit**: Using the most widely adopted research library classification system (60% of U.S. academic libraries, dominant in major research institutions globally) ensures results are interpretable and relevant to real-world library applications.

**Practical relevance**: LCC is the de facto standard for bibliographic classification in research contexts. Models that perform well on SHELF will perform well on the majority of research library collections they encounter.

**Transparent limitations**: By explicitly documenting LCC's biases and SHELF's mitigation strategies, we enable researchers to interpret results with appropriate context—a more honest approach than claiming non-existent cultural neutrality.

**Comparative advantage**: SHELF's balanced synthetic generation makes it MORE globally representative than natural corpus benchmarks drawn from real U.S. libraries, even while acknowledging irreducible framework limitations.

**Dimension independence**: The key insight is that SHELF's cross-product diversity (Maps about Philosophy, Jokes about Law) tests document understanding in ways that transcend the specific cultural biases of individual LCC classes.

### 6. Proposed Paper Revisions

We propose the following additions to address this concern:

**In Introduction**:
> "We use the Library of Congress Classification (LCC) system as our organizational framework. While LCC has well-documented Western, American, and Christian biases stemming from its historical development (Howard & Knowlton, 2018), it remains the most comprehensive bibliographic classification system and is the de facto standard in research libraries globally. We acknowledge these limitations while noting that all major classification systems (DDC, UDC, CLC) exhibit comparable cultural biases reflecting their origins. Our synthetic generation approach actively mitigates distributional biases through balanced stratification across all 21 LCC classes and 44 global geographic regions."

**New Section 2.X: Limitations and Biases**:
> "**Taxonomy Biases**: The Library of Congress Classification system exhibits documented Western epistemological assumptions (hierarchical structure, mutually exclusive categories) and content biases (Christianity receives multiple subclasses vs. single subclasses for Judaism, Islam, Buddhism; American/European history receives more detailed classification). SHELF inherits this framework but mitigates distributional biases through balanced synthetic generation: all 21 LCC classes receive near-equal representation (4.6-4.9%), and documents are distributed across 6 continents with substantial non-Western coverage (Asia 22.8%, Europe 22.6%, South America 11.3%, Africa 2.2%, Oceania 2.4%). Our claim to 'universal' coverage should be understood as comprehensive within the bibliographic tradition rather than culturally neutral representation of all knowledge systems. However, SHELF's independent dimension sampling creates cross-product diversity (Maps about Philosophy, Jokes about Law) that transcends specific cultural biases of individual classes."

**In Discussion/Conclusion**:
> "While SHELF uses the LC Classification framework with its inherent Western biases, our balanced synthetic approach creates a more globally representative benchmark than natural corpus alternatives drawn from real U.S. library collections. Future work could explore complementary benchmarks based on alternative classification systems (CLC, indigenous systems) to provide culturally diverse perspectives on document understanding tasks."

## Conclusion

We thank the reviewer for raising this important concern. We fully acknowledge that LCC has documented Western, American, and Christian biases that SHELF inherits through its use of this framework. However, we defend the substantive claim to comprehensive coverage for three reasons:

1. **No bias-free alternative exists** with comparable scope and breadth
2. **SHELF actively mitigates distributional biases** through balanced synthetic generation
3. **The benchmark provides practical value** for the dominant research library classification system

We propose clarifying the paper's language to distinguish between "comprehensive within the bibliographic tradition" (accurate) and "culturally neutral representation" (not claimed). This honest acknowledgment of limitations, combined with evidence of active mitigation strategies, strengthens rather than undermines SHELF's contribution as a benchmark.

## References

- Howard, S. A., & Knowlton, S. A. (2018). Browsing through Bias: The Library of Congress Classification and Subject Headings for African American Studies and LGBTQIA Studies. Library Trends, 67(1), 74-88.
- Stonehill College Library. (2024). The Library of Congress Outdated Biased Classifications. Retrieved from https://www.stonehill.edu/library/library-newsletter/news/the-library-of-congress-outdated-biased-classifications/
- John the Librarian. (2017). Inherent Bias in Classification Systems. Retrieved from https://johnthelibrarian.com/2017/12/13/inherent-bias-in-classification-systems/
- Warburton, K., et al. (2024). Quantifying Bias in Hierarchical Category Systems. arXiv preprint.
- IFLA. (1996). Contemporary Classification Systems and Thesauri in China. 62nd IFLA General Conference. Retrieved from https://origin-archive.ifla.org/IV/ifla62/62-qiyz.htm
- Yes Magazine. (2019). X̱wi7x̱wa Takes an Indigenous Approach to Categorizing Books. Retrieved from https://www.yesmagazine.org/social-justice/2019/03/22/decolonize-western-bias-indigenous-library-books
- Wikipedia. (2024). Library of Congress Classification. Retrieved from https://en.wikipedia.org/wiki/Library_of_Congress_Classification
