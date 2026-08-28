# SHELF human validation — coding instructions

You are assigning library classification labels to 399 documents. Work
independently: do not discuss any document with the other coder until both of
you have finished. Disagreement is data, not a problem to resolve in advance.

## What you are labelling

For each document, record two labels.

**1. LCC class** — one letter from the Library of Congress Classification:

- `A` — general reference, encyclopedias, journalism, museums
- `B` — philosophy, psychology, ethics, religion, spirituality
- `C` — historical sciences, archaeology, genealogy, biography
- `D` — world history, ancient civilizations, modern nations, wars
- `E` — American history, United States, colonial era, civil war
- `F` — Americas history, Canada, Latin America, local US history
- `G` — geography, maps, anthropology, folklore, sports, recreation
- `H` — social sciences, economics, sociology, statistics, commerce
- `J` — government, politics, policy, elections, political systems, international relations
- `K` — law, legal systems, courts, legislation, constitutional law
- `L` — education, schools, teaching, curriculum, higher education
- `M` — music, musical instruments, compositions, music theory
- `N` — visual arts, painting, sculpture, architecture, photography
- `P` — language, linguistics, literature, fiction, poetry, drama
- `Q` — science, mathematics, physics, chemistry, biology, astronomy
- `R` — medicine, healthcare, diseases, anatomy, nursing, public health
- `S` — agriculture, farming, crops, livestock, forestry, fishing
- `T` — technology, engineering, manufacturing, construction, crafts
- `U` — military science, armies, warfare, defense, veterans
- `V` — naval science, navies, ships, maritime, coast guard
- `Z` — bibliography, libraries, publishing, book history, information science

**2. LCGFT category** — one of the fourteen Library of Congress Genre/Form
categories:

- Cartographic materials
- Commemorative works
- Creative nonfiction
- Discursive works
- Ephemera
- Informational works
- Instructional and educational works
- Law materials
- Literature
- Music
- Recreational works
- Religious materials
- Sound recordings
- Visual works

## How to decide

- Judge the document **as it is**, not what it might have been. Classify what is
  on the page.
- LCC asks *what is this about*. LCGFT asks *what kind of thing is this*. They
  are independent: a joke about military science is `U` and `Recreational
  works`. Unusual combinations are deliberate and common in this corpus.
- Titles are omitted on purpose. Classify from the body alone.
- If two labels seem equally defensible, pick one and mark `uncertain` = `y`.
  Do not leave a cell blank. A forced choice plus an uncertainty flag is far
  more informative than a gap.
- Spend roughly a minute per document. First reading is usually right; this
  measures achievable agreement, not maximum effort.

## Documents that are not classifiable

Some documents may be too short, incoherent, or not English. Mark
`unusable` = `y` and still record your best-guess labels. The rate at which this
happens is itself a corpus-quality measurement.

## Returning your work

Fill in `coding_sheet.csv` and return it unchanged apart from your entries.
Do not reorder or delete rows.
