# Text Similarity in News Articles - Report

## Dataset

Number of articles: 9

| Article | Total tokens | Unique tokens |
| --- | --- | --- |
| acq_01 | 231 | 160 |
| acq_02 | 134 | 91 |
| acq_03 | 132 | 97 |
| crude_01 | 163 | 111 |
| crude_02 | 138 | 77 |
| crude_03 | 81 | 57 |
| grain_01 | 134 | 95 |
| grain_02 | 134 | 74 |
| grain_03 | 98 | 78 |

## Most Similar Pairs

### Jaccard

| Article A | Article B | Similarity |
| --- | --- | --- |
| grain_01 | grain_02 | 0.105 |
| crude_01 | crude_02 | 0.056 |
| acq_02 | acq_03 | 0.044 |

### Cosine

| Article A | Article B | Similarity |
| --- | --- | --- |
| grain_01 | grain_02 | 0.342 |
| grain_02 | grain_03 | 0.271 |
| grain_01 | grain_03 | 0.207 |

## Least Similar Pairs

### Jaccard

| Article A | Article B | Similarity |
| --- | --- | --- |
| acq_03 | crude_03 | 0.000 |
| acq_01 | crude_02 | 0.004 |
| acq_01 | crude_03 | 0.005 |

### Cosine

| Article A | Article B | Similarity |
| --- | --- | --- |
| acq_03 | crude_03 | 0.000 |
| crude_02 | grain_02 | 0.002 |
| acq_01 | crude_03 | 0.004 |

## Metric Comparison

- The most similar pair is the same for both metrics: grain_01 - grain_02.
- Pearson correlation between Jaccard and cosine: 0.788

## Limitations

- Similarity is lexical (shared words), not semantic: synonyms and paraphrases are not detected.
- Preprocessing removes accented letters and the stop word "won".
- The dataset contains only a few articles.
- Results depend on the length of each text.
- Custom domain stop words were added for newswire boilerplate.

## Interpretation

**Overall pattern.** Cosine similarity separates the three topics well in this dataset: the nine highest-scoring pairs are exactly the nine same-topic pairs, so every same-topic pair scores above every cross-topic pair. Grain is the clearest cluster, with the three highest values overall (grain_01-grain_02 at 0.342, grain_02-grain_03 at 0.271 and grain_01-grain_03 at 0.207). Crude and acq pairs score lower in absolute terms, and the gap between the weakest same-topic pair and the strongest cross-topic pair is small (0.015), so the separation is clear in the ranking but not in magnitude.

**Least similar pairs.** All six rows in the least-similar tables are cross-topic pairs, mostly acq-crude combinations, with scores between 0.000 and 0.005. crude_03 appears in four of those six rows and acq_01 in three. crude_03 is the shortest document (81 tokens after preprocessing) and acq_01 the longest (231), so text length probably contributes to these near-zero values, although this report does not test that directly.

**Jaccard vs. cosine.** Both metrics agree on the most similar pair (grain_01-grain_02), and a Pearson correlation of 0.788 shows they rank pairs in a broadly similar way, but not identically. Jaccard scores are much smaller (maximum 0.105 against 0.342) because it only compares sets of unique tokens and ignores how often a word is repeated, and those sets are small in short articles. Cosine similarity uses term frequencies, so repeated topical words weigh more, which is consistent with all of its top pairs being grain articles. Jaccard's top three, in contrast, include one pair from each topic.

**Caveats.** The dataset has only 9 articles of 81 to 231 tokens after preprocessing, so a few shared words can move a score noticeably, and these observations are not statistically tested conclusions. Similarity is lexical, so articles about the same event written with different words would score low. The articles come from the Reuters corpus in NLTK. The first selection used only the corpus labels and produced topics that were not coherent (for example, "crude" articles about a Turkey-Greece dispute), so keyword and numeric-content filters were added afterwards, and the results depend on those criteria. The keyword filter guarantees that each article contains words from its topic, which favours high within-topic similarity, so this clean separation is partly a product of the selection and should not be read as how the method would perform on unfiltered news. Finally, grain_03 comes from the corpus training split, because too few test-split grain articles passed the filters.
