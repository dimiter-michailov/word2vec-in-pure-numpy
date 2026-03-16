# Word2Vec Results

## Current run
- Dataset(s) used in this run: `WikiText2-small.txt`
- Vocabulary size: `27228`
- Number of tokens produced from the dataset(s): `1694562`

### Parameters used in this run
- model: skipgram
- variant: hierarchical softmax
- embedding size: 80
- context size: 4
- epochs: 1

## Nearest Neighbours
A nearest-neighbour search starts with a query word, and for that query word, 
the model compares its word vector with every other learned word vector using cosine similarity.
The words with the highest cosine similarity are displayed below.
If the model learned meaningful structure, the displayed words should be related.

### york
- jersey (0.7512)
- zealand (0.5935)
- orleans (0.5663)
- yankees (0.5625)
- hampshire (0.5523)

### city
- building (0.5832)
- council (0.5829)
- state (0.5653)
- town (0.5614)
- centre (0.5489)

### season
- mlb (0.5895)
- ninth (0.5572)
- sixth (0.5540)
- postseason (0.5539)
- episode (0.5483)

### the
- in (0.5789)
- irish (0.5496)
- an (0.5330)
- whiskey (0.5301)
- of (0.5098)

### was
- is (0.6795)
- were (0.5631)
- by (0.5445)
- became (0.5422)
- are (0.5420)

### unk
- and (0.6597)
- in (0.5981)
- as (0.5735)
- irish (0.5545)
- gaelic (0.5531)

### of
- and (0.6235)
- in (0.5912)
- for (0.5763)
- irish (0.5695)
- with (0.5627)

### competitive
- players (0.5012)
- controller (0.4610)
- model (0.4596)
- rating (0.4461)
- pick (0.4312)

### association
- industry (0.6116)
- recording (0.5732)
- academy (0.5187)
- sports (0.5137)
- international (0.5062)

### football
- basketball (0.7117)
- hockey (0.7065)
- premier (0.6909)
- baseball (0.6769)
- league (0.6708)


## Datasets provided for analogy evaluation
- `text8.txt`: a cleaned Wikipedia text file often used for small word2vec experiments (100 MB)
- `WikiText103-train.txt`: a much larger, well-known Wikipedia-derived text used in this project as a stronger training dataset (539.2 MB)
- `WikiText2-small.txt`: the smallest Wikipedia-derived data included for quicker analogy-evaluation runs (10.8 MB)
- Analogy evaluation is only run when "perform analogy evaluation" is selected in `main.py`
- If the analogy evaluation was not chosen, the sections below should stay as `Not run in this execution.`

## Analogy Questions
An analogy question has the form:
`a : b :: c : d`

This means:
`a` is to `b` as `c` is to `d`

The model answers an analogy question by computing:
`vec(b) - vec(a) + vec(c)`
and then searching for the word whose embedding is closest to that result.

### What the evaluation numbers mean
- **questions in file**: how many analogy questions were listed in that category in the analogy file
- **questions asked**: how many of those questions could actually be evaluated on the model
- **correct**: how many asked questions were answered correctly
- **skipped**: how many questions were not evaluated because at least one required word was not found in the vocabulary set
- **accuracy**: `correct / questions asked`

## Custom analogies
This section is for `custom_analogies.txt`, which is my own analogy file. I followed the original analogy file format and created more basic questions that I considered suitable. More analogy questions could be added in the same format if desired.

### custom_analogies.txt

#### comparative
- questions in category: **4**
- questions asked: **4**
- skipped: **0**
- correct: **0**
- accuracy: **0.0000**

#### family
- questions in category: **5**
- questions asked: **5**
- skipped: **0**
- correct: **0**
- accuracy: **0.0000**

### Summary by analogy type

**Semantic**
- questions in file: **5**
- questions asked: **5**
- skipped: **0**
- correct: **0**
- accuracy: **0.0000**

**Syntactic**
- questions in file: **4**
- questions asked: **4**
- skipped: **0**
- correct: **0**
- accuracy: **0.0000**


## Google analogy questions
This section is for `google_analogies.txt`.

These are the standard analogy test questions discussed in the original word2vec paper and used to evaluate whether learned word vectors capture semantic and syntactic relationships.

When this evaluation is run, the report shows:
- per-category results
- an overall semantic accuracy
- an overall syntactic accuracy

### google_analogies.txt

#### capital-common-countries
- questions in category: **506**
- questions asked: **420**
- skipped: **86**
- correct: **6**
- accuracy: **0.0143**

#### capital-world
- questions in category: **4524**
- questions asked: **758**
- skipped: **3766**
- correct: **19**
- accuracy: **0.0251**

#### currency
- questions in category: **866**
- questions asked: **70**
- skipped: **796**
- correct: **0**
- accuracy: **0.0000**

#### city-in-state
- questions in category: **2467**
- questions asked: **1114**
- skipped: **1353**
- correct: **21**
- accuracy: **0.0189**

#### family
- questions in category: **506**
- questions asked: **342**
- skipped: **164**
- correct: **34**
- accuracy: **0.0994**

#### gram1-adjective-to-adverb
- questions in category: **992**
- questions asked: **702**
- skipped: **290**
- correct: **2**
- accuracy: **0.0028**

#### gram2-opposite
- questions in category: **812**
- questions asked: **272**
- skipped: **540**
- correct: **1**
- accuracy: **0.0037**

#### gram3-comparative
- questions in category: **1332**
- questions asked: **1056**
- skipped: **276**
- correct: **20**
- accuracy: **0.0189**

#### gram4-superlative
- questions in category: **1122**
- questions asked: **506**
- skipped: **616**
- correct: **3**
- accuracy: **0.0059**

#### gram5-present-participle
- questions in category: **1056**
- questions asked: **870**
- skipped: **186**
- correct: **18**
- accuracy: **0.0207**

#### gram6-nationality-adjective
- questions in category: **1599**
- questions asked: **1160**
- skipped: **439**
- correct: **79**
- accuracy: **0.0681**

#### gram7-past-tense
- questions in category: **1560**
- questions asked: **1482**
- skipped: **78**
- correct: **56**
- accuracy: **0.0378**

#### gram8-plural
- questions in category: **1332**
- questions asked: **756**
- skipped: **576**
- correct: **22**
- accuracy: **0.0291**

#### gram9-plural-verbs
- questions in category: **870**
- questions asked: **702**
- skipped: **168**
- correct: **5**
- accuracy: **0.0071**

### Summary by analogy type

**Semantic**
- questions in file: **8869**
- questions asked: **2704**
- skipped: **6165**
- correct: **80**
- accuracy: **0.0296**

**Syntactic**
- questions in file: **10675**
- questions asked: **7506**
- skipped: **3169**
- correct: **206**
- accuracy: **0.0274**
