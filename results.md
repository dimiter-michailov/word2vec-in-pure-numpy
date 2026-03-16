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
- epochs: 3

## Nearest Neighbours
A nearest-neighbour search starts with a query word, and for that query word, 
the model compares its word vector with every other learned word vector using cosine similarity.
The words with the highest cosine similarity are displayed below.
If the model learned meaningful structure, the displayed words should be related.

### york
- jersey (0.7517)
- yankees (0.6169)
- knicks (0.6167)
- orleans (0.5611)
- zealand (0.5440)

### city
- lambton (0.5520)
- town (0.5405)
- dominion (0.5378)
- council (0.5376)
- southwestern (0.5334)

### f
- c (0.7367)
- nf (0.5714)
- xe (0.5557)
- cl (0.5511)
- xeof (0.5507)

### c
- f (0.7367)
- g (0.6427)
- nf (0.6262)
- e (0.6189)
- cl (0.5796)

### season
- sixth (0.5715)
- episode (0.5604)
- dreamscape (0.5524)
- seasons (0.5488)
- game (0.5455)

### the
- in (0.6204)
- irish (0.6000)
- of (0.5974)
- whiskey (0.5967)
- unk (0.5733)

### was
- is (0.6616)
- in (0.6094)
- by (0.5688)
- became (0.5630)
- are (0.5605)

### unk
- and (0.6983)
- gaelic (0.6286)
- in (0.6134)
- as (0.6128)
- whiskey (0.6080)

### of
- and (0.7030)
- with (0.6428)
- in (0.6352)
- irish (0.5987)
- the (0.5974)

### competitive
- coaching (0.5270)
- qualification (0.5004)
- score (0.4941)
- liverpool (0.4910)
- match (0.4850)


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
This section is for `custom_analogies.txt`, which is my own analogy file. I followed the original analogy file format and created more basic questions that I considered suitable.

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
- correct: **10**
- accuracy: **0.0238**

#### capital-world
- questions in category: **4524**
- questions asked: **758**
- skipped: **3766**
- correct: **24**
- accuracy: **0.0317**

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
- correct: **29**
- accuracy: **0.0260**

#### family
- questions in category: **506**
- questions asked: **342**
- skipped: **164**
- correct: **24**
- accuracy: **0.0702**

#### gram1-adjective-to-adverb
- questions in category: **992**
- questions asked: **702**
- skipped: **290**
- correct: **4**
- accuracy: **0.0057**

#### gram2-opposite
- questions in category: **812**
- questions asked: **272**
- skipped: **540**
- correct: **4**
- accuracy: **0.0147**

#### gram3-comparative
- questions in category: **1332**
- questions asked: **1056**
- skipped: **276**
- correct: **15**
- accuracy: **0.0142**

#### gram4-superlative
- questions in category: **1122**
- questions asked: **506**
- skipped: **616**
- correct: **10**
- accuracy: **0.0198**

#### gram5-present-participle
- questions in category: **1056**
- questions asked: **870**
- skipped: **186**
- correct: **24**
- accuracy: **0.0276**

#### gram6-nationality-adjective
- questions in category: **1599**
- questions asked: **1160**
- skipped: **439**
- correct: **83**
- accuracy: **0.0716**

#### gram7-past-tense
- questions in category: **1560**
- questions asked: **1482**
- skipped: **78**
- correct: **78**
- accuracy: **0.0526**

#### gram8-plural
- questions in category: **1332**
- questions asked: **756**
- skipped: **576**
- correct: **15**
- accuracy: **0.0198**

#### gram9-plural-verbs
- questions in category: **870**
- questions asked: **702**
- skipped: **168**
- correct: **11**
- accuracy: **0.0157**

### Summary by analogy type

**Semantic**
- questions in file: **8869**
- questions asked: **2704**
- skipped: **6165**
- correct: **87**
- accuracy: **0.0322**

**Syntactic**
- questions in file: **10675**
- questions asked: **7506**
- skipped: **3169**
- correct: **244**
- accuracy: **0.0325**
