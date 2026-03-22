# Word2Vec Results

## Current run
- Dataset(s) used in this run: `text8.txt`
- Vocabulary size: `253854`
- Number of tokens produced from the dataset(s): `17005207`

### Parameters used in this run
- run id: run_001
- model: skipgram
- variant: hierarchical softmax
- embedding size: 70
- context size: 4
- epochs: 3
- embedding file: run_001__text8__skipgram__hierarchical_softmax__ctx4__dim70__ep3.txt

## Nearest Neighbours
A nearest-neighbour search starts with a query word, and for that query word, 
the model compares its word vector with every other learned word vector using cosine similarity.
The words with the highest cosine similarity are displayed below.
If the model learned meaningful structure, the displayed words should be related.

### anarchism
- anarcho (0.8091)
- libertarianism (0.7962)
- individualist (0.7545)
- anarchist (0.7404)
- liberalism (0.7338)

### originated
- emerged (0.6801)
- existed (0.6548)
- canon (0.6322)
- evolved (0.5920)
- originating (0.5724)

### as
- is (0.5474)
- reasserts (0.5192)
- by (0.5167)
- almogavars (0.5163)
- krasin (0.5160)

### term
- however (0.5971)
- concept (0.5821)
- naevus (0.5732)
- perestroika (0.5707)
- when (0.5425)

### of
- and (0.5486)
- the (0.5325)
- masada (0.5196)
- symbolizing (0.5167)
- syrian (0.4962)

### abuse
- addiction (0.7187)
- painkillers (0.6913)
- involuntary (0.6839)
- consensual (0.6785)
- treatment (0.6399)

### first
- following (0.5767)
- new (0.5439)
- last (0.5285)
- third (0.5167)
- hanwell (0.5006)

### used
- utilized (0.7305)
- treated (0.6744)
- regarded (0.6698)
- labeled (0.6650)
- seen (0.6549)

### against
- while (0.6128)
- with (0.5692)
- non (0.5562)
- nazi (0.5507)
- towards (0.5473)

### early
- late (0.6711)
- modern (0.5627)
- eight (0.5190)
- evangelizations (0.4972)
- miseries (0.4712)


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
This section is for `custom_analogies.txt`, which is my own analogy file. I followed the original analogy file format and created a few basic questions that I considered suitable. More analogy questions could be added in the same format if desired.

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
- correct: **1**
- accuracy: **0.2000**

### Summary by analogy type

**Semantic**
- questions in file: **5**
- questions asked: **5**
- skipped: **0**
- correct: **1**
- accuracy: **0.2000**

**Syntactic**
- questions in file: **4**
- questions asked: **4**
- skipped: **0**
- correct: **0**
- accuracy: **0.0000**

**Overall**
- questions in file: **9**
- questions asked: **9**
- skipped: **0**
- correct: **1**
- accuracy: **0.1111**


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
- questions asked: **506**
- skipped: **0**
- correct: **81**
- accuracy: **0.1601**

#### capital-world
- questions in category: **4524**
- questions asked: **4216**
- skipped: **308**
- correct: **248**
- accuracy: **0.0588**

#### currency
- questions in category: **866**
- questions asked: **866**
- skipped: **0**
- correct: **10**
- accuracy: **0.0115**

#### city-in-state
- questions in category: **2467**
- questions asked: **2467**
- skipped: **0**
- correct: **170**
- accuracy: **0.0689**

#### family
- questions in category: **506**
- questions asked: **506**
- skipped: **0**
- correct: **80**
- accuracy: **0.1581**

#### gram1-adjective-to-adverb
- questions in category: **992**
- questions asked: **992**
- skipped: **0**
- correct: **14**
- accuracy: **0.0141**

#### gram2-opposite
- questions in category: **812**
- questions asked: **812**
- skipped: **0**
- correct: **23**
- accuracy: **0.0283**

#### gram3-comparative
- questions in category: **1332**
- questions asked: **1332**
- skipped: **0**
- correct: **161**
- accuracy: **0.1209**

#### gram4-superlative
- questions in category: **1122**
- questions asked: **992**
- skipped: **130**
- correct: **33**
- accuracy: **0.0333**

#### gram5-present-participle
- questions in category: **1056**
- questions asked: **1056**
- skipped: **0**
- correct: **48**
- accuracy: **0.0455**

#### gram6-nationality-adjective
- questions in category: **1599**
- questions asked: **1599**
- skipped: **0**
- correct: **287**
- accuracy: **0.1795**

#### gram7-past-tense
- questions in category: **1560**
- questions asked: **1560**
- skipped: **0**
- correct: **56**
- accuracy: **0.0359**

#### gram8-plural
- questions in category: **1332**
- questions asked: **1332**
- skipped: **0**
- correct: **133**
- accuracy: **0.0998**

#### gram9-plural-verbs
- questions in category: **870**
- questions asked: **870**
- skipped: **0**
- correct: **33**
- accuracy: **0.0379**

### Summary by analogy type

**Semantic**
- questions in file: **8869**
- questions asked: **8561**
- skipped: **308**
- correct: **589**
- accuracy: **0.0688**

**Syntactic**
- questions in file: **10675**
- questions asked: **10545**
- skipped: **130**
- correct: **788**
- accuracy: **0.0747**

**Overall**
- questions in file: **19544**
- questions asked: **19106**
- skipped: **438**
- correct: **1377**
- accuracy: **0.0721**
