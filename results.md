# Word2Vec Results

## Current run
- Dataset(s) used in this run: `loaded from run_003__text8__skipgram__negative_sampling__ctx4__dim90__ep3.txt`
- Vocabulary size: `253854`
- Number of tokens produced from the dataset(s): `N/A`

### Parameters used in this run
- run id: run_003
- embedding file: run_003__text8__skipgram__negative_sampling__ctx4__dim90__ep3.txt
- embedding size: 90

## Nearest Neighbours
A nearest-neighbour search starts with a query word, and for that query word, 
the model compares its word vector with every other learned word vector using cosine similarity.
The words with the highest cosine similarity are displayed below.
If the model learned meaningful structure, the displayed words should be related.

### anarchism
- libertarianism (0.8284)
- anarcho (0.8249)
- individualist (0.8099)
- collectivism (0.7928)
- marxism (0.7859)

### originated
- emerged (0.7569)
- originates (0.7410)
- originating (0.7128)
- arose (0.6990)
- existed (0.6970)

### as
- reconstituted (0.6914)
- superpower (0.6824)
- insofar (0.6783)
- such (0.6648)
- constituting (0.6524)

### term
- coining (0.7282)
- phrase (0.7167)
- neologism (0.7153)
- celt (0.7064)
- isolationism (0.7036)

### of
- bloodiest (0.7231)
- pdpa (0.7170)
- reconquest (0.7111)
- forfeiting (0.7076)
- promulgation (0.7033)

### abuse
- harassment (0.7510)
- masturbation (0.7453)
- incest (0.7380)
- manslaughter (0.7309)
- negligence (0.7305)

### first
- last (0.7470)
- second (0.7192)
- iala (0.7098)
- third (0.7052)
- homerun (0.7016)

### used
- employed (0.8192)
- utilized (0.7958)
- invoked (0.7247)
- touted (0.7069)
- done (0.7035)

### against
- unleashing (0.6941)
- loathed (0.6713)
- incite (0.6677)
- waged (0.6661)
- instigated (0.6657)

### early
- late (0.7943)
- seventies (0.6733)
- formative (0.6638)
- earliest (0.6589)
- palaeolithic (0.6561)


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

#### syntactic_comparative
- questions in category: **5**
- questions asked: **5**
- skipped: **0**
- correct: **2**
- accuracy: **0.4000**

#### semantic_family
- questions in category: **5**
- questions asked: **5**
- skipped: **0**
- correct: **3**
- accuracy: **0.6000**

#### syntactic_agent_noun
- questions in category: **5**
- questions asked: **5**
- skipped: **0**
- correct: **0**
- accuracy: **0.0000**

#### semantic_person_place
- questions in category: **5**
- questions asked: **5**
- skipped: **0**
- correct: **0**
- accuracy: **0.0000**

### Summary by analogy type

**Semantic**
- questions in file: **10**
- questions asked: **10**
- skipped: **0**
- correct: **3**
- accuracy: **0.3000**

**Syntactic**
- questions in file: **10**
- questions asked: **10**
- skipped: **0**
- correct: **2**
- accuracy: **0.2000**

**Overall**
- questions in file: **20**
- questions asked: **20**
- skipped: **0**
- correct: **5**
- accuracy: **0.2500**


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
- correct: **156**
- accuracy: **0.3083**

#### capital-world
- questions in category: **4524**
- questions asked: **4216**
- skipped: **308**
- correct: **364**
- accuracy: **0.0863**

#### currency
- questions in category: **866**
- questions asked: **866**
- skipped: **0**
- correct: **23**
- accuracy: **0.0266**

#### city-in-state
- questions in category: **2467**
- questions asked: **2467**
- skipped: **0**
- correct: **413**
- accuracy: **0.1674**

#### family
- questions in category: **506**
- questions asked: **506**
- skipped: **0**
- correct: **220**
- accuracy: **0.4348**

#### gram1-adjective-to-adverb
- questions in category: **992**
- questions asked: **992**
- skipped: **0**
- correct: **34**
- accuracy: **0.0343**

#### gram2-opposite
- questions in category: **812**
- questions asked: **812**
- skipped: **0**
- correct: **56**
- accuracy: **0.0690**

#### gram3-comparative
- questions in category: **1332**
- questions asked: **1332**
- skipped: **0**
- correct: **497**
- accuracy: **0.3731**

#### gram4-superlative
- questions in category: **1122**
- questions asked: **992**
- skipped: **130**
- correct: **119**
- accuracy: **0.1200**

#### gram5-present-participle
- questions in category: **1056**
- questions asked: **1056**
- skipped: **0**
- correct: **166**
- accuracy: **0.1572**

#### gram6-nationality-adjective
- questions in category: **1599**
- questions asked: **1599**
- skipped: **0**
- correct: **477**
- accuracy: **0.2983**

#### gram7-past-tense
- questions in category: **1560**
- questions asked: **1560**
- skipped: **0**
- correct: **234**
- accuracy: **0.1500**

#### gram8-plural
- questions in category: **1332**
- questions asked: **1332**
- skipped: **0**
- correct: **271**
- accuracy: **0.2035**

#### gram9-plural-verbs
- questions in category: **870**
- questions asked: **870**
- skipped: **0**
- correct: **151**
- accuracy: **0.1736**

### Summary by analogy type

**Semantic**
- questions in file: **8869**
- questions asked: **8561**
- skipped: **308**
- correct: **1176**
- accuracy: **0.1374**

**Syntactic**
- questions in file: **10675**
- questions asked: **10545**
- skipped: **130**
- correct: **2005**
- accuracy: **0.1901**

**Overall**
- questions in file: **19544**
- questions asked: **19106**
- skipped: **438**
- correct: **3181**
- accuracy: **0.1665**
