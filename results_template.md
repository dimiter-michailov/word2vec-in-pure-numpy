# Word2Vec Results

## Current run
- Dataset(s) used in this run: `{{dataset_name}}`
- Vocabulary size: `{{vocab_size}}`
- Number of tokens produced from the dataset(s): `{{token_count}}`

### Parameters used in this run
{{parameters_text}}

## Nearest Neighbours
A nearest-neighbour search starts with a query word, and for that query word, 
the model compares its word vector with every other learned word vector using cosine similarity.
The words with the highest cosine similarity are displayed below.
If the model learned meaningful structure, the displayed words should be related.

{{neighbors_text}}

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

{{custom_results_text}}

## Google analogy questions
This section is for `google_analogies.txt`.

These are the standard analogy test questions discussed in the original word2vec paper and used to evaluate whether learned word vectors capture semantic and syntactic relationships.

When this evaluation is run, the report shows:
- per-category results
- an overall semantic accuracy
- an overall syntactic accuracy

{{google_results_text}}