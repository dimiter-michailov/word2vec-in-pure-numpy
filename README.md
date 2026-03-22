# Word2Vec in Pure NumPy
This repository contains the solution to the Word2Vec implementation task, only using NumPy for the core model functionality.

Implementations were created for both standard models:
- CBOW and Skipgram

and for both of them also included the following implementation variants:
- with standard softmax
- with hierarchical softmax
- with negative sampling

In total, the repository contains the following six variants, located in `src/cbow` and `src/skipgram`:
1. standard CBOW — `src/cbow/cbow.py`
2. hierarchical CBOW — `src/cbow/hierarchical_cbow.py`
3. negative-sampling CBOW — `src/cbow/negative_cbow.py`
4. standard Skipgram — `src/skipgram/skipgram.py`
5. hierarchical Skipgram — `src/skipgram/hierarchical_skipgram.py`
6. negative-sampling Skipgram — `src/skipgram/negative_skipgram.py`

## Main entry point(s)
- `src/main.py` — provides a CLI for selecting the execution workflow, dataset, model family, model variant, context size, word embedding size, and number of training epochs

## Requirements
Install dependencies with:

```bash
pip install -r requirements.txt
```

Current `requirements.txt`:

```text
numpy
```

This repository also uses Git LFS for large files.

To install Git LFS run:
```bash
git lfs install
```

Git LFS is used for large dataset files and can also be used for saved embedding files tracked in `saved_embeddings/`.

## How to run the code

Run from the project root:

```bash
python3 src/main.py
```

Then provide the requested inputs in the CLI.

### 0. Select workflow

Choose one of:
- `1` = train a new model configuration (standard run)
- `2` = generate report from saved embeddings (`.txt`)

Option `2` lets you choose any saved embedding file from the `saved_embeddings/` folder.

### 1. Dataset file name(s)

When prompted, select one or more dataset files from the `datasets/` folder by number.

### 2. Model family

Choose one of:
- `1` = `cbow`
- `2` = `skipgram`

### 3. Model variant

Choose one of:
- `1` = standard (no optimizations)
- `2` = hierarchical softmax
- `3` = negative sampling

### 4. Context size

This is the total number of words surrounding the target (or input) center word in a single training example.
Only even numbers are accepted.
- context size `4` means 2 words on the left and 2 words on the right

### 5. Embedding size

This is the dimensionality of the learned word embedding vectors.

### 6. Number of epochs

This is the number of full passes over the training data provided in this run.

## Reporting

After training, the learned word embeddings (`input_hidden_matrix`) are saved in `saved_embeddings/` as `.txt` files in standard word2vec text format.

Furthermore, the `src/reporting.py` file prints in the console nearest neighbors for 10 example words using cosine similarity.

The reported words are currently the first 10 words in the vocabulary.

Additionally, the reporting can provide evaluation of semantic and syntactic analogy questions (read more about this in `results.md`).

The exact analogy questions are located in:
- `analogy_questions/custom_analogies.txt`
- `analogy_questions/google_analogies.txt`

To use additional analogy files, the current `main.py` logic needs to be changed.

A `scoreboard.csv` file is also maintained for fresh training runs. It stores a run summary together with the main analogy evaluation results. Nearest-neighbors are not included in the scoreboard.

## Expected run times
The following timing results were recorded for 50,000 training samples with:
- context size = 4
- embedding size = 80

**CBOW**
- Basic CBOW: 109.77 sec
- CBOW with Hierarchical Softmax: 5.63 sec
- CBOW with Negative Sampling: 8.42 sec

**Skipgram**
- Basic Skipgram: 101.45 sec
- Skipgram with Hierarchical Softmax: 15.49 sec
- Skipgram with Negative Sampling: 22.39 sec

## Other files in the repository

Core data processing, reporting, and run storage:
- `src/text_processing.py` — tokenization, vocabulary building, and other necessary text processing
- `src/reporting.py` — nearest-neighbor reporting, analogy evaluation, and embedding `.txt` loading/saving
- `src/run_storage.py` — saved embedding file management and scoreboard storage
- `src/huffman_tree.py` — Huffman tree building for hierarchical softmax

Saved outputs:
- `saved_embeddings/` — saved word embedding files in standard word2vec text format
- `scoreboard.csv` — run summaries and main analogy evaluation results
- `results.md` — full report from the current execution

Tests:
- `tests/`

Run all tests from the project root with:

```bash
python3 -m unittest discover -s tests -v
```

## Main references

1. Mikolov, Tomas, Kai Chen, Greg Corrado, and Jeffrey Dean.  
   *Efficient Estimation of Word Representations in Vector Space*.  
   arXiv:1301.3781, 2013.

2. Mikolov, Tomas, Ilya Sutskever, Kai Chen, Greg S. Corrado, and Jeffrey Dean.  
   *Distributed Representations of Words and Phrases and their Compositionality*.  
   NeurIPS, 2013.

3. Xin Rong.  
   *word2vec Parameter Learning Explained*.  
   Main reference used for gradient derivations, parameter updates, and clarification of the Word2Vec training equations.

## Dataset sources

The datasets used in this repository were taken from the following sources.

**Kaggle**
- WikiText-103:  
  `https://www.kaggle.com/datasets/vadimkurochkin/wikitext-103?resource=download`

- text8 word embedding dataset:  
  `https://www.kaggle.com/datasets/gupta24789/text8-word-embedding`

- WikiText-2 reference / EDA page:  
  `https://www.kaggle.com/code/harshpraharaj98/wikitext2-eda`

**Other text sources**
- Shakespeare:  
  `https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt`
- *Alice’s Adventures in Wonderland* — Project Gutenberg
- *Frankenstein* — Project Gutenberg

To use a new dataset, add the `.txt` file to `datasets/` and provide its name when running `main.py`.
