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

## Main entry point
- `main.py` — provides a CLI for selecting dataset, model family, variant, context size, word embedding size, and number of training epochs

## Requirements
Install dependencies with:

```bash
pip install -r requirements.txt
```

Current `requirements.txt`:

```text
numpy
markdown-pdf
```

`markdown-pdf` is used to generate a .pdf report about the model's training results (more on that below).

## How to run

Run from the project root:

```bash
python3 main.py
```

Then provide the requested inputs in the CLI.

### 1. Dataset file name(s)

When prompted, enter one or more dataset file names from the `datasets/` folder.

Examples:

```text
shakespeare
```

or

```text
shakespeare, frankenstein, alice
```

### 2. Model family

Choose one of:
- `cbow`
- `skipgram`

### 3. Model variant

Choose one of:
- `basic` = full softmax
- `hs` = hierarchical softmax
- `ns` = negative sampling

### 4. Context size

This is the total number of words surrounding the target (or input) center word in a single training example.
- context size `4` means 2 words on the left and 2 words on the right

### 5. Embedding size

This is the dimensionality of the learned word embedding vectors.

### 6. Number of epochs

This is the number of full passes over the training data provided in this run.

## Reporting

After training, `src/reporting.py` prints nearest neighbors for 10 example words using cosine similarity.

The reported words are currently the first 10 words in vocabulary order.

Additionally, for the datasets `text8`, `wiki_train`, and `wiki2_small`, the reporting provides evaluation of semantic and syntactic analogy questions (read more about this in `results.md`).

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

Core data processing and reporting:
- `src/text_processing.py` — tokenization, vocabulary building, and other necessary text processing
- `src/reporting.py` — nearest-neighbor reporting and semantic / syntactic analogy evaluation
- `src/huffman_tree.py` — Huffman tree building for hierarchical softmax

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